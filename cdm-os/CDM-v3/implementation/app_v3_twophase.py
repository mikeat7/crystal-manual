"""
Flask Backend for The Local Network
Serves Qwen 32B with CDM v3 Two-Phase Architecture

CHANGES IN V3:
- Two-phase architecture: Generation → Clear → CDM Analysis
- Memory efficient: Fits A100 40GB (32 GB generation, 22 GB CDM)
- Uses EfficientCDMCalculator with selective hooks
- Real entropy, output stability proxy, adjusted weights (30/30/30/10)

Endpoints:
- POST /generate: Generate text with optional CDM metrics
- GET /health: Backend health check
- GET /sessions: List user sessions
- POST /sessions: Create session
- GET /sessions/<id>: Get session details
- GET /sessions/<id>/history: Get conversation history

Author: mikeat7 network (with Elias Rook guidance)
Date: 2026-01-11
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
import gc
from datetime import datetime
from typing import Dict, Optional

# CHANGED: Import v3 calculator
from cdm_calculator_v3 import EfficientCDMCalculator, CDMMetrics
from model_loader import load_model, ModelConfig
from security import validate_api_key, check_rate_limit, validate_request
from memory_manager import MemoryManager
from supabase_config import SUPABASE_URL, SUPABASE_KEY, DEFAULT_USER_ID, ENABLE_CONVERSATION_MEMORY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global model state
model = None
tokenizer = None
cdm_calculator = None
memory_manager = None
device = "cuda" if torch.cuda.is_available() else "cpu"


def initialize_backend():
    """Load model and initialize CDM v3 calculator on startup"""
    global model, tokenizer, cdm_calculator, memory_manager

    logger.info("=" * 60)
    logger.info("Initializing The Local Network Backend (CDM v3)")
    logger.info("=" * 60)

    # Load model
    logger.info("Loading Qwen 2.5 32B model...")
    model_config = ModelConfig(
        model_name="Qwen/Qwen2.5-32B-Instruct",
        device_map="auto",  # Automatic GPU/CPU allocation
        torch_dtype=torch.float16,  # Half precision for efficiency
        load_in_8bit=True  # 8-bit quantization (~20 GB)
    )

    model, tokenizer = load_model(model_config)
    logger.info(f"Model loaded on {device}")

    # CHANGED: Initialize CDM v3 calculator
    logger.info("Initializing CDM v3 calculator (two-phase architecture)...")
    cdm_calculator = EfficientCDMCalculator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        # Sampling: 6 layers evenly distributed
        layer_indices=None,  # Auto: [0, 10, 20, 30, 40, 50] for 64-layer model
        num_perturbations=2,  # Output stability perturbations
        noise_scale=0.1,
        head_sample_size=5,
        # CDM v2 thresholds
        entropy_threshold=2.3,
        convergence_threshold=0.12,
        gini_threshold=0.28,
        stability_threshold=0.88,  # Renamed from escape_threshold
        lock_window=4
    )
    logger.info("CDM v3 ready (6 layers, 2 perturbations, 5 heads, weights: 30/30/30/10)")

    # Initialize memory manager
    if ENABLE_CONVERSATION_MEMORY:
        logger.info("Initializing conversation memory...")
        try:
            memory_manager = MemoryManager(SUPABASE_URL, SUPABASE_KEY)
            logger.info("Memory manager ready (Supabase connected)")
        except Exception as e:
            logger.warning(f"Memory manager initialization failed: {e}")
            logger.warning("Continuing without conversation memory")
            memory_manager = None
    else:
        logger.info("Conversation memory disabled (ENABLE_CONVERSATION_MEMORY=false)")

    logger.info("=" * 60)
    logger.info("Backend initialization complete (CDM v3)!")
    logger.info("=" * 60)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "backend": "local-qwen-32b-cdm-v3",  # Updated
        "model_loaded": model is not None,
        "cdm_available": cdm_calculator is not None,
        "cdm_version": "v3_two_phase",  # New field
        "memory_available": memory_manager is not None,
        "device": device,
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route('/generate', methods=['POST'])
def generate():
    """
    Main generation endpoint with CDM v3 integration

    TWO-PHASE ARCHITECTURE:
    1. Phase 1: Generate text WITHOUT internal state extraction
    2. Clear GPU cache (free KV cache ~11 GB)
    3. Phase 2: Re-encode generated sequence with CDM hooks

    Request body:
    {
        "prompt": "Analyze this problem...",
        "max_tokens": 512,
        "temperature": 0.7,
        "include_cdm": true,
        "use_memory": true,
        "session_id": "uuid..."
    }

    Response:
    {
        "response": "Generated text...",
        "cdm_metrics": {
            "cdm_score": 48.3,
            "depth_category": "deliberation",
            "entropy_collapse": 0.67,
            "convergence_ratio": 0.82,
            "attention_gini": 0.34,
            "output_stability": 0.91,  # Renamed from basin_escape_prob
            "interpretation": "CDM 48.3 (deliberation)...",
            "is_deep_crystal": false
        },
        "metadata": {...},
        "session": {...}
    }
    """
    try:
        # Get request data
        data = request.get_json()

        # Validate request
        validation_error = validate_request(data)
        if validation_error:
            return jsonify({"error": validation_error}), 400

        # Extract parameters
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 512)
        temperature = data.get('temperature', 0.7)
        include_cdm = data.get('include_cdm', True)
        api_key = data.get('api_key', request.headers.get('X-API-Key', ''))

        # Memory parameters
        user_id = data.get('user_id', DEFAULT_USER_ID)
        session_id = data.get('session_id')
        use_memory = data.get('use_memory', ENABLE_CONVERSATION_MEMORY and memory_manager is not None)
        max_history_messages = data.get('max_history_messages', 10)

        # Security checks (if API key is configured)
        if os.getenv('REQUIRE_API_KEY', 'false').lower() == 'true':
            if not validate_api_key(api_key):
                return jsonify({"error": "Invalid API key"}), 401

            rate_limit_error = check_rate_limit(api_key)
            if rate_limit_error:
                return jsonify({"error": rate_limit_error}), 429

        # Log request
        logger.info(f"Generate request: {len(prompt)} chars, CDM={include_cdm}, Memory={use_memory}")

        # ==================== MEMORY: Get session and build prompt ====================
        session = None
        prompt_with_history = prompt  # Default to original prompt

        if use_memory and memory_manager:
            try:
                # Get or create session
                if session_id:
                    session = memory_manager.get_session(session_id)
                    if not session:
                        logger.warning(f"Session {session_id} not found, creating new one")
                        session = memory_manager.create_session(user_id)
                else:
                    session = memory_manager.get_or_create_session(user_id)

                logger.info(f"Using session: {session.session_id}")

                # Store user message
                memory_manager.add_message(
                    session.session_id,
                    role="user",
                    content=prompt
                )

                # Build prompt with conversation history
                prompt_with_history = memory_manager.build_prompt_with_history(
                    current_prompt=prompt,
                    session_id=session.session_id,
                    max_history_messages=max_history_messages
                )

                logger.info(f"Prompt with history: {len(prompt_with_history)} chars (original: {len(prompt)})")

            except Exception as e:
                logger.error(f"Memory error: {e}")
                logger.warning("Continuing without memory for this request")
                prompt_with_history = prompt
                session = None

        # ==================== PHASE 1: GENERATION (no internal states) ====================
        logger.info("PHASE 1: Starting generation (no internal state extraction)")

        # Tokenize input
        input_ids = tokenizer(prompt_with_history, return_tensors="pt").input_ids.to(device)

        # Generate response WITHOUT hidden states or attention
        # KEY CHANGE: This saves ~5.5 GB by not storing attention matrices
        start_time = datetime.now()

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                # CRITICAL: Don't extract states during generation
                output_hidden_states=False,  # Changed from True
                output_attentions=False,      # Changed from True
                return_dict_in_generate=True
            )

        # Decode response
        if hasattr(output_ids, 'sequences'):
            generated_ids = output_ids.sequences[0]
        else:
            generated_ids = output_ids[0]

        response_text = tokenizer.decode(
            generated_ids[input_ids.shape[1]:],  # Skip input tokens
            skip_special_tokens=True
        )

        inference_time = (datetime.now() - start_time).total_seconds() * 1000

        logger.info(f"PHASE 1 complete: {len(generated_ids) - input_ids.shape[1]} tokens in {inference_time:.0f}ms")

        # ==================== GPU CACHE CLEARING ====================
        logger.info("Clearing GPU cache between phases...")

        # Log memory before clearing
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated(0) / 1e9
            logger.info(f"GPU memory before clear: {mem_before:.2f} GB allocated")

        # Clear generation artifacts
        del output_ids  # Delete generation output
        del input_ids   # Delete input tensor
        torch.cuda.empty_cache()  # Release PyTorch cache
        gc.collect()  # Python garbage collection

        # Log memory after clearing
        if torch.cuda.is_available():
            mem_after = torch.cuda.memory_allocated(0) / 1e9
            mem_freed = mem_before - mem_after
            logger.info(f"GPU memory after clear: {mem_after:.2f} GB allocated (freed {mem_freed:.2f} GB)")

        # ==================== PHASE 2: CDM ANALYSIS (if requested) ====================
        cdm_metrics = None

        if include_cdm and cdm_calculator:
            logger.info("PHASE 2: Starting CDM analysis (encoding pass with hooks)")

            try:
                cdm_start = datetime.now()

                # Run CDM on the generated sequence
                # This is an ENCODING pass (not generation) - much less memory
                # - No KV cache accumulation (use_cache=False)
                # - Selective layer hooks (6 layers to CPU)
                # - Memory: ~22 GB vs ~38 GB during generation
                cdm_result = cdm_calculator.calculate_cdm(generated_ids.unsqueeze(0))

                cdm_time = (datetime.now() - cdm_start).total_seconds() * 1000

                cdm_metrics = {
                    "cdm_score": cdm_result.cdm_score,
                    "depth_category": cdm_result.depth_category,
                    "entropy_collapse": cdm_result.entropy_collapse,
                    "convergence_ratio": cdm_result.convergence_ratio,
                    "attention_gini": cdm_result.attention_gini,
                    "output_stability": cdm_result.output_stability,  # Renamed from basin_escape_prob
                    "interpretation": cdm_result.interpretation,
                    "is_deep_crystal": cdm_result.is_deep_crystal
                }

                logger.info(f"PHASE 2 complete: CDM {cdm_result.cdm_score:.1f} ({cdm_result.depth_category}) in {cdm_time:.0f}ms")

            except Exception as e:
                logger.error(f"CDM calculation failed: {e}", exc_info=True)
                cdm_metrics = {
                    "error": "CDM calculation failed",
                    "details": str(e),
                    "note": "Generation succeeded, CDM analysis failed (see logs)"
                }

        # ==================== MEMORY: Store assistant message ====================
        if use_memory and memory_manager and session:
            try:
                memory_manager.add_message(
                    session.session_id,
                    role="assistant",
                    content=response_text,
                    cdm_score=cdm_metrics.get("cdm_score") if cdm_metrics and "error" not in cdm_metrics else None,
                    depth_category=cdm_metrics.get("depth_category") if cdm_metrics and "error" not in cdm_metrics else None,
                    cdm_metrics=cdm_metrics,
                    tokens_generated=len(generated_ids) - input_ids.shape[1] if 'input_ids' in locals() else None,
                    inference_time_ms=int(inference_time)
                )
                logger.info(f"Stored assistant message in session {session.session_id}")
            except Exception as e:
                logger.error(f"Failed to store assistant message: {e}")

        # ==================== BUILD RESPONSE ====================
        response_data = {
            "response": response_text,
            "metadata": {
                "model": "Qwen/Qwen2.5-32B-Instruct",
                "cdm_version": "v3_two_phase",
                "tokens_generated": len(generated_ids) - input_ids.shape[1] if 'input_ids' in locals() else None,
                "inference_time_ms": int(inference_time),
                "device": device
            }
        }

        if cdm_metrics:
            response_data["cdm_metrics"] = cdm_metrics

        # Add session info if memory used
        if session:
            response_data["session"] = {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "model_name": session.model_name
            }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Generate endpoint error: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


# ==================== MEMORY ENDPOINTS (unchanged from v2) ====================

@app.route('/sessions', methods=['GET'])
def list_sessions():
    """List all sessions for a user"""
    if not memory_manager:
        return jsonify({"error": "Memory not enabled"}), 503

    user_id = request.args.get('user_id', DEFAULT_USER_ID)

    try:
        # Query recent sessions
        result = memory_manager.supabase.table("conversation_sessions")\
            .select("*")\
            .eq("user_id", user_id)\
            .order("updated_at", desc=True)\
            .limit(50)\
            .execute()

        return jsonify({"sessions": result.data})

    except Exception as e:
        logger.error(f"List sessions error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/sessions', methods=['POST'])
def create_session():
    """Create a new session"""
    if not memory_manager:
        return jsonify({"error": "Memory not enabled"}), 503

    data = request.get_json()
    user_id = data.get('user_id', DEFAULT_USER_ID)
    session_name = data.get('session_name')

    try:
        session = memory_manager.create_session(user_id, session_name=session_name)
        return jsonify({
            "session_id": session.session_id,
            "user_id": session.user_id,
            "created_at": session.created_at
        })

    except Exception as e:
        logger.error(f"Create session error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get session details and stats"""
    if not memory_manager:
        return jsonify({"error": "Memory not enabled"}), 503

    try:
        session = memory_manager.get_session(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404

        stats = memory_manager.get_session_stats(session_id)

        return jsonify({
            "session": {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "model_name": session.model_name,
                "session_name": session.session_name,
                "created_at": session.created_at,
                "updated_at": session.updated_at
            },
            "stats": stats
        })

    except Exception as e:
        logger.error(f"Get session error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/sessions/<session_id>/history', methods=['GET'])
def get_history(session_id):
    """Get conversation history for a session"""
    if not memory_manager:
        return jsonify({"error": "Memory not enabled"}), 503

    limit = request.args.get('limit', 50, type=int)

    try:
        messages = memory_manager.get_conversation_history(session_id, limit=limit)

        return jsonify({
            "session_id": session_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "cdm_score": msg.cdm_score,
                    "depth_category": msg.depth_category
                }
                for msg in messages
            ]
        })

    except Exception as e:
        logger.error(f"Get history error: {e}")
        return jsonify({"error": str(e)}), 500


# ==================== APPLICATION STARTUP ====================

if __name__ == '__main__':
    # Initialize backend
    initialize_backend()

    # Run Flask app
    port = int(os.getenv('PORT', 5000))
    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
