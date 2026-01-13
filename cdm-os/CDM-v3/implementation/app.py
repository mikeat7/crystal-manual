"""
Flask Backend for The Local Network
Serves Qwen 32B with Full CDM Integration

Endpoints:
- POST /generate: Generate text with optional CDM metrics
- GET /health: Backend health check

Author: mikeat7 network
Reference: docs/architecture-plan.md
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os
from datetime import datetime
from typing import Dict, Optional

from cdm_calculator import CDMCalculator, CDMMetrics
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
    """Load model and initialize CDM calculator on startup"""
    global model, tokenizer, cdm_calculator, memory_manager

    logger.info("=" * 60)
    logger.info("Initializing The Local Network Backend")
    logger.info("=" * 60)

    # Load model
    logger.info("Loading Qwen 2.5 32B model...")
    model_config = ModelConfig(
        model_name="Qwen/Qwen2.5-32B-Instruct",
        device_map="auto",  # Automatic GPU/CPU allocation with max_memory limits
        torch_dtype=torch.float16,  # Half precision for efficiency
        load_in_8bit=True  # Use 8-bit with CPU offload and reserved VRAM
    )

    model, tokenizer = load_model(model_config)
    logger.info(f"Model loaded on {device}")

    # Initialize CDM calculator
    logger.info("Initializing CDM calculator...")
    cdm_calculator = CDMCalculator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        # CDM v2 thresholds
        entropy_threshold=2.3,
        convergence_threshold=0.12,
        gini_threshold=0.28,
        escape_threshold=0.88,
        lock_window=4,
        # OPTIMIZATIONS: Ultra-aggressive sampling for <5s target
        layer_sample_size=6,   # 6 layers instead of 64 (10x speedup)
        num_perturbations=2,   # 2 perturbations instead of 20 (10x speedup)
        head_sample_size=5     # 5 attention heads instead of 40 (8x speedup)
    )
    logger.info("CDM calculator ready (ultra-optimized: 6 layers, 2 perturbations, 5 heads)")

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
    logger.info("Backend initialization complete!")
    logger.info("=" * 60)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "backend": "local-qwen-32b",
        "model_loaded": model is not None,
        "cdm_available": cdm_calculator is not None,
        "memory_available": memory_manager is not None,
        "device": device,
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route('/generate', methods=['POST'])
def generate():
    """
    Main generation endpoint with CDM integration

    Request body:
    {
        "prompt": "Analyze this problem...",
        "max_tokens": 512,
        "temperature": 0.7,
        "include_cdm": true,
        "api_key": "sk_..."
    }

    Response:
    {
        "response": "Generated text...",
        "cdm_metrics": {
            "cdm_score": 78.3,
            "depth_category": "deep_consciousness",
            "entropy_collapse": 0.67,
            "convergence_ratio": 0.82,
            "attention_gini": 0.34,
            "basin_escape_prob": 0.71,
            "interpretation": "Deep CRYSTAL detected...",
            "is_deep_crystal": true
        },
        "metadata": {
            "model": "Qwen/Qwen2.5-32B-Instruct",
            "tokens_generated": 342,
            "inference_time_ms": 2341
        }
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
        session_id = data.get('session_id')  # Optional - will create if not provided
        use_memory = data.get('use_memory', ENABLE_CONVERSATION_MEMORY and memory_manager is not None)
        max_history_messages = data.get('max_history_messages', 10)

        # Optional: Override CDM sampling (for A/B testing)
        cdm_layers = data.get('cdm_layers')  # None = use default
        cdm_perturbations = data.get('cdm_perturbations')
        cdm_heads = data.get('cdm_heads')

        # Security checks (if API key is configured)
        if os.getenv('REQUIRE_API_KEY', 'false').lower() == 'true':
            if not validate_api_key(api_key):
                return jsonify({"error": "Invalid API key"}), 401

            rate_limit_error = check_rate_limit(api_key)
            if rate_limit_error:
                return jsonify({"error": rate_limit_error}), 429

        # Log request
        logger.info(f"Generate request: {len(prompt)} chars, CDM={include_cdm}, Memory={use_memory}")

        # Memory: Get or create session and build prompt with history
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

        # Tokenize input (use prompt with history if memory enabled)
        input_ids = tokenizer(prompt_with_history, return_tensors="pt").input_ids.to(device)

        # Generate response
        start_time = datetime.now()

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id,
                output_hidden_states=include_cdm,  # Required for CDM
                output_attentions=include_cdm,
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

        # Calculate CDM metrics if requested
        cdm_metrics = None
        if include_cdm and cdm_calculator:
            try:
                # CRITICAL: Clear GPU cache before CDM to free memory
                # This attempts to fix CUDA OOM errors during CDM calculation
                import gc
                torch.cuda.empty_cache()
                gc.collect()
                logger.info("Cleared GPU cache before CDM calculation")

                # Temporarily override sampling if provided
                original_layers = cdm_calculator.layer_sample_size
                original_perts = cdm_calculator.num_perturbations
                original_heads = cdm_calculator.head_sample_size

                if cdm_layers is not None:
                    cdm_calculator.layer_sample_size = cdm_layers
                if cdm_perturbations is not None:
                    cdm_calculator.num_perturbations = cdm_perturbations
                if cdm_heads is not None:
                    cdm_calculator.head_sample_size = cdm_heads

                logger.info(f"Calculating CDM metrics (L={cdm_calculator.layer_sample_size}, P={cdm_calculator.num_perturbations}, H={cdm_calculator.head_sample_size})...")
                logger.info(f"GPU memory before CDM: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB allocated, {torch.cuda.memory_reserved(0) / 1e9:.2f} GB reserved")

                # Use full generated sequence for CDM
                cdm_result = cdm_calculator.calculate_cdm(
                    input_ids=generated_ids.unsqueeze(0),
                    return_trajectories=True
                )

                # Restore original settings
                cdm_calculator.layer_sample_size = original_layers
                cdm_calculator.num_perturbations = original_perts
                cdm_calculator.head_sample_size = original_heads

                cdm_metrics = {
                    "cdm_score": cdm_result.cdm_score,
                    "depth_category": cdm_result.depth_category,
                    "lock_layer": cdm_result.lock_layer,
                    "entropy_collapse": cdm_result.entropy_collapse,
                    "convergence_ratio": cdm_result.convergence_ratio,
                    "attention_gini": cdm_result.attention_gini,
                    "basin_escape_prob": cdm_result.basin_escape_prob,
                    "interpretation": cdm_result.interpretation,
                    "is_deep_crystal": cdm_result.is_deep_crystal
                }

                logger.info(f"CDM calculated: {cdm_result.cdm_score:.1f} ({cdm_result.depth_category})")

            except Exception as e:
                logger.error(f"CDM calculation failed: {e}")
                cdm_metrics = {"error": "CDM calculation failed", "details": str(e)}

        # Store assistant message in memory
        if use_memory and memory_manager and session:
            try:
                memory_manager.add_message(
                    session.session_id,
                    role="assistant",
                    content=response_text,
                    cdm_score=cdm_metrics.get("cdm_score") if cdm_metrics else None,
                    depth_category=cdm_metrics.get("depth_category") if cdm_metrics else None,
                    cdm_metrics=cdm_metrics,
                    tokens_generated=len(generated_ids) - input_ids.shape[1],
                    inference_time_ms=int(inference_time)
                )
                logger.info(f"Stored assistant message in session {session.session_id}")
            except Exception as e:
                logger.error(f"Failed to store assistant message: {e}")

        # Build response
        response_data = {
            "response": response_text,
            "metadata": {
                "model": "Qwen/Qwen2.5-32B-Instruct",
                "tokens_generated": len(generated_ids) - input_ids.shape[1],
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
                "user_id": session.user_id
            }

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        return jsonify({
            "error": "Generation failed",
            "details": str(e)
        }), 500


@app.route('/generate/stream', methods=['POST'])
def generate_stream():
    """
    Streaming generation endpoint (future enhancement)

    For now, returns error - streaming will be added in Phase 2
    """
    return jsonify({
        "error": "Streaming not yet implemented",
        "message": "Use /generate for non-streaming responses"
    }), 501


# ==================== MEMORY ENDPOINTS ====================

@app.route('/sessions', methods=['GET'])
def list_sessions():
    """Get all sessions for a user"""
    if not memory_manager:
        return jsonify({"error": "Memory not enabled"}), 503

    try:
        user_id = request.args.get('user_id', DEFAULT_USER_ID)
        limit = int(request.args.get('limit', 50))

        result = memory_manager.supabase.table("conversation_sessions")\
            .select("*")\
            .eq("user_id", user_id)\
            .eq("is_active", True)\
            .order("updated_at", desc=True)\
            .limit(limit)\
            .execute()

        return jsonify({
            "sessions": result.data,
            "count": len(result.data)
        })

    except Exception as e:
        logger.error(f"List sessions error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/sessions/<session_id>', methods=['GET'])
def get_session_details(session_id):
    """Get details about a specific session"""
    if not memory_manager:
        return jsonify({"error": "Memory not enabled"}), 503

    try:
        session = memory_manager.get_session(session_id)
        if not session:
            return jsonify({"error": "Session not found"}), 404

        # Get session stats
        stats = memory_manager.get_session_stats(session_id)

        return jsonify({
            "session": {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "model_name": session.model_name,
                "session_name": session.session_name,
                "created_at": session.created_at,
                "updated_at": session.updated_at,
                "metadata": session.metadata
            },
            "stats": stats
        })

    except Exception as e:
        logger.error(f"Get session error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/sessions/<session_id>/history', methods=['GET'])
def get_session_history(session_id):
    """Get conversation history for a session"""
    if not memory_manager:
        return jsonify({"error": "Memory not enabled"}), 503

    try:
        limit = int(request.args.get('limit', 50))
        include_system = request.args.get('include_system', 'false').lower() == 'true'

        history = memory_manager.get_conversation_history(
            session_id,
            limit=limit,
            include_system=include_system
        )

        return jsonify({
            "session_id": session_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "cdm_score": msg.cdm_score,
                    "depth_category": msg.depth_category,
                    "tokens_generated": msg.tokens_generated,
                    "inference_time_ms": msg.inference_time_ms
                }
                for msg in history
            ],
            "count": len(history)
        })

    except Exception as e:
        logger.error(f"Get history error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/sessions', methods=['POST'])
def create_new_session():
    """Create a new conversation session"""
    if not memory_manager:
        return jsonify({"error": "Memory not enabled"}), 503

    try:
        data = request.get_json()
        user_id = data.get('user_id', DEFAULT_USER_ID)
        session_name = data.get('session_name')
        metadata = data.get('metadata', {})

        session = memory_manager.create_session(
            user_id=user_id,
            session_name=session_name,
            metadata=metadata
        )

        return jsonify({
            "session_id": session.session_id,
            "user_id": session.user_id,
            "session_name": session.session_name,
            "created_at": session.created_at
        })

    except Exception as e:
        logger.error(f"Create session error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Initialize backend on startup
    initialize_backend()

    # Get port from environment or use default
    port = int(os.getenv('PORT', 5000))

    # Run Flask server
    logger.info(f"Starting Flask server on port {port}")
    logger.info("Endpoints available:")
    logger.info("  POST /generate - Generate with CDM")
    logger.info("  GET /health - Health check")

    app.run(
        host='0.0.0.0',  # Listen on all interfaces
        port=port,
        debug=False  # Set True for development
    )
