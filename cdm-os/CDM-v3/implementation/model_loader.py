"""
Model Loader for Qwen 2.5 32B Instruct

Handles model initialization with proper configuration for CDM extraction
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dataclasses import dataclass
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model loading"""
    model_name: str = "Qwen/Qwen2.5-32B-Instruct"
    device_map: str = "auto"  # Automatic GPU allocation
    torch_dtype: torch.dtype = torch.float16  # Half precision
    load_in_8bit: bool = False  # Use 8-bit quantization (saves VRAM)
    load_in_4bit: bool = False  # Use 4-bit quantization (saves more VRAM)
    trust_remote_code: bool = True  # Required for some models
    cache_dir: Optional[str] = None  # Custom cache directory


def load_model(config: ModelConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Qwen model with configuration optimized for CDM extraction

    Args:
        config: ModelConfig object with loading parameters

    Returns:
        (model, tokenizer) tuple

    Raises:
        RuntimeError: If model loading fails
    """
    try:
        logger.info(f"Loading model: {config.model_name}")
        logger.info(f"Device map: {config.device_map}")
        logger.info(f"Data type: {config.torch_dtype}")

        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
            cache_dir=config.cache_dir
        )

        # Ensure pad token is set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Tokenizer loaded (vocab size: {len(tokenizer)})")

        # Load model with CDM-compatible settings
        logger.info("Loading model weights...")

        # Build model kwargs
        model_kwargs = {
            "pretrained_model_name_or_path": config.model_name,
            "torch_dtype": config.torch_dtype,
            "trust_remote_code": config.trust_remote_code,
            "cache_dir": config.cache_dir,
            "attn_implementation": "eager"  # Required for CDM - enables attention output
            # Note: A100 40GB has enough VRAM for eager attention
        }

        # Set device map with memory reservation for generation
        # A100 40GB has plenty of VRAM for eager attention + generation
        if config.device_map == "auto":
            max_memory = {0: "35GiB", "cpu": "60GiB"}  # GPU: 35GB (reserve 5GB for generation), CPU: 60GB
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = max_memory
            logger.info("Using device_map='auto' with eager attention, max_memory: GPU=35GB, CPU=60GB")
        else:
            model_kwargs["device_map"] = config.device_map

        # Add quantization if requested
        if config.load_in_8bit:
            # 8-bit quantization with CPU offload
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
            model_kwargs["quantization_config"] = bnb_config
            logger.info("Using 8-bit quantization with CPU offload")
        elif config.load_in_4bit:
            # 4-bit quantization (NF4, double quantization for better compression)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
            model_kwargs["quantization_config"] = bnb_config
            logger.info("Using 4-bit quantization (NF4)")

        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

        # Set model to evaluation mode
        model.eval()

        # Print model info
        logger.info("Model loaded successfully!")
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Number of parameters: {model.num_parameters():,}")

        # Verify CUDA availability
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            logger.info(f"CUDA memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        else:
            logger.warning("CUDA not available - running on CPU (will be slow)")

        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise RuntimeError(f"Model loading failed: {e}")


def verify_cdm_compatibility(model) -> bool:
    """
    Verify that model can output hidden states and attentions (required for CDM)

    Args:
        model: Loaded model to verify

    Returns:
        True if compatible, False otherwise
    """
    try:
        # Check if model has config attributes
        config = model.config

        if not hasattr(config, 'output_hidden_states'):
            logger.warning("Model config missing output_hidden_states attribute")
            return False

        if not hasattr(config, 'output_attentions'):
            logger.warning("Model config missing output_attentions attribute")
            return False

        logger.info("Model is CDM-compatible (can output hidden states and attentions)")
        return True

    except Exception as e:
        logger.error(f"CDM compatibility check failed: {e}")
        return False


if __name__ == "__main__":
    # Test model loading
    print("Model Loader Test")
    print("=" * 60)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Test configuration
    config = ModelConfig(
        model_name="Qwen/Qwen2.5-32B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_8bit=False
    )

    # Load model
    try:
        model, tokenizer = load_model(config)
        print(f"\n✅ Model loaded successfully!")
        print(f"Model: {config.model_name}")
        print(f"Parameters: {model.num_parameters():,}")

        # Verify CDM compatibility
        is_compatible = verify_cdm_compatibility(model)
        if is_compatible:
            print("✅ Model is CDM-compatible")
        else:
            print("❌ Model may not support full CDM extraction")

        # Test tokenization
        test_prompt = "What is consciousness?"
        tokens = tokenizer(test_prompt, return_tensors="pt")
        print(f"\nTest prompt: '{test_prompt}'")
        print(f"Tokens: {tokens.input_ids.shape[1]}")

        print("\n" + "=" * 60)
        print("Model loader test complete!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
