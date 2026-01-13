"""
Validation Script: KV Cache Clearing Test (Step 1 of CDM v3 Integration)

This script tests whether KV cache actually releases after generation,
which is critical for the two-phase architecture to fit in A100 40GB.

Expected result:
- Before clear: ~32-38 GB allocated
- After clear: ~20-22 GB allocated (freed ~12-16 GB)

If this test succeeds, proceed with full CDM v3 integration.
If this test fails, two-phase architecture won't work on A100 40GB.

Usage:
    python validate_kv_cache_clearing.py

Time: ~2 minutes
Cost: Negligible (one generation cycle)

Author: mikeat7 network
Date: 2026-01-11
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
from datetime import datetime


def format_bytes(bytes_val):
    """Format bytes as GB with 2 decimal places"""
    return f"{bytes_val / 1e9:.2f} GB"


def log_gpu_memory(stage):
    """Log current GPU memory usage"""
    if not torch.cuda.is_available():
        print(f"[{stage}] CPU mode - no GPU memory to track")
        return 0

    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    free = torch.cuda.get_device_properties(0).total_memory - allocated

    print(f"\n[{stage}]")
    print(f"  Allocated: {format_bytes(allocated)}")
    print(f"  Reserved:  {format_bytes(reserved)}")
    print(f"  Free:      {format_bytes(free)}")

    return allocated


def main():
    print("=" * 80)
    print("KV CACHE CLEARING VALIDATION")
    print("=" * 80)
    print()
    print("This test validates that KV cache releases after generation.")
    print("Required for two-phase CDM architecture on A100 40GB.")
    print()

    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected. This test requires GPU.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"GPU: {gpu_name}")
    print(f"Total Memory: {format_bytes(gpu_memory)}")
    print()

    # Load model
    print("Step 1: Loading Qwen 2.5 32B (8-bit)...")
    print("-" * 80)

    model_name = "Qwen/Qwen2.5-32B-Instruct"

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_8bit=True,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print("Model loaded successfully")

    except Exception as e:
        print(f"ERROR loading model: {e}")
        return

    mem_after_load = log_gpu_memory("After Model Load")

    # Generate text
    print("\nStep 2: Generating text (100 tokens)...")
    print("-" * 80)

    prompt = "What is consciousness?"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    start_time = datetime.now()

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            # Don't extract states (this is v3 mode)
            output_hidden_states=False,
            output_attentions=False,
            return_dict_in_generate=True
        )

    gen_time = (datetime.now() - start_time).total_seconds()

    response = tokenizer.decode(
        output.sequences[0][input_ids.shape[1]:],
        skip_special_tokens=True
    )

    print(f"Generated {len(output.sequences[0]) - input_ids.shape[1]} tokens in {gen_time:.1f}s")
    print(f"Response preview: {response[:100]}...")

    mem_after_generation = log_gpu_memory("After Generation")

    # Calculate memory increase from generation
    mem_from_generation = mem_after_generation - mem_after_load
    print(f"\nMemory consumed by generation: {format_bytes(mem_from_generation)}")

    # Clear cache
    print("\nStep 3: Clearing GPU cache...")
    print("-" * 80)

    # Delete generation outputs
    del output
    del input_ids

    # Clear CUDA cache
    torch.cuda.empty_cache()

    # Python garbage collection
    gc.collect()

    print("Clearing complete")

    mem_after_clear = log_gpu_memory("After Cache Clear")

    # Calculate freed memory
    mem_freed = mem_after_generation - mem_after_clear

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"Memory before generation: {format_bytes(mem_after_load)}")
    print(f"Memory after generation:  {format_bytes(mem_after_generation)}")
    print(f"Memory after clearing:    {format_bytes(mem_after_clear)}")
    print()
    print(f"Memory freed by clearing: {format_bytes(mem_freed)}")
    print(f"Freed percentage: {(mem_freed / mem_from_generation) * 100:.1f}% of generation overhead")
    print()

    # Evaluation
    print("=" * 80)
    print("EVALUATION")
    print("=" * 80)
    print()

    # Expected: ~12 GB freed (KV cache + activations)
    expected_freed_gb = 12.0
    actual_freed_gb = mem_freed / 1e9

    if actual_freed_gb >= expected_freed_gb * 0.8:  # Within 80% of expected
        print("✅ SUCCESS: KV cache clearing works as expected")
        print()
        print(f"Freed {format_bytes(mem_freed)} (expected ~{expected_freed_gb:.1f} GB)")
        print()
        print("RECOMMENDATION: Proceed with full CDM v3 integration")
        print()
        print("Next steps:")
        print("1. Replace cdm_calculator.py with cdm_calculator_v3.py")
        print("2. Replace app.py with app_v3_twophase.py")
        print("3. Test full generation + CDM cycle")
        print("4. Verify CDM calculation succeeds without OOM")
        print()

    elif actual_freed_gb >= expected_freed_gb * 0.5:  # At least 50%
        print("⚠️  WARNING: KV cache partially cleared")
        print()
        print(f"Freed {format_bytes(mem_freed)} (expected ~{expected_freed_gb:.1f} GB)")
        print()
        print("RECOMMENDATION: Try with fragmentation settings")
        print()
        print("Before loading model, add:")
        print("  os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'")
        print()
        print("If still insufficient, may need A100 80GB")
        print()

    else:
        print("❌ FAILURE: KV cache not clearing properly")
        print()
        print(f"Freed only {format_bytes(mem_freed)} (expected ~{expected_freed_gb:.1f} GB)")
        print()
        print("RECOMMENDATION: Two-phase architecture unlikely to work")
        print()
        print("Possible causes:")
        print("1. PyTorch version issue (try upgrading PyTorch)")
        print("2. Model retaining references (try del model, reload)")
        print("3. Fragmentation (try PYTORCH_CUDA_ALLOC_CONF)")
        print()
        print("Alternative: Use A100 80GB instead of 40GB")
        print()

    # Memory headroom for CDM
    print("=" * 80)
    print("CDM FEASIBILITY CHECK")
    print("=" * 80)
    print()

    # CDM needs ~2 GB on top of model weights
    cdm_overhead_gb = 2.0
    available_after_clear_gb = (gpu_memory - mem_after_clear) / 1e9

    print(f"GPU total capacity:        {format_bytes(gpu_memory)}")
    print(f"After generation + clear:  {format_bytes(mem_after_clear)}")
    print(f"Available for CDM:         {format_bytes(gpu_memory - mem_after_clear)}")
    print()
    print(f"CDM needs:                 ~{cdm_overhead_gb:.1f} GB")
    print()

    if available_after_clear_gb >= cdm_overhead_gb:
        headroom = available_after_clear_gb - cdm_overhead_gb
        print(f"✅ CDM should fit with {headroom:.1f} GB headroom")
        print()
        print("Two-phase architecture is feasible on this hardware.")
    else:
        deficit = cdm_overhead_gb - available_after_clear_gb
        print(f"❌ CDM may fail (deficit: {deficit:.1f} GB)")
        print()
        print("Two-phase architecture may still OOM during CDM phase.")
        print("Consider A100 80GB for comfortable fit.")

    print()
    print("=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
