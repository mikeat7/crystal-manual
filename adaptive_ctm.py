# adaptive_ctm.py
from cdm import cdm_v2
from transformers import AutoTokenizer

def adaptive_think(model, tokenizer, prompt, target_cdm=72, max_think=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    think_token = tokenizer.encode(" <think>")[0] if "llama" in tokenizer.name_or_path.lower() else 32000

    generated = inputs.input_ids.clone()
    thinking = 0

    while thinking < max_think:
        cdm_value, label = cdm_v2(model, generated)
        if cdm_value >= target_cdm:
            break
        generated = torch.cat([generated, torch.tensor([[think_token]], device=model.device)], dim=1)
        thinking += 1

    # Now generate real answer
    output = model.generate(generated, max_new_tokens=512, do_sample=False)
    return tokenizer.decode(output[0], skip_special_tokens=True)
