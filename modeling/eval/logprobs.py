import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.trainer.utils import selective_log_softmax
from tqdm import tqdm
import pydra
import math
import statistics


class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        self.input_file: str = "generations/verifree_backup_500samples_ckpt_800.jsonl"
        self.output_file: str = "logprobs/verifree_backup_500samples_ckpt_800_logprobs.jsonl"
        self.sft: bool = False
        self.base: bool = False
        self.checkpoint_path: str = "/scr/agam/model_checkpoints/kanishk_modelling_dialogue/verifree_backup_checkpoint-800"
        self.device: str = "cuda:0"


def load_model_and_tokenizer(checkpoint_path, device):
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    return model, tokenizer


def calculate_logprob_for_example(example, model, tokenizer, device, sft, base):
    completion = example["completion"]
    dialogue_in_completion = '<dialogue>' in completion 
    dialogue_in_completion_newline = '<dialogue>\n' in completion

    if not sft and not dialogue_in_completion and not dialogue_in_completion_newline:
        return None
    
    if base:
            prefix_text = completion.split('<dialogue>')[0] + '<dialogue>'
    elif sft:
        prefix_text = None # directly predicts dialogue, no dialogue tags
    else:
        # order is important here
        if dialogue_in_completion_newline:
            prefix_text = completion.split('<dialogue>\n')[0] + '<dialogue>\n'

        elif dialogue_in_completion:
            prefix_text = completion.split('<dialogue>')[0] + '<dialogue>'

    if sft:
        # Build full conversation like SFTTrainer does to match tokenization exactly
        full_messages = example["prompt"] + [{"role": "assistant", "content": example["true_response"]}]
        full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        prompt_text = tokenizer.apply_chat_template(example["prompt"], tokenize=False, add_generation_prompt=True)
        
        full_ids = tokenizer.encode(full_text, add_special_tokens=False, return_tensors="pt")[0]
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")[0]
        
        # GT tokens are everything after the prompt
        gt_ids = full_ids[len(prompt_ids):]
        full_seq = full_ids.unsqueeze(0).to(device)
    else:
        gt_text = example["true_response"] + "\n</dialogue>"
        prompt_text = tokenizer.apply_chat_template(
            example["prompt"], 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")[0]
        gt_ids = tokenizer.encode(gt_text, add_special_tokens=False, return_tensors="pt")[0]
        
        if prefix_text is not None:
            prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False, return_tensors="pt")[0]
            full_seq = torch.cat([prompt_ids, prefix_ids, gt_ids]).unsqueeze(0).to(device)
        else:
            full_seq = torch.cat([prompt_ids, gt_ids]).unsqueeze(0).to(device)
    #breakpoint()
    with torch.no_grad():
        logits = model(input_ids=full_seq).logits
        logits = logits[:, :-1, :] # no normalization by temperature
        per_token_logps = selective_log_softmax(logits, full_seq[:, 1:])
        
        num_gt_tokens = len(gt_ids)
        gt_logps = per_token_logps[0, -num_gt_tokens:]
        sum_gt_logp = gt_logps.sum().item() 
        mean_gt_logp = gt_logps.mean().item()
        perplexity = math.exp(-mean_gt_logp)
    return {
        "mean_gt_logprob": mean_gt_logp,
        "sum_gt_logprob": sum_gt_logp, 
        "num_gt_tokens": num_gt_tokens,
        "perplexity": perplexity
    }


@pydra.main(Config)
def main(config: Config):
    print(f"Loading model from {config.checkpoint_path}")
    model, tokenizer = load_model_and_tokenizer(config.checkpoint_path, config.device)
    
    print(f"Reading input from {config.input_file}")
    with open(config.input_file, 'r') as f:
        examples = [json.loads(line) for line in f]
    
    os.makedirs(os.path.dirname(config.output_file), exist_ok=True)
    
    skipped_count = 0
    logprobs = []
    perplexities = []
    sum_gt_logprobs = []
    num_gt_tokens = []
    
    with open(config.output_file, 'w') as f:
        for example in tqdm(examples, desc="Calculating logprobs"):
            result = calculate_logprob_for_example(
                example, model, tokenizer, config.device, config.sft, config.base
            )
            
            if result is None:
                skipped_count += 1
                continue
            
            output_record = {**example, **result, "checkpoint_used": config.checkpoint_path}
            f.write(json.dumps(output_record) + "\n")
            logprobs.append(result["mean_gt_logprob"])
            perplexities.append(result["perplexity"])
            sum_gt_logprobs.append(result["sum_gt_logprob"])
            num_gt_tokens.append(result["num_gt_tokens"])
    
    print(f"\n{'='*50}")
    print(f"Results saved to {config.output_file}")
    print(f"Total examples: {len(examples)}")
    print(f"Processed: {len(logprobs)}")
    print(f"Skipped (no <dialogue> tag): {skipped_count}")
    if logprobs:
        mean_logprob = statistics.mean(logprobs)
        std_logprob = statistics.stdev(logprobs) if len(logprobs) > 1 else 0.0
        mean_ppl = statistics.mean(perplexities)
        std_ppl = statistics.stdev(perplexities) if len(perplexities) > 1 else 0.0
        global_sum_logprob = sum(sum_gt_logprobs)
        global_num_tokens = sum(num_gt_tokens)
        global_mean_logprob = global_sum_logprob / global_num_tokens
        global_perplexity = math.exp(-global_mean_logprob)
        print(f"{'='*50}")
        print(f"Mean GT logprob: {mean_logprob:.4f}, Std: {std_logprob:.4f}")
        print(f"Mean PPL: {mean_ppl:.4f}, Std: {std_ppl:.4f}")
        # print(f"Global Mean GT logprob: {global_mean_logprob:.4f}, Global PPL: {global_perplexity:.4f}")
        # print(f"Global Sum GT logprob: {global_sum_logprob:.4f}, Global Num Tokens: {global_num_tokens}")


if __name__ == "__main__":
    main()