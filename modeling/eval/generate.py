import os
import logging
from itertools import chain
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
from tqdm import tqdm
import pydra
import json
from openai import OpenAI
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        self.checkpoint_path: str = "meta-llama/Llama-3.2-3B-Instruct"
        self.reward_model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
        self.base: bool = False
        self.thinking: bool = True
        self.sft: bool = False
        self.num_samples: int = 100
        self.max_new_tokens: int = 512
        self.temperature: float = 0.3
        self.top_p: float = 0.95
        self.vllm_generation_url: str = "http://localhost:8000/v1"
        self.output_file: str = "generations/llama_instruct_100samples.jsonl"


def build_examples(dialog, base_model=False, reward_model_id=None, thinking=True, sft=False):
    examples = []
    
    for idx in range(1, len(dialog)):
        dialog_cleaned = [
            re.sub(r'\s+([.,!?;:])', r'\1', utt.strip()) 
            for utt in dialog
        ]

        context_lines = [
            f"Speaker {(i % 2) + 1}: {utt}" 
            for i, utt in enumerate(dialog_cleaned[:idx])
        ]
        context = "\n".join(context_lines)
        next_speaker =  len(context_lines) % 2 + 1
        next_speaker_text = f"Speaker {next_speaker}: "
        
        if sft:
            sys_prompt = (
                "You are a dialogue prediction system. "
                "Your goal is to predict only the next dialogue based on conversation."
            )
            prompt = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Dialogue:\n{context}\nPredict just the next dialogue."}
            ]
        elif not thinking:
            # non thinking mode
            sys_prompt = """You are a dialogue prediction system. Your goal is to predict the immediate next dialogue for a conversation in <dialogue> tags.
REMEMBER: The speakers alternate in their dialogues.
Here is how to format your prediction:
<dialogue> Predicted next dialogue </dialogue>"""
            prompt = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Now predict the next dialogue for this conversation:\n{context}\nWhat will the next speaker, Speaker {next_speaker}, say?"}            
            ]
        else:
            # thinking mode
            if base_model:
                prompt = f"""Your goal is to predict the immediate next dialogue for a conversation in <dialogue> tags. Think step by step before you predict the next dialogue.

Here is how to format your prediction:
<think> Let's think step by step what the next dialogue might be:
[Your reasoning about the context and speakers here]
</think>
<dialogue> Predicted next dialogue </dialogue>

Now predict the next dialogue for this conversation. Think in <think></think> tags before you predict the dialogue:
{context}

<think> Let's think step by step what the next dialogue might be:"""
            else:
                sys_prompt = """You are a dialogue prediction system. Your goal is to predict the immediate next dialogue for a conversation in <dialogue> tags. 
Think step by step before you predict the next dialogue.
REMEMBER: The speakers alternate in their dialogues.
Here is how to format your prediction:
<think> Let's think step by step what the next dialogue might be:
[Your reasoning about the context and speakers here]
</think>
<dialogue>
Predicted next dialogue 
</dialogue>"""
                prompt = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"Now predict the next dialogue for this conversation. Think in <think></think> tags before you predict the dialogue:\n{context}\nWhat will the next speaker, Speaker {next_speaker}, say?"}
                ]
        
        if not sft:
            example = {
                "prompt": prompt,
                "true_response": next_speaker_text + dialog_cleaned[idx],
                "context": dialog_cleaned[:idx],
                "reward_model_id": reward_model_id
            }
            examples.append(example)
        else:
            example = {
                "prompt": prompt,
                "true_response": dialog_cleaned[idx],
                "context": dialog_cleaned[:idx],
                "reward_model_id": reward_model_id
            }
            examples.append(example)
    
    return examples


def generate_completions_vllm(prompts, config):
    client = OpenAI(base_url=config.vllm_generation_url, api_key="dummy")
    tokenizer = AutoTokenizer.from_pretrained(config.checkpoint_path)
    
    if config.base:
        formatted_prompts = prompts
    else:
        formatted_prompts = [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]
    
    def generate_one(idx, prompt):
        resp = client.completions.create(
            model=config.checkpoint_path,
            prompt=prompt,
            temperature=config.temperature,
            max_tokens=config.max_new_tokens,
            top_p=config.top_p,
        )
        return resp.choices[0].text
    
    results = [None] * len(formatted_prompts)
    
    with ThreadPoolExecutor(max_workers=32) as pool:
        future_to_idx = {
            pool.submit(generate_one, i, p): i 
            for i, p in enumerate(formatted_prompts)
        }
        
        pbar = tqdm(
            total=len(formatted_prompts),
            desc="Generating completions",
            unit="req",
        )
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()
            pbar.update(1)
        pbar.close()
    
    return results


@pydra.main(Config)
def main(config: Config):
    logger.info("Loading DailyDialog validation set...")
    dd = load_from_disk("/scr/kanishkg/dialogue_models/daily_dialog_local")
    
    val_examples = list(chain.from_iterable(
        build_examples(d["dialog"], base_model=config.base, reward_model_id=config.reward_model_id, thinking=config.thinking, sft=config.sft)
        for d in dd["validation"]
    ))
    
    random.seed(42)
    random.shuffle(val_examples)
    
    if config.num_samples < len(val_examples):
        val_examples = val_examples[:config.num_samples]
    
    logger.info(f"Evaluating on {len(val_examples)} examples")
    prompts = [ex["prompt"] for ex in val_examples]
    
    logger.info(f"Using vLLM generation from: {config.vllm_generation_url}")
    logger.info(f"Generating completions...")
    all_completions = generate_completions_vllm(prompts, config)
    
    os.makedirs(os.path.dirname(config.output_file), exist_ok=True)
    with open(config.output_file, 'w') as f:
        for sample_id, (ex, comp) in enumerate(zip(val_examples, all_completions)):
            record = {
                "sample_id": sample_id,
                "example_id": sample_id,
                "prompt": ex["prompt"],
                "completion": comp,
                "true_response": ex["true_response"],
                "context": ex["context"],
                "reward_model_id": ex["reward_model_id"],
            }
            f.write(json.dumps(record) + "\n")
    logger.info(f"Saved {len(val_examples)} completions to {config.output_file}")


if __name__ == "__main__":
    main()
