import os
from itertools import chain
import logging
import torch
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from lp_trainer.grpo_trainer import LPGRPOTrainer
from lp_trainer.sft_grpo_trainer import ElboGRPOTrainer
import pydra
from rewards import (
   combined_reward
)
import re

class Config(pydra.Config):
    def __init__(self):
        super().__init__()
        # self.model_id: str = "meta-llama/Llama-3.2-3B-Instruct"
        self.model_id: str = "Qwen/Qwen2.5-3B-Instruct"
        self.reward_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
        self.base: bool = False
        self.thinking: bool = True
        self.lp_trainer: bool = False
        self.elbo_trainer: bool = True
        self.sft_weight: float = 1.0  # Weight for SFT loss in SFTLPGRPOTrainer
        self.debug: bool = False
        
        self.output_dir: str = ""
        self.run_name: str = None
        
        self.learning_rate: float = 1e-6
        self.per_device_train_batch_size: int = 8
        self.gradient_accumulation_steps: int = 8
        self.num_train_epochs: int = 1
        
        self.num_generations: int = 16
        self.max_completion_length: int = 1024
        self.max_prompt_length: int = 512
        self.temperature: float = 1.0
        self.top_p: float = 1.0
        self.min_p: float = 0.0
        
        self.warmup_steps: int = 20
        self.epsilon_high: float = 0.28
        self.beta: float = 0.
        self.loss_type: str = "dr_grpo" 
        self.mask_truncated_completions: bool = True
        self.sync_ref_model: bool = False 
        self.ref_model_sync_steps: int = 50
        self.ref_model_mixup_alpha: float = 1.0
        self.num_gpus: int = 3
        self.logging_steps: int = 10
        self.save_steps: int = 50
        self.save_total_limit: int = 15
        
        self.use_vllm: bool = True
        self.vllm_server_host: str = "localhost"
        self.vllm_server_port: int = 8000
        
        self.max_grad_norm: float = 1.0
    
    def finalize(self):
        if self.run_name is None:
            from datetime import datetime
            model_name = self.model_id.split("/")[-1].replace("-", "_").lower()
            mode = 'base' if self.base else 'instruct'
            thinking = 'thinking' if self.thinking else 'no_thinking'
            lp_trainer = 'lp_trainer' if self.lp_trainer else ''
            temp = f"t{self.temperature}".replace(".", "")
            comp_len = f"c{self.max_completion_length}"
            batch = f"bs{self.per_device_train_batch_size}"
            timestamp = datetime.now().strftime("%m%d_%H%M")
            
            self.run_name = f"{model_name}_{mode}_{thinking}_{lp_trainer}_{temp}_{comp_len}_{batch}_{timestamp}"
        
        effective_batch_size = (
            self.per_device_train_batch_size * 
            self.num_gpus *   
            self.gradient_accumulation_steps
        )
        assert effective_batch_size % self.num_generations == 0, "Effective batch size must be divisible by number of generations so that it can be grouped"
        total_gens = effective_batch_size * self.num_generations
        print(f"\n📊 Training Configuration:")
        print(f"   Run name: {self.run_name}")
        print(f"   Effective batch size: {effective_batch_size}")
        print(f"   Generations per batch: {total_gens}")

def build_examples(dialog, base_model=False, reward_model_id=None, thinking=True):
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

        if not thinking:
            # non thinking mode
            sys_prompt = """You are a dialogue prediction system. Your goal is to predict the immediate next dialogue for a conversation in <dialogue> tags.
Here is how to format your prediction:
<dialogue> Predicted next dialogue </dialogue>"""
            prompt = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Now predict the next dialogue for this conversation:\n {context}"}
            ]
        else:
            if base_model:
                prompt = f"""Your goal is to predict the immediate next dialogue for a conversation in <dialogue> tags. Think step by step before you predict the next dialogue.

Here is how to format your prediction:
<think> Let's think step by step what the next dialogue might be:
[Your reasoning about the context and speakers here]
</think>
<dialogue>
Predicted next dialogue
</dialogue>

Now predict the next dialogue for this conversation. Think in <think></think> tags before you predict the dialogue:
{context}
What will the next speaker, Speaker {next_speaker}, say?
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
        
        example = {
            "prompt": prompt,
            "true_response": next_speaker_text + dialog_cleaned[idx],
            "context": dialog_cleaned[:idx],
            "reward_model_id": reward_model_id
        }
        examples.append(example)
    
    return examples


@pydra.main(Config)
def main(config: Config):
    logging.getLogger("httpx").setLevel(logging.WARNING)

    os.environ["WANDB_PROJECT"] = "modelling_dialogue"
    os.environ["WANDB_ENTITY"] = "cocolab"
    os.environ["WANDB_DIR"] = "/scr/kanishkg/wandb_data"
    os.environ["WANDB_CACHE_DIR"] = "/scr/kanishkg/wandb_data/cache"
    os.environ["WANDB_CONFIG_DIR"] = "/scr/kanishkg/wandb_data/config"
    os.environ["WANDB_DISABLE_SYMLINKS"] = "true"
    os.environ["WANDB_ARTIFACTS_DIR"] = "/scr/kanishkg/wandb_data/artifacts"
    os.environ["WANDB_LOG_MODEL"] = "false"
    
    os.environ["TMPDIR"] = "/scr/kanishkg/tmp"
    os.makedirs("/scr/kanishkg/tmp", exist_ok=True)
    os.makedirs("/scr/kanishkg/wandb_data", exist_ok=True)
    
    print("Loading daily_dialog dataset...")
    dd = load_from_disk("/scr/kanishkg/dialogue_models/daily_dialog_local")
    train_examples = list(chain.from_iterable(
        build_examples(d["dialog"], base_model=config.base, reward_model_id=config.reward_model_id, thinking=config.thinking) 
        for d in dd["train"]
    ))
    train_ds = Dataset.from_list(train_examples)
    print(f"Training examples: {len(train_ds)}")
    # shuffle the train_ds
    train_ds = train_ds.shuffle(seed=42)
    
    print(f"Loading model: {config.model_id} ({'base' if config.base else 'instruct'})")
    tok = AutoTokenizer.from_pretrained(config.model_id, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    grpo_cfg = GRPOConfig(
        output_dir=config.output_dir,
        model_init_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "flash_attention_2",
        },
        run_name=config.run_name,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        num_generations=config.num_generations,
        max_completion_length=config.max_completion_length,
        max_prompt_length=config.max_prompt_length,
        bf16=True,
        logging_steps=config.logging_steps,
        fp16=False,
        remove_unused_columns=False,
        torch_compile=True,  # Disabled - causes gradient issues with gradient_checkpointing
        
        warmup_steps=config.warmup_steps,
        lr_scheduler_type="linear",
        max_grad_norm=config.max_grad_norm, 
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # Required for proper gradient flow
        dataloader_num_workers=8,
        
        temperature=config.temperature,
        top_p=config.top_p,
        min_p=config.min_p,
        
        epsilon=0.2,
        epsilon_high=config.epsilon_high,
        scale_rewards=False,
        loss_type=config.loss_type,
        mask_truncated_completions=config.mask_truncated_completions,
        beta=config.beta,
        
        sync_ref_model=config.sync_ref_model,
        ref_model_sync_steps=config.ref_model_sync_steps,
        ref_model_mixup_alpha=config.ref_model_mixup_alpha,
        
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        
        eval_strategy="no",
        
        use_vllm=config.use_vllm,
        vllm_mode="server",
        vllm_server_host="localhost",
        vllm_server_port=8000,
        vllm_gpu_memory_utilization=0.95,
        
        log_completions=True,
        num_completions_to_print=10,
        
        report_to=["wandb"],
        logging_dir=f"{config.output_dir}/logs",
    )
    
    reward_funcs = [
        combined_reward
    ]
    
    print("Starting training...")
    if config.elbo_trainer:
        def elbo_trainer_gt_logprob_reward(prompts, completions, completion_ids, **kwargs):
            return [None] * len(prompts)
            
        trainer = ElboGRPOTrainer(
            model=config.model_id,
            args=grpo_cfg,
            processing_class=tok,
            train_dataset=train_ds,
            reward_funcs=[elbo_trainer_gt_logprob_reward],
            debug=config.debug,
            sft_weight=config.sft_weight,
        )
    else:
        trainer = GRPOTrainer(
            model=config.model_id,
            args=grpo_cfg,
            processing_class=tok,
            train_dataset=train_ds,
            reward_funcs=reward_funcs,
        )
    
    trainer.train()
    
    print(f"Saving model to {config.output_dir}")
    trainer.model.save_pretrained(config.output_dir)
    tok.save_pretrained(config.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()