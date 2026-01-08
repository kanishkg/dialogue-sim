from itertools import chain
import torch
from datasets import load_dataset, Dataset
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
import os
import re

def build_examples_for_sft(dialog):
    sys_prompt = (
        "You are a dialogue prediction system. "
        "Your goal is to predict only the next dialogue based on conversation."
    )
    
    dialog_cleaned = [
        re.sub(r'\s+([.,!?;:])', r'\1', utt.strip()) 
        for utt in dialog
    ]
    
    examples = []
    for idx in range(1, len(dialog_cleaned)):
        context_lines = [
            f"Speaker {(i % 2) + 1}: {utt}" 
            for i, utt in enumerate(dialog_cleaned[:idx])
        ]
        context = "\n".join(context_lines)
        example = {
            "prompt": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"Dialogue:\n{context}\nPredict just the next dialogue."}
            ],
            "completion": [
                {"role": "assistant", "content": dialog_cleaned[idx]}
            ]
        }
        examples.append(example)
    
    return examples

def build_examples_for_sft_base(dialog):
    sys_prompt = (
        "You are a dialogue prediction system. "
        "Your goal is to predict only the next dialogue based on conversation."
    )
    
    dialog_cleaned = [
        re.sub(r'\s+([.,!?;:])', r'\1', utt.strip()) 
        for utt in dialog
    ]
    
    examples = []
    for idx in range(1, len(dialog_cleaned)):
        context_lines = [
            f"Speaker {(i % 2) + 1}: {utt}" 
            for i, utt in enumerate(dialog_cleaned[:idx])
        ]
        context = "\n".join(context_lines)
        
        prompt_text = f"{sys_prompt}\n\nDialogue:\n{context}\nPredict just the next dialogue.\n\n"
        completion_text = dialog_cleaned[idx]
        
        example = {
            "text": prompt_text + completion_text
        }
        examples.append(example)
    
    return examples

def main():
    print("Loading DailyDialog dataset...")
    dd = load_from_disk("sft/daily_dialog_local")
    print("Building training examples...")
    train_examples = list(chain.from_iterable(
        build_examples_for_sft(d["dialog"]) for d in dd["train"]
    ))
    train_dataset = Dataset.from_list(train_examples)
    print(f"Created {len(train_dataset)} training examples")
    print("Building validation examples...")
    val_examples = list(chain.from_iterable(
        build_examples_for_sft(d["dialog"]) for d in dd["validation"]
    ))
    val_dataset = Dataset.from_list(val_examples)
    print(f"Created {len(val_dataset)} validation examples")
    
    print("\nExample training sample:")
    print(f"Prompt: {train_dataset[0]['prompt']}")
    print(f"Completion: {train_dataset[0]['completion']}")
    
    config = SFTConfig(
        output_dir="./dialogue_predictor_sft_cleaned",
        model_init_kwargs={"torch_dtype": "bfloat16", "device_map": None},
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # effective bs = 16
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        torch_compile=True,
        #packing=True,
        warmup_ratio=0.05,
        completion_only_loss=True,  # train on assistant response
        eval_strategy="steps",
        eval_steps=100,
        per_device_eval_batch_size=8,
        logging_steps=1,
        save_steps=200,
        save_total_limit=20,
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name="dialogue-sft-2epoch_2e-5_cleaned",
    )
    
    print("\nInitializing SFTTrainer...")
    trainer = SFTTrainer(
        model="meta-llama/Llama-3.2-3B-Instruct",
        args=config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    print("\nSaving final model...")
    trainer.save_model("./dialogue_predictor_sft_cleaned")
    print("Training complete!")

if __name__ == "__main__":
    main()
