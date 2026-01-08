import os
import re
import logging
import time
from transformers import AutoTokenizer
from utils.prompts import SEMANTIC_SIMILARITY_PROMPT, INFORMATION_COMPLETENESS_PROMPT
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import wandb

logger = logging.getLogger(__name__)

reward_tok = None

def reward_tokenizer(model_id):
    global reward_tok
    if reward_tok is None:
        reward_tok = AutoTokenizer.from_pretrained(model_id)
        logger.info(f"Loaded reward tokenizer: {model_id}")
    return reward_tok
    
def extract_dialogue(text, ctx):
    last_utterance = ctx[-1].strip()
    
    # Check if <dialogue> tag exists
    if "<dialogue>" not in text:
        logger.debug("No dialogue tags found in text")
        return None
    
    # Extract everything after first <dialogue> tag
    dialogue = text.split("<dialogue>", 1)[1]  # Split only at FIRST occurrence
    
    # If closing tag exists, take content before it
    if "</dialogue>" in dialogue:
        dialogue = dialogue.split("</dialogue>")[0].strip()
    
    # Take only first line (stop at newline)
    if "\n" in dialogue:
        dialogue = dialogue.split("\n")[0].strip()
    else:
        dialogue = dialogue.strip()
    
    # Clean up speaker prefixes
    dialogue = dialogue.replace('Speaker 1:', '').replace('Speaker 2:', '').strip()
    
    # Filter out placeholder text
    if dialogue.lower() == "predicted next dialogue":
        logger.debug("Skipping placeholder 'Predicted next dialogue' text")
        return None
    
    # Filter out if same as last utterance
    if dialogue.lower() == last_utterance.lower():
        logger.debug("Extracted dialogue is exactly the same as the last utterance")
        return None
    
    # Filter out empty
    if not dialogue:
        logger.debug("Extracted dialogue is empty")
        return None
    
    logger.debug(f"Extracted dialogue: {dialogue[:100]}...")
    return dialogue

def extract_score(text):
    patterns = [
        re.compile(r"<score>\s*([01](?:\.\d+)?)\s*</score>", re.I | re.S),
        re.compile(r"score[:\s]+([01](?:\.\d+)?)", re.I),
        re.compile(r"\b([01]\.\d+)\b"),
    ]
    
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            score = float(match.group(1))
            if 0.0 <= score <= 1.0:
                logger.debug(f"Extracted score: {score}")
                return score
    
    logger.warning(f"Could not extract score from text, returning None")
    return None


def get_completion_text(completion):
    if isinstance(completion, str):
        return completion
    elif isinstance(completion, list) and len(completion) > 0:
        if isinstance(completion[0], dict):
            return completion[0]["content"]
    return str(completion)
    
def chat_completion_batch(
    messages,
    model,
    temperature,
    max_tokens,
    num_completions=1,
    max_concurrency=None,
):
    client = OpenAI(base_url="http://localhost:8001/v1", api_key="dummy")
    
    def _one(idx, msgs):
        resp = client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
            n=num_completions,
        )
        return resp
                
    results = [None] * len(messages)
    
    if max_concurrency == None:
        max_concurrency = len(messages)
    print(f"sending {max_concurrency} reqs")
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        future_to_idx = {
            pool.submit(_one, i, m): i 
            for i, m in enumerate(messages)
        }
        
        pbar = tqdm(
            total=len(messages),
            disable=False,
            dynamic_ncols=True,
            desc="Processing",
            unit="req",
            miniters=1,
            file=sys.stdout,
        )
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()
            pbar.update(1)
        pbar.close()
    return results


def chat_completion_batch_claude(
    messages,
    api_key,
    model,
    temperature,
    max_tokens,
    max_concurrency=None,
):
    client = Anthropic(api_key=api_key)
    
    def _one(idx, msgs):
        user_content = msgs[0]["content"]
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user_content}]
        )
        return resp
                
    results = [None] * len(messages)
    
    if max_concurrency == None:
        max_concurrency = min(len(messages), 3)
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        future_to_idx = {
            pool.submit(_one, i, m): i 
            for i, m in enumerate(messages)
        }
        
        pbar = tqdm(
            total=len(messages),
            disable=False,
            dynamic_ncols=True,
            desc="Claude judging",
            unit="req",
            miniters=1,
            file=sys.stdout,
        )
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()
            pbar.update(1)
        pbar.close()
    return results


def chat_completion_batch_gemini(
    messages,
    api_key,
    model,
    temperature,
    max_tokens,
    max_concurrency=None,
):
    client = genai.Client(api_key=api_key) if api_key else genai.Client()
    
    def _one(idx, msgs):
        user_content = msgs[0]["content"]
        resp = client.models.generate_content(
            model=model,
            contents=user_content,
            config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
        )
        return resp
                
    results = [None] * len(messages)
    
    if max_concurrency == None:
        max_concurrency = min(len(messages), 10)
    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        future_to_idx = {
            pool.submit(_one, i, m): i 
            for i, m in enumerate(messages)
        }
        
        pbar = tqdm(
            total=len(messages),
            disable=False,
            dynamic_ncols=True,
            desc="Gemini judging",
            unit="req",
            miniters=1,
            file=sys.stdout,
        )
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            results[idx] = fut.result()
            pbar.update(1)
        pbar.close()
    return results

def combined_reward(
    prompts, 
    completions, 
    *, 
    true_response, 
    context, 
    reward_model_id, 
    use_claude_judge=False,
    claude_api_key=None,
    claude_model="claude-haiku-4-5",
    use_gemini_judge=False,
    gemini_api_key=None,
    gemini_model="gemini-2.5-pro",
    use_length_reward=False,
    return_details=False,
    sft=False,
    **kwargs
):    
    logger.info(f"Computing rewards for {len(completions)} completions")
    
    completion_texts = [get_completion_text(c) for c in completions]
    
    if sft:
        extracted_dialogues = completion_texts
    else:
        extracted_dialogues = [
            extract_dialogue(text, ctx) 
            for text, ctx in zip(completion_texts, context)
        ]
    valid_indices = [i for i, pred in enumerate(extracted_dialogues) if pred is not None]
    logger.info(f"Valid dialogues: len {len(valid_indices)}")
    
    length_rewards = []
    if use_length_reward:
        for pred_text, gt in zip(extracted_dialogues, true_response):
            if pred_text is None:
                length_rewards.append(0.0)
            else:
                pred_len, gt_len = len(pred_text), len(gt)
                reward = max(1 - abs(pred_len - gt_len) / gt_len, 0)
                length_rewards.append(reward)
    else:
        length_rewards = [1.0] * len(extracted_dialogues)
    
    if not valid_indices:
        logger.warning("No valid dialogues extracted, returning zero rewards")
        if return_details:
            return [{'reward': 0.0, 'length': 0.0, 'semantic': 0.0, 'info': 0.0, 'extracted': None} for _ in completions]
        return [0.0] * len(completions)
    
    judge_prompts = []
    for idx in valid_indices:
        dialogue_str = "\n".join([f"Speaker {(j % 2) + 1}: {utt}" for j, utt in enumerate(context[idx])])
        pred_text = extracted_dialogues[idx]
        gt = true_response[idx].strip()
        
        judge_prompts.extend([
            [{"role": "user", "content": SEMANTIC_SIMILARITY_PROMPT.format(
                reference=gt, predicted=pred_text
            )}],
            [{"role": "user", "content": INFORMATION_COMPLETENESS_PROMPT.format(
                reference=gt, predicted=pred_text
            )}]
        ])
    
    if use_claude_judge:
        all_responses = chat_completion_batch_claude(
            messages=judge_prompts,
            api_key=claude_api_key,
            model=claude_model,
            temperature=0.3,
            max_tokens=512,
        )
        def safe_extract_claude_text(resp):
            try:
                if resp is None or not resp.content or not resp.content[0].text:
                    return ""
                return resp.content[0].text
            except (AttributeError, IndexError, TypeError):
                return ""
        
        all_response_texts = [safe_extract_claude_text(resp) for resp in all_responses]
        all_scores = [extract_score(text) for text in all_response_texts]
    elif use_gemini_judge:
        all_responses = chat_completion_batch_gemini(
            messages=judge_prompts,
            api_key=gemini_api_key,
            model=gemini_model,
            temperature=0.3,
            max_tokens=2048
        )
        def safe_extract_gemini_text(resp):
            try:
                if resp is None or not resp.text:
                    return ""
                return resp.text
            except (AttributeError, TypeError):
                return ""
        
        all_response_texts = [safe_extract_gemini_text(resp) for resp in all_responses]
        all_scores = [extract_score(text) for text in all_response_texts]
    else:
        model_id = reward_model_id[0] if isinstance(reward_model_id, list) else reward_model_id
        tok = reward_tokenizer(model_id)
        
        all_responses = chat_completion_batch(
            messages=judge_prompts, model=model_id, temperature=0.3, max_tokens=512
        )
        
        def safe_extract_openai_text(resp):
            try:
                if resp is None or not resp.choices or not resp.choices[0].message.content:
                    return ""
                return resp.choices[0].message.content
            except (AttributeError, IndexError, TypeError):
                return ""
        
        all_response_texts = [safe_extract_openai_text(resp) for resp in all_responses]
        all_scores = [extract_score(text) for text in all_response_texts]
    
    sem_scores = [all_scores[2*i] for i in range(len(valid_indices))]
    info_scores = [all_scores[2*i + 1] for i in range(len(valid_indices))]
    length_scores = [length_rewards[i] for i in valid_indices]
    # For combined scores, treat None as 0 (extraction failures get 0 reward)
    combined_scores = [
        length * ((sem if sem is not None else 0) + (info if info is not None else 0)) 
        for length, sem, info in zip(length_scores, sem_scores, info_scores)
    ]
    
    # Filter out None values for mean calculations
    valid_sem_scores = [s for s in sem_scores if s is not None]
    valid_info_scores = [s for s in info_scores if s is not None]
    
    if wandb.run is not None:
        wandb.log({
            'reward/mean_semantic': sum(valid_sem_scores) / len(valid_sem_scores) if valid_sem_scores else 0.0,
            'reward/mean_info': sum(valid_info_scores) / len(valid_info_scores) if valid_info_scores else 0.0,
            'reward/mean_length': sum(length_scores) / len(length_scores) if length_scores else 0.0,
            'reward/mean_combined': sum(combined_scores) / len(combined_scores) if combined_scores else 0.0,
            'reward/valid_extractions': len(valid_indices) / len(completions) if completions else 0.0,
            'reward/valid_score_extractions': len(valid_sem_scores) / len(sem_scores) if sem_scores else 0.0,
        })
    
    if return_details:
        details = []
        for i in range(len(completions)):
            if i in valid_indices:
                idx_in_valid = valid_indices.index(i)
                sem = sem_scores[idx_in_valid]
                info = info_scores[idx_in_valid]
                details.append({
                    'reward': combined_scores[idx_in_valid],
                    'length': length_scores[idx_in_valid],
                    'semantic': sem,  # Can be None if extraction failed
                    'info': info,  # Can be None if extraction failed
                    'extracted': extracted_dialogues[i],
                    'semantic_response': all_response_texts[2*idx_in_valid],
                    'info_response': all_response_texts[2*idx_in_valid + 1],
                    'score_extraction_failed': sem is None or info is None
                })
            else:
                details.append({
                    'reward': 0.0,
                    'length': length_rewards[i],
                    'semantic': None,
                    'info': None,
                    'extracted': extracted_dialogues[i],
                    'semantic_response': None,
                    'info_response': None,
                    'score_extraction_failed': True
                })
        return details 
    
    rewards = [0.0] * len(completions)
    for i, idx in enumerate(valid_indices):
        rewards[idx] = combined_scores[i]
    
    return rewards