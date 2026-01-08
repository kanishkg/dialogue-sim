import torch
from trl.trainer.utils import selective_log_softmax
from .grpo_trainer import LPGRPOTrainer


class ElboGRPOTrainer(LPGRPOTrainer):
    
    def __init__(self, *args, sft_weight: float = 1.0, **kwargs):
        """
        Args:
            sft_weight: Weight for the SFT loss term. Set to 0 to disable.
        """
        super().__init__(*args, **kwargs)
        self.sft_weight = sft_weight
    
    def _generate_and_score_completions(self, inputs):
        """Override to pass through true_responses to compute_loss."""
        output = super()._generate_and_score_completions(inputs)
        
        # Store true responses as padded tensor so it can be shuffled by TRL
        # (shuffle_tensor_dict expects tensors, not lists)
        true_responses = [x["true_response"] for x in inputs]
        true_response_encodings = self.processing_class(
            text=true_responses,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False
        )
        output["true_response_ids"] = true_response_encodings["input_ids"]
        output["true_response_mask"] = true_response_encodings["attention_mask"]
        
        return output
    
    def _compute_sft_loss(self, model, inputs):
        """
        Compute SFT loss: -log P(y | x, z)
        
        For each sample:
        - x = prompt  
        - z = thinking prefix (everything up to and including <dialogue>\n)
        - y = ground truth response
        
        Returns mean negative log probability over GT tokens.
        """
        device = self.accelerator.device
        
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        true_response_ids = inputs["true_response_ids"]
        true_response_mask = inputs["true_response_mask"]
        
        batch_size = prompt_ids.size(0)
        sft_losses = []
        total_gt_tokens = 0
        
        # Decode completions to text for splitting
        completions_text = self.processing_class.batch_decode(
            completion_ids,
            skip_special_tokens=True
        )
        
        # Decode true responses back to text (they were encoded as tensors for shuffling)
        true_responses = self.processing_class.batch_decode(
            true_response_ids,
            skip_special_tokens=True
        )
        
        for i in range(batch_size):
            completion_text = completions_text[i]
            
            # Extract thinking prefix (everything up to and including <dialogue>\n)
            # This matches the logic in LPGRPOTrainer._calculate_rewards
            if "<dialogue>" not in completion_text:
                prefix_text = completion_text
            else:
                prefix_text = completion_text.split('<dialogue>\n')[0] + '<dialogue>\n'
            
            # Encode the prefix
            prefix_ids = self.processing_class.encode(
                prefix_text,
                add_special_tokens=False,
                return_tensors="pt"
            )[0].to(device)
            
            # Format GT text (matching _calculate_rewards format)
            gt_text = true_responses[i] + "\n</dialogue>"
            gt_ids = self.processing_class.encode(
                gt_text,
                add_special_tokens=False,
                return_tensors="pt"
            )[0].to(device)
            
            num_gt_tokens = len(gt_ids)
            if num_gt_tokens == 0:
                continue
            
            # Get unpadded prompt (remove left padding)
            prompt_unpadded = prompt_ids[i][prompt_mask[i].bool()]
            
            # Construct full sequence: [prompt | thinking_prefix | GT]
            full_seq = torch.cat([prompt_unpadded, prefix_ids, gt_ids]).unsqueeze(0)
            attention_mask = torch.ones_like(full_seq)
            
            # Forward pass WITH gradients
            logits = model(
                input_ids=full_seq,
                attention_mask=attention_mask
            ).logits
            
            # logits[:, i] predicts token[:, i+1] (shift by 1)
            logits = logits[:, :-1, :] / self.temperature
            per_token_logps = selective_log_softmax(logits, full_seq[:, 1:])
            
            # Extract log probs for GT tokens only (at the end of sequence)
            gt_logps = per_token_logps[0, -num_gt_tokens:]
            
            # Accumulate for proper averaging
            sft_losses.append(gt_logps.sum())
            total_gt_tokens += num_gt_tokens
        
        if total_gt_tokens > 0 and len(sft_losses) > 0:
            # Mean negative log prob across all GT tokens in batch
            total_logp = torch.stack(sft_losses).sum()
            mean_sft_loss = -total_logp / total_gt_tokens
            return mean_sft_loss
        else:
            # Return zero loss that still participates in gradient computation
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    def _compute_loss(self, model, inputs):
        """Override to add SFT loss to GRPO loss."""
        # Get GRPO loss from parent
        grpo_loss = super()._compute_loss(model, inputs)
        
        if self.sft_weight == 0:
            return grpo_loss
        
        # Compute SFT loss with gradients
        sft_loss = self._compute_sft_loss(model, inputs)
        
        # Log metrics
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["sft/loss"].append(
            self.accelerator.gather(sft_loss.detach()).mean().item()
        )
        
        # Scale by gradient accumulation to match GRPO loss scaling
        scaled_sft_loss = sft_loss / self.current_gradient_accumulation_steps
        
        # Combine losses
        total_loss = grpo_loss + self.sft_weight * scaled_sft_loss
        
        return total_loss