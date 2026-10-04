"""Early stop on sustained decoder loops; abstention, never a learner error."""
import torch
from transformers import StoppingCriteria
from src.training_with_gpu.decoding_safety import repeated_cycle


class LoopStop(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.prefix_length = len(tokenizer.prefix_tokens)

    def __call__(self, input_ids, scores, **kwargs):
        stopped = []
        for sequence in input_ids.tolist():
            content = sequence[self.prefix_length:]
            cyclic = repeated_cycle(content[-32:])
            if not cyclic and len(content) % 16 == 0 and len(content) >= 16:
                cyclic = repeated_cycle(self.tokenizer.decode(content, skip_special_tokens=True).split())
            stopped.append(cyclic)
        return torch.tensor(stopped, device=input_ids.device, dtype=torch.bool)
