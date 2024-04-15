import os
import torch
import torch.nn as nn
import json

from transformers import Trainer
from typing import Dict, Optional, Sequence


def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
WEIGHTS_NAME = "pytorch_model.bin"
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
SAFE_WEIGHTS_NAME = "model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"

class PinkTrainer(Trainer):
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        if model is None:
            model = self.model
        weights_file = os.path.join(resume_from_checkpoint, WEIGHTS_NAME)
        weights_index_file = os.path.join(resume_from_checkpoint, WEIGHTS_INDEX_NAME)
        safe_weights_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_NAME)
        safe_weights_index_file = os.path.join(resume_from_checkpoint, SAFE_WEIGHTS_INDEX_NAME)

        if not any(
            [os.path.isfile(f) for f in [weights_file, safe_weights_file, weights_index_file, safe_weights_index_file]]
        ):
            state_dict = torch.load(os.path.join(resume_from_checkpoint, "saved_parameters.pth"), map_location="cpu")
            load_result = model.load_state_dict(state_dict, False)
        else:
            super(PinkTrainer, self)._load_from_checkpoint(resume_from_checkpoint, model=model)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        model_to_save = unwrap_model(self.model)
        # if state_dict is None:
            # state_dict = model_to_save.state_dict()
        if hasattr(model_to_save, "orig_embeds_params"):
            assert model_to_save.original_tokens_length == model_to_save.orig_embeds_params[0].shape[0]
            model_to_save.get_input_embeddings().weight.data[:model_to_save.original_tokens_length].copy_(model_to_save.orig_embeds_params[0].clone().detach())
            assert model_to_save.orig_lm_head is not None
            model_to_save.get_output_embeddings().weight.data[:model_to_save.original_tokens_length].copy_(model_to_save.orig_lm_head[0].clone().detach())
            print("back parameters")
        state_dict = model_to_save.state_dict()
        need_to_save = {}
        for name, param in model_to_save.named_parameters():
            if param.requires_grad:
                need_to_save[name] = state_dict[name]
        need_to_save['model.embed_tokens.weight'] = state_dict['model.embed_tokens.weight']
        need_to_save['lm_head.weight'] = state_dict['lm_head.weight']
        torch.save(need_to_save, os.path.join(output_dir, "saved_parameters.pth"))

        super(PinkTrainer, self)._save(output_dir, {})
        if os.path.exists(os.path.join(output_dir, "pytorch_model.bin")):
            os.remove(os.path.join(output_dir, "pytorch_model.bin"))
