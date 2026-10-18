import os

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer import TRAINING_ARGS_NAME, WEIGHTS_NAME
from transformers.utils import logging

from lora_plus import LoraPlusTrainer

logger = logging.get_logger(__name__)


class XlaLoraPlusTrainer(LoraPlusTrainer):
    """LoraPlusTrainer with a TPU-safe checkpoint save.

    transformers 4.37.2's Trainer._save_tpu does model.to("cpu") ... model.to(device).
    With torch >= 2.x an XLA<->CPU move replaces the nn.Parameter objects, so the
    optimizer keeps pointing at the stale CPU parameters: the next xm.optimizer_step
    all-reduces CPU tensors and segfaults (and the live model would stop being updated).
    xm.save already copies tensors to CPU, so the model never needs to move.
    """

    def _save_tpu(self, output_dir=None):
        import torch_xla.core.xla_model as xm

        output_dir = output_dir if output_dir is not None else self.args.output_dir
        logger.info(f"Saving model checkpoint to {output_dir}")
        model = self.model
        xm.mark_step()

        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        xm.rendezvous("saving_checkpoint")
        if isinstance(model, PreTrainedModel):
            model.save_pretrained(
                output_dir,
                is_main_process=self.args.should_save,
                save_function=xm.save,
                safe_serialization=self.args.save_safetensors,
            )
        else:
            logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
            xm.save(model.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))

        if self.tokenizer is not None and self.args.should_save:
            self.tokenizer.save_pretrained(output_dir)
