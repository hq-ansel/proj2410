from typing import List, Tuple

from easydict import EasyDict as edict
from gptqmodel import GPTQModel, QuantizeConfig
import torch
from transformers import AutoTokenizer, PreTrainedModel



def load_model_and_tokenizer(model_path: str):
    model = GPTQModel.load(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model.model, tokenizer

def gptq_pipeline(
        model: PreTrainedModel,
        train_dataset: List[Tuple[torch.Tensor, torch.Tensor]],
        args: edict,
)->GPTQModel:
    del model
    w_bit = args.wbits
    quant_config = QuantizeConfig(bits=w_bit, group_size=args.group_size)
    gptq_model = GPTQModel.load(
        args.model,
        quant_config
    )
    train_dataset = [
        {
            "input_ids": x[0].squeeze(0),
            "attention_mask": torch.ones_like(x[0].squeeze(0)),
            }
                      for x in train_dataset]
    gptq_model.quantize(train_dataset)
    return gptq_model
    pass
