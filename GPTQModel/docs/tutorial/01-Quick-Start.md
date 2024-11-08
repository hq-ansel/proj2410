# Quick Start

Welcome to the tutorial of GPTQModel, in this chapter, you will learn quick install `gptqmodel` from pypi and the basic usages of this library.

## Quick Installation

Start from v0.0.4, one can install `gptqmodel` directly from pypi using `pip`:
```shell
pip install gptqmodel
```

GPTQModel supports using `triton` to speedup inference, but it currently **only supports Linux**. To integrate triton, using:
```shell
pip install gptqmodel[triton]
```

For some people who want to try the newly supported `llama` type models in 🤗 Transformers but not update it to the latest version, using:
```shell
pip install gptqmodel[llama]
```

By default, CUDA extension will be built at installation if CUDA and pytorch are already installed.

To disable building CUDA extension, you can use the following commands:

For Linux
```shell
BUILD_CUDA_EXT=0 pip install gptqmodel
```
For Windows
```shell
set BUILD_CUDA_EXT=0 && pip install gptqmodel
```

## Basic Usage
*The full script of basic usage demonstrated here is `examples/quantization/basic_usage.py`*

The two main classes currently used in GPTQModel are `GPTQModel` and `QuantizeConfig`.

```python
from gptqmodel import GPTQModel, QuantizeConfig
```
### Quantize a pretrained model
To quantize a model, you need to load pretrained model and tokenizer first, for example:
```python
from transformers import AutoTokenizer

pretrained_model_name = "facebook/opt-125m"
quantize_config = QuantizeConfig(bits=4, group_size=128)
model = GPTQModel.from_pretrained(pretrained_model_name, quantize_config)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
```
This will download `opt-125m` from 🤗 Hub and cache it to local disk, then load into **CPU memory**.

*In later tutorial, you will learn advanced model loading strategies such as CPU offload and load model into multiple devices.*

Then, prepare calibration_dataset(a list of dict with only two keys, 'input_ids' and 'attention_mask') to guide quantization. Here we use only one text to simplify the code, but you should be noticed that the more calibration_dataset used, the better(most likely) the quantized model.
```python
calibration_dataset = [
    tokenizer(
        "gptqmodel is an easy-to-use model quantization library with user-friendly apis, based on GPTQ algorithm."
    )
]
```
After all recipes are prepared, we can now start to quantize the pretrained model.
```python
model.quantize(calibration_dataset)
```
Finally, we can save the quantized model:
```python
quantized_model_dir = "opt-125m-4bit-128g"
model.save_quantized(quantized_model_dir)
```
By default, the saved file type is `.bin`, you can also set `use_safetensors=True` to save a `.safetensors` model file. The format of model file base name saved using this method is: `gptq_model-{bits}bit-{group_size}g`.

Pretrained model's config and the quantize config will also be saved with file names `config.json` and `quantize_config.json`, respectively.

### Load quantized model and do inference 
Instead of `.from_pretrained`, you should use `.from_quantized` to load a quantized model.
```python
device = "cuda:0"
model = GPTQModel.from_quantized(quantized_model_dir, device=device)
```
This will first read and load `quantize_config.json` in `opt-125m-4bit-128g` directory, then based on the values of `bits` and `group_size` in it, load `gptq_model-4bit-128g.bin` model file into the first visible GPU.

Then you can initialize 🤗 Transformers' `TextGenerationPipeline` and do inference.
```python
from transformers import TextGenerationPipeline

pipeline = TextGenerationPipeline(model=model, tokenizer=tokenizer, device=device)
print(pipeline("gptqmodel is")[0]["generated_text"])
```

## Conclusion
Congrats! You learned how to quickly install `gptqmodel` and integrate with it. In the next chapter, you will learn the advanced loading strategies for pretrained or quantized model and some best practices on different situations.