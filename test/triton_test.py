
from EfficientQAT.quantize.int_linear_real import load_quantized_model
from EfficientQAT.main_block_ap import evaluate
from EfficientQAT.datautils_block import BlockTrainDataset, get_loaders
from EfficientQAT.quantize.crossblockquant import update_dataset
from template.datautils import *

quant_path = "/home/ubuntu/data/exp/proj2410/quant_model/Qwen-2.5-0.5B/EfficientQAT/w2gs128-fast/"
quant_model,tokenizer = load_quantized_model(quant_path,2,128)
text = "This is a test text."
inps = tokenizer(text,return_tensors='pt')
quant_model.to("cuda")
quant_model(**inps.to("cuda"))