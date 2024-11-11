import os
import random
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import shutil
from concurrent.futures import ThreadPoolExecutor

import asyncio
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer,PreTrainedTokenizer,PreTrainedModel
from datasets import load_dataset
import numpy as np

from .quantize.utils import StopException,Catcher


def generate_llama_mask_and_position_embedding(seq_len,
                          rotary_emb: nn.Module,
                            batch_size=1,
                            hidden_size:int=4096,
                            dtype=torch.float32,
                            device="cpu") -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    # 获取数据类型的最小值
    min_dtype = torch.finfo(dtype).min
    
    # 生成因果掩码，使用 min_dtype 填充上三角部分
    attention_mask = torch.full((seq_len, seq_len), fill_value=min_dtype, dtype=dtype, device=device)
    attention_mask = torch.triu(attention_mask, diagonal=1)  # 上三角填充为 min_dtype
    attention_mask = attention_mask.masked_fill(attention_mask == 0, 0)  # 下三角保留0

    # 将 attention_mask 扩展到批次维度
    attention_mask = attention_mask[None,None,:,:].expand(batch_size,1, -1, -1)
    
    # 生成位置 id，从 0 到 seq_len-1，并扩展到批次维度
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    hidden_states = torch.zeros((batch_size, seq_len, hidden_size), dtype=dtype, device=device)
    position_embeddings = rotary_emb(hidden_states, position_ids.to(device))
    return attention_mask, position_embeddings

class LazyLoadDataset(Dataset):
    def __init__(self,
            data_dir:str,
            tmp_dir:str,
            split:str='train',
            file_list:List[str]=None,
            meta_device_list:List[str]=None,
            data_list:List[Any]=None,
             ):
        """
        初始化Dataset

        参数:
            data_dir (str): 数据文件的路径
            文件类似 input/output_layer{layer_idx}_{sample_idx}.pt
            tmp_dir (str): 临时文件存放路径 因为后续需要用模型层更新参数
            transform (callable, optional): 应用于数据的转换函数
        """
        if file_list is not None:
            self.file_list = file_list
            self.meta_device_list = meta_device_list
            self.data_list = data_list
            return 
        self.data_dir = os.path.join(data_dir, split)
        assert os.path.exists(self.data_dir), f"data_dir {self.data_dir} not exists"
        self.tmp_dir = os.path.join(tmp_dir, split)
        os.makedirs(self.tmp_dir, exist_ok=True)

        self.layer_idx = 0 # 初始从0开始
        self.total_file_list = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.pt')])
        self.file_dict = {}
        for f in self.total_file_list:
            filename = f.split("/")[-1]
            file_type,layer_idx,sample_idx = filename.split("_")
            layer_idx,sample_idx = int(layer_idx[5:]),int(sample_idx[:-3])
            self.file_dict[file_type] = self.file_dict.get(file_type,{})
            self.file_dict[file_type][layer_idx] = self.file_dict[file_type].get(layer_idx,{})
            self.file_dict[file_type][layer_idx][sample_idx] = os.path.join(self.data_dir,f)
        # print(self.file_dict.keys())
        self.file_list = [ self.file_dict["input"][0][idx]
                           for idx in range(len(self.file_dict["input"][0]))]
        self.meta_device_list = [
            "disk" for idx in range(len(self.file_dict["input"][0]))
        ]
        self.data_list = [None for idx in range(len(self.file_dict["input"][0]))]
        self.max_layer_idx = len(self.file_dict["input"])
        self.executor = ThreadPoolExecutor(max_workers=32)

    def __len__(self):
        """返回数据集中样本的数量"""
        return len(self.file_list)

    def __getitem__(self, idx) -> Union[ Tuple[torch.Tensor, torch.Tensor]]:
        """
        根据索引idx加载并返回样本数据

        参数:
            idx (int): 数据索引
        返回:
            Tuple[torch.Tensor, torch.Tensor]: 输入与输出的张量
        """
        # 读取 .pt 文件
        input_file_path = self.file_list[idx]
        output_file_path = input_file_path.replace("input","output")
        if self.meta_device_list[idx] == "disk":
            input_sample = torch.load(input_file_path,weights_only=True)
            output_sample = torch.load(output_file_path,weights_only=True)
            self.data_list[idx] = (input_sample, output_sample)
            self.meta_device_list[idx] = "cpu"
        else:
            input_sample,output_sample = self.data_list[idx]
        return input_sample, output_sample
    # @torch.no_grad()
    # def update_dataset(self, module: Callable,
    #                     layer_idx: int ,
    #                     batch_size: int=1 ,
    #                     num_workers: int=2,
    #                     attention_mask:torch.Tensor=None,
    #                     position_embeddings:Tuple[torch.Tensor, torch.Tensor]=None):
    #     """
    #     更新第i层的参数
    #     还是不要设置batch size 因为这里是单个样本的更新?
    #     """
    #     assert attention_mask is not None and position_embeddings is not None, f"attention_mask is {attention_mask} and position_embeddings is {position_embeddings}, they should not be None"
    #     new_file_list = []
    #     tmpDataset = LazyLoadDataset(None,
    #                 None,
    #                 file_list=self.file_list,
    #                 meta_device_list=self.meta_device_list,
    #                 data_list=self.data_list)
    #     tmpDataLoader = DataLoader(tmpDataset, batch_size=batch_size,num_workers=num_workers, shuffle=False)
    #     device = next(module.parameters()).device
    #     futures = []
    #     _dtype = next(module.parameters()).dtype

    #     for it in position_embeddings:
    #         it.to(device)
    #     for idx, batch in tqdm(enumerate(tmpDataLoader),total=len(tmpDataLoader),desc="update_dataset"):
    #         input_sample, output_sample = batch
    #         output = module(input_sample.to(device,dtype=_dtype),
    #                         attention_mask=attention_mask.to(device),
    #                           position_embeddings=position_embeddings)[0]
    #         # print(f"output-output_sample{output-output_sample.to(device,dtype=_dtype)}")
    #         # 保存更新后的参数
    #         for innder_idx in range(batch_size):
    #             real_idx = idx * batch_size + innder_idx
    #             new_input_file_name = "input_layer{}_{}.pt".format(layer_idx, real_idx)
    #             new_input_file_path = os.path.join(self.tmp_dir, new_input_file_name)
    #             new_file_list.append(new_input_file_path)
    #             new_output_file_name = "output_layer{}_{}.pt".format(layer_idx, real_idx)
    #             new_output_file_path = os.path.join(self.tmp_dir, new_output_file_name)
    #             # print(f"save {new_input_file_path} and {new_output_file_path}")
    #             assert output[innder_idx].dim() == 2, f"output should be 2-dimensional,get {output[innder_idx].dim()}"
    #             if self.meta_device_list[real_idx] == "disk":
    #                 futures.append(self.executor.submit(torch.save, output[innder_idx].detach().cpu(), new_input_file_path))
    #             else:
    #                 self.data_list[real_idx] = output[innder_idx].detach().cpu()
    #             futures.append(self.executor.submit(
    #                 shutil.copy, self.file_dict["output"][layer_idx][real_idx], new_output_file_path))

    #     # 等待所有 Future 完成
    #     for future in futures:
    #         future.result() 
    #     self.file_list = new_file_list
    #     self.layer_idx = layer_idx
    @torch.no_grad()
    def update_dataset(self, module: Callable,
                    layer_idx: int,
                    batch_size: int = 1,
                    num_workers: int = 2,
                    attention_mask: torch.Tensor = None,
                    position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None):
        """
        更新第 i 层的参数
        """
        assert attention_mask is not None and position_embeddings is not None, (
            f"attention_mask is {attention_mask} and position_embeddings is {position_embeddings}, "
            "they should not be None"
        )

        new_file_list = []
        tmpDataset = LazyLoadDataset(None,
                                    None,
                                    file_list=self.file_list,
                                    meta_device_list=self.meta_device_list,
                                    data_list=self.data_list)
        tmpDataLoader = DataLoader(tmpDataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        device = next(module.parameters()).device
        _dtype = next(module.parameters()).dtype

        # 确保 position_embeddings 移动到正确的设备
        position_embeddings = tuple(it.to(device) for it in position_embeddings)

        async def async_update():
            loop = asyncio.get_running_loop()
            tasks = []

            async def save_to_disk(tensor, path):
                await loop.run_in_executor(self.executor, torch.save, tensor, path)

            async def copy_to_disk(src, dst):
                await loop.run_in_executor(self.executor, shutil.copy, src, dst)

            for idx, batch in tqdm(enumerate(tmpDataLoader), total=len(tmpDataLoader), desc="update_dataset"):
                input_sample, output_sample = batch
                output = module(input_sample.to(device, dtype=_dtype),
                                attention_mask=attention_mask.to(device),
                                position_embeddings=position_embeddings)[0]

                for inner_idx in range(batch_size):
                    real_idx = idx * batch_size + inner_idx
                    new_input_file_name = f"input_layer{layer_idx}_{real_idx}.pt"
                    new_input_file_path = os.path.join(self.data_dir, new_input_file_name)
                    new_file_list.append(new_input_file_path)
                    # new_output_file_name = f"output_layer{layer_idx}_{real_idx}.pt"
                    # new_output_file_path = os.path.join(self.tmp_dir, new_output_file_name)

                    assert output[inner_idx].dim() == 2, f"Output should be 2-dimensional, got {output[inner_idx].dim()}"

                    if self.meta_device_list[real_idx] == "disk":
                        tasks.append(save_to_disk(output[inner_idx].detach().cpu(), new_input_file_path))
                    else:
                        output_sample = torch.load(self.file_dict["output"][layer_idx][real_idx], weights_only=True)
                        self.data_list[real_idx] = (output[inner_idx].detach().cpu(), output_sample)

                    # tasks.append(copy_to_disk(self.file_dict["output"][layer_idx][real_idx], new_output_file_path))

            # 等待所有异步任务完成
            await asyncio.gather(*tasks)

        # 使用 asyncio.run 调用异步任务
        asyncio.run(async_update())

        # 更新属性
        self.file_list = new_file_list
        self.layer_idx = layer_idx



async def async_torch_save(tensor, path, executor):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, torch.save, tensor, path)

async def async_torch_load(path, executor):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, lambda: torch.load(path, weights_only=True))

async def async_generate_block_train_data(
        model: PreTrainedModel,
        dataloader: List[Tuple[torch.Tensor, torch.Tensor]],
        out_dir: str,
        executor: ThreadPoolExecutor,
):
    device = "cuda:0"
    layers = dict(model.named_modules()).get("model.layers", None)
    result = []

    class Interrupt(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.attention_mask = None
            self.position_embeddings = None
            self.idx = 0
            self.data_list = []

        def forward(self, inp, **kwargs):
            self.attention_mask = kwargs.get("attention_mask", None)
            self.position_embeddings = kwargs.get("position_embeddings", None)
            result.append(asyncio.create_task(
                async_torch_save(
                    inp.squeeze(0).detach().cpu(),
                    os.path.join(out_dir, f"input_layer0_{self.idx}.pt"),
                    executor
                )
            ))
            self.idx += 1
            raise StopException()

    layers[0] = Interrupt(layers[0])
    for n, m in model.named_modules():
        if "layer" in n:
            break
        m.to(device)
    layers[0].to(device)

    for idx, batch in enumerate(dataloader):
        inp, tar = batch
        try:
            model(inp.to(device))
        except StopException:
            pass

    attention_mask, position_embeddings = layers[0].attention_mask, layers[0].position_embeddings
    total = layers[0].idx
    layers[0] = layers[0].module
    model.to("cpu")
    print(f"save attention_mask and position_embeddings")

    split_path = out_dir.split("/")
    raw_dir = "/".join(split_path[:-1])

    await async_torch_save(
        {"attention_mask": attention_mask, "position_embeddings": position_embeddings},
        os.path.join(raw_dir, "mask_and_position_embedding.pt"),
        executor
    )

    for layer_idx in tqdm(range(len(layers)), total=len(layers), desc="update_dataset"):
        await asyncio.gather(*result)
        result.clear()
        layer = layers[layer_idx]
        layer.to(device)

        for file_idx in range(total):
            if layer_idx == 0:
                input_file_path = os.path.join(out_dir, f"input_layer{layer_idx}_{file_idx}.pt")
            else:
                input_file_path = os.path.join(out_dir, f"output_layer{layer_idx - 1}_{file_idx}.pt")

            inp = await async_torch_load(input_file_path, executor)
            
            output = layer(
                inp.unsqueeze(0).to(device),
                attention_mask=attention_mask.to(device),
                position_embeddings=position_embeddings
            )[0]

            result.append(asyncio.create_task(
                async_torch_save(
                    output.squeeze(0).detach().cpu(),
                    os.path.join(out_dir, f"output_layer{layer_idx}_{file_idx}.pt"),
                    executor
                )
            ))

        layer.to("cpu")

    await asyncio.gather(*result)  # 确保所有的保存操作完成

    return attention_mask, position_embeddings

@torch.no_grad()
def generate_block_train_data(
        model: PreTrainedModel,
        dataloader: List[Tuple[torch.Tensor, torch.Tensor]],
        out_dir: str,
):
    # 创建一个线程池用于文件操作
    executor = ThreadPoolExecutor(max_workers=32)
    # 运行异步函数
    return asyncio.run(async_generate_block_train_data(model, dataloader, out_dir, executor))

def get_wikitext2(
    tokenizer: PreTrainedTokenizer, 
    train_size: int, 
    val_size: int, 
    seed: int, 
    seqlen: int, 
    test_only: bool
) -> Union[Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]], Dict[str, torch.Tensor]]:
    """
    Load and preprocess the Wikitext-2 dataset, returning train and validation data.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer for encoding the text data.
        train_size (int): Number of training samples to generate.
        val_size (int): Number of validation samples to generate.
        seed (int): Seed for random number generator to ensure reproducibility.
        seqlen (int): Length of the input sequences.
        test_only (bool): If True, only return the tokenized test data.

    Returns:
        Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
            A tuple of two lists, each containing tuples of input and target tensors for 
            the training and validation datasets.
    """
    
    print("Loading Wikitext-2 dataset...")
    
    # Load datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Tokenize test data
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    if test_only:
        return testenc

    # Tokenize training data
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')

    # Set random seed for reproducibility
    random.seed(seed)
    
    # Prepare training and validation data loaders
    trainloader: List[Tuple[torch.Tensor, torch.Tensor]] = []
    val_sample_ratio = 0.9  # 90% for training, 10% for validation to avoid overlap

    # Generate training samples
    for _ in range(train_size):
        i = random.randint(0, int(trainenc.input_ids.shape[1] * val_sample_ratio) - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100  # Shift target for language modeling
        trainloader.append((inp, tar))

    # Generate validation samples
    valloader: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(val_size):
        i = random.randint(
            int(trainenc.input_ids.shape[1] * val_sample_ratio) - seqlen - 1,
            trainenc.input_ids.shape[1] - seqlen - 1
        )
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100  # Shift target for language modeling
        valloader.append((inp, tar))

    return trainloader, valloader


def get_c4(
    tokenizer: PreTrainedTokenizer, 
    train_size: int, 
    val_size: int, 
    seed: int, 
    seqlen: int, 
    test_only: bool
) -> Union[torch.Tensor, Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]],Dict[str, torch.Tensor]]:
    """
    Load and preprocess the C4 dataset, returning train and validation data or just the validation set if test_only is True.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer for encoding the text data.
        train_size (int): Number of training samples to generate.
        val_size (int): Number of validation samples to generate.
        seed (int): Seed for random number generator to ensure reproducibility.
        seqlen (int): Length of the input sequences.
        test_only (bool): If True, only return the tokenized validation data.

    Returns:
        Union[torch.Tensor, Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]]:
            Either the validation tensor if test_only is True, or a tuple containing two lists for train and validation data.
    """
    print("get_c4")
    
    # Attempt to load dataset from local path for faster loading
    try:
        traindata = load_dataset(
            "arrow",
            data_files={
                "train": "/path/to/local/train.arrow",
                "validation": "/path/to/local/validation.arrow",
            },
            split='train'
        )
        valdata = load_dataset(
            "arrow",
            data_files={"validation": "/path/to/local/validation.arrow"},
            split='validation'
        )
    except:
        # Fallback to remote dataset
        traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
        valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    random.seed(0)
    valenc = []

    # Generate validation data
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])

    valenc = torch.hstack(valenc)

    if test_only:
        return valenc  # Return validation data if test_only is True

    # Prepare train data
    random.seed(seed)
    trainloader = []
    val_sample_ratio = 0.9  # Avoid overlap between training and validation samples

    # Generate training samples
    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata) * val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100  # Shift target for language modeling
        trainloader.append((inp, tar))

    valloader = []

    # Generate validation samples
    for _ in range(val_size):
        while True:
            i = random.randint(int(len(traindata) * val_sample_ratio), len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100  # Shift target for language modeling
        valloader.append((inp, tar))

    return trainloader, valloader


def get_redpajama(
    tokenizer: PreTrainedTokenizer, 
    train_size: int, 
    val_size: int, 
    seed: int, 
    seqlen: int
) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
    """
    Load and preprocess the RedPajama dataset, returning train and validation data.

    Args:
        tokenizer (PreTrainedTokenizer): Tokenizer for encoding the text data.
        train_size (int): Number of training samples to generate.
        val_size (int): Number of validation samples to generate.
        seed (int): Seed for random number generator to ensure reproducibility.
        seqlen (int): Length of the input sequences.

    Returns:
        Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]:
            A tuple of two lists, each containing tuples of input and target tensors for 
            the training and validation datasets.
    """
    print("get_redpajama")
    try:
        loacal_dataset = "/path/to/local/redpajama-dataset"
        traindata = load_dataset(loacal_dataset, split='train')
    except:
        traindata = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split='train')

    random.seed(seed)
    traindata = traindata.shuffle(seed=seed)

    trainloader = []
    val_sample_ratio = 0.9  # 90% for training, 10% for validation

    # Generate training samples
    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata) * val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100  # Shift target for language modeling
        trainloader.append((inp, tar))

    valloader = []

    # Generate validation samples
    for _ in range(val_size):
        while True:
            i = random.randint(int(len(traindata) * val_sample_ratio), len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100  # Shift target for language modeling
        valloader.append((inp, tar))

    return trainloader, valloader


def get_loaders(
    name: str, 
    tokenizer: PreTrainedTokenizer, 
    train_size: int = 128, 
    val_size: int = 64, 
    seed: int = 0, 
    seqlen: int = 2048, 
    test_only: bool = False
) -> Union[torch.Tensor, Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]]:
    """
    Return appropriate dataset loaders based on the dataset name provided.

    Args:
        name (str): The name of the dataset (e.g., 'wikitext2', 'c4', 'redpajama').
        tokenizer (PreTrainedTokenizer): Tokenizer for encoding the text data.
        train_size (int, optional): Number of training samples. Defaults to 128.
        val_size (int, optional): Number of validation samples. Defaults to 64.
        seed (int, optional): Seed for random number generation. Defaults to 0.
        seqlen (int, optional): Sequence length for the inputs. Defaults to 2048.
        test_only (bool, optional): Whether to load only test data. Defaults to False.

    Returns:
        Union[torch.Tensor, Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]]:
            Dataset loaders or validation data depending on the input parameters.
    """
    if 'wikitext2' in name:
        return get_wikitext2(tokenizer, train_size, val_size, seed, seqlen, test_only)
    elif 'c4' in name:
        return get_c4(tokenizer, train_size, val_size, seed, seqlen, test_only)
    elif 'redpajama' in name:
        return get_redpajama(tokenizer, train_size, val_size, seed, seqlen)
    else:
        raise NotImplementedError(f"Dataset {name} is not supported.")


@torch.no_grad()
def test_ppl(
    model: nn.Module, 
    tokenizer, 
    datasets: List[str] = ['wikitext2'], 
    ppl_seqlen: int = 2048
) -> Dict[str, float]:
    """
    Test the perplexity (PPL) of the model on the specified datasets.

    Args:
        model (nn.Module): The language model to test.
        tokenizer: The tokenizer for encoding the datasets.
        datasets (List[str]): List of dataset names to test on.
        ppl_seqlen (int): Sequence length for perplexity calculation.

    Returns:
        Dict[str, float]: A dictionary mapping dataset names to their calculated perplexity.
    """
    results = {}
    for dataset in datasets:
        # Load the dataset
        testloader = get_loaders(
            dataset,
            tokenizer,
            seed=0,
            seqlen=ppl_seqlen,
            test_only=True
        )
        
        # Prepare the input data
        if "c4" in dataset:
            testenc = testloader
        else:
            testenc = testloader.input_ids

        seqlen = ppl_seqlen
        nsamples = testenc.numel() // seqlen

        # Disable caching for evaluation
        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()

        nlls = []

        # Select the classifier for the model
        if hasattr(model, 'lm_head') and isinstance(model.lm_head, nn.Linear):
            classifier = model.lm_head
        elif hasattr(model.model, 'lm_head'):
            # For GPTQ models
            classifier = None
        elif hasattr(model, 'output'):
            # For InternLM models
            classifier = model.output
        else:
            raise NotImplementedError("Model head not implemented.")

        # Evaluate on the test set
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)].to(model.device)
            outputs = model.model(batch)
            
            # Apply the classifier if available
            if classifier is not None:
                hidden_states = outputs[0]
                logits = classifier(hidden_states.to(classifier.weight.dtype))
            else:
                logits = outputs[0]

            shift_logits = logits[:, :-1, :]  # Shift for next token prediction
            shift_labels = testenc[:, (i * seqlen): ((i + 1) * seqlen)][:, 1:].to(shift_logits.device)

            # CrossEntropyLoss for language modeling
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

        # Calculate perplexity
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        print(f'{dataset}: {ppl}')
        results[dataset] = ppl.item()

    # Restore the original cache setting
    model.config.use_cache = use_cache
    return results


class BlockTrainDataset(Dataset):
    def __init__(
        self, 
        size: int, 
        seqlen: int, 
        hidden_size: int, 
        batch_size: int, 
        dtype: torch.dtype, 
        cache_path: str = './cache/block_training_data', 
        off_load_to_disk: bool = False
    ):
        """
        Initialize the BlockTrainDataset for block-wise training data.

        Args:
            size (int): Total size of the dataset.
            seqlen (int): Sequence length for each batch.
            hidden_size (int): Hidden size for each sequence.
            batch_size (int): Number of samples per batch.
            dtype (torch.dtype): Data type for the tensors.
            cache_path (str): Path to store cache files if off_load_to_disk is True.
            off_load_to_disk (bool): If True, data is stored on disk rather than in memory.
        """
        self.size = size
        self.seqlen = seqlen
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.cache_path = cache_path
        self.off_load_to_disk = off_load_to_disk
        self.batch_size = batch_size
        assert size % batch_size == 0, "Size must be divisible by batch_size"

        # Initialize data storage either in-memory or on disk
        if self.off_load_to_disk:
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
                self._initialize_data_on_disk()
        else:
            self.data = torch.zeros(
                (self.size // self.batch_size, self.batch_size, self.seqlen, self.hidden_size), 
                dtype=self.dtype
            )

    def _initialize_data_on_disk(self) -> None:
        """Initialize data files on disk for storing block training data."""
        for idx in range(self.size // self.batch_size):
            tensor = torch.zeros((self.batch_size, self.seqlen, self.hidden_size), dtype=self.dtype)
            filepath = self._get_file_path(idx)
            torch.save(tensor, filepath)

    def _get_file_path(self, idx: int) -> str:
        """Get the file path for the cached data block."""
        return os.path.join(self.cache_path, f"data_{idx}.pt")

    def __len__(self) -> int:
        """Return the total number of batches in the dataset."""
        return self.size // self.batch_size

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieve a batch of data from the dataset.

        Args:
            idx (int): Index of the batch.

        Returns:
            torch.Tensor: The tensor data for the batch.
        """
        if idx >= self.__len__():
            raise IndexError("Index out of range")
        
        if self.off_load_to_disk:
            filepath = self._get_file_path(idx)
            tensor = torch.load(filepath)
        else:
            tensor = self.data[idx]
        
        return tensor

    def update_data(self, idx: int, new_data: torch.Tensor) -> None:
        """
        Update a specific batch of data in the dataset.

        Args:
            idx (int): Index of the batch to update.
            new_data (torch.Tensor): New data to update.
        """
        if self.off_load_to_disk:
            filepath = self._get_file_path(idx)
            torch.save(new_data.to(self.dtype), filepath)
        else:
            self.data[idx] = new_data

import os
import torch
import numpy as np
from torch.utils.data import Dataset

class OptimBlockTrainDataset(Dataset):
    def __init__(self, size, seqlen, hidden_size, batch_size, dtype, cache_path='./cache/block_training_data', off_load_to_disk=False, cache_size=5):
        """
        初始化数据集类
        :param size: 数据集的总大小
        :param seqlen: 序列长度
        :param hidden_size: 每个数据块的隐藏维度大小
        :param batch_size: 批次大小
        :param dtype: 数据类型 (如 torch.float32)
        :param cache_path: 存储缓存的路径
        :param off_load_to_disk: 是否将数据卸载到磁盘
        :param cache_size: 内存中缓存的数据块数量
        """
        self.size = size
        self.seqlen = seqlen
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.cache_path = cache_path
        self.off_load_to_disk = off_load_to_disk
        self.batch_size = batch_size
        self.cache_size = cache_size  # 缓存的大小，用于缓存最近使用的数据块
        self.cache = {}  # 用于缓存最近访问的数据块
        assert size % batch_size == 0, "Size 必须能被 batch_size 整除"
        
        # 定义 numpy 的数据类型，与 torch 对应
        self.np_dtype = np.float32 if dtype == torch.float32 else np.float16
        
        if self.off_load_to_disk:
            # 如果将数据卸载到磁盘，检查缓存路径并初始化内存映射文件
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
            self._initialize_memmap_files()
        else:
            # 如果不卸载到磁盘，则直接在内存中存储数据
            self.data = torch.zeros((self.size // self.batch_size, self.batch_size, self.seqlen, self.hidden_size), dtype=self.dtype)

    def _initialize_memmap_files(self):
        """
        初始化用于存储数据的内存映射文件。
        如果文件不存在，创建一个新的内存映射文件。
        """
        for idx in range(self.size // self.batch_size):
            filepath = self._get_file_path(idx)
            if not os.path.exists(filepath):
                # 创建一个新的内存映射文件
                memmap_data = np.memmap(filepath, dtype=self.np_dtype, mode='w+', shape=(self.batch_size, self.seqlen, self.hidden_size))
                memmap_data.flush()  # 确保数据写入磁盘

    def _get_file_path(self, idx):
        """
        获取指定索引的数据块文件路径。
        :param idx: 数据块的索引
        :return: 数据块文件的完整路径
        """
        return os.path.join(self.cache_path, f"data_{idx}.npy")

    def __len__(self):
        """
        返回数据集的总批次数量。
        :return: 数据集中的批次数
        """
        return self.size // self.batch_size

    def __getitem__(self, idx):
        """
        获取指定索引处的数据块。
        首先检查缓存，如果缓存中有该数据块，直接返回。
        如果缓存中没有，则从内存映射文件或内存中读取数据。
        :param idx: 数据块的索引
        :return: 索引处的数据块（torch.Tensor 格式）
        """
        if idx >= self.__len__():
            raise IndexError("索引超出范围")
        
        # 检查数据块是否在缓存中
        if idx in self.cache:
            return self.cache[idx]
        
        if self.off_load_to_disk:
            # 如果数据卸载到磁盘，从内存映射文件中加载数据
            filepath = self._get_file_path(idx)
            memmap_data = np.memmap(filepath, dtype=self.np_dtype, mode='r', shape=(self.batch_size, self.seqlen, self.hidden_size))
            
            # 复制 memmap 数据以确保可写性
            memmap_data = np.copy(memmap_data)
            tensor = torch.from_numpy(memmap_data).to(self.dtype)
        else:
            # 如果数据在内存中，直接获取
            tensor = self.data[idx]

        # 将数据加入缓存，并保持缓存大小不超过指定的大小
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))  # 删除最早缓存的数据块
        self.cache[idx] = tensor

        return tensor


    def update_data(self, idx, new_data):
        """
        更新指定索引处的数据块。
        如果数据存储在磁盘，则更新内存映射文件中的数据。
        否则，直接更新内存中的数据。
        :param idx: 数据块的索引
        :param new_data: 新的数据块（torch.Tensor 格式）
        """
        new_data = new_data.to(self.dtype) if new_data.dtype != self.dtype else new_data
        
        if self.off_load_to_disk:
            # 更新内存映射文件中的数据
            filepath = self._get_file_path(idx)
            memmap_data = np.memmap(filepath, dtype=self.np_dtype, mode='r+', shape=(self.batch_size, self.seqlen, self.hidden_size))
            memmap_data[:] = new_data.cpu().numpy()  # 将新的数据写入内存映射文件
            memmap_data.flush()  # 确保更改写入磁盘
        else:
            # 直接更新内存中的数据
            self.data[idx] = new_data
