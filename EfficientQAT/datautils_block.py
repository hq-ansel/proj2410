import os
import random
from tqdm import tqdm
from typing import List, Tuple,Union,Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer,PreTrainedTokenizer,PreTrainedModel
from datasets import load_dataset
import numpy as np

from .quantize.utils import StopException,Catcher

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        初始化Dataset

        参数:
            data_dir (str): 数据文件的路径
            transform (callable, optional): 应用于数据的转换函数
        """
        self.data_dir = data_dir
        self.file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
        self.transform = transform

    def __len__(self):
        """返回数据集中样本的数量"""
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        根据索引idx加载并返回样本数据

        参数:
            idx (int): 数据索引
        返回:
            sample: 数据样本，经过transform后（如果存在transform）
        """
        # 读取 .pt 文件
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        sample = torch.load(file_path)

        # 应用 transform（如果有）
        if self.transform:
            sample = self.transform(sample)

        return sample


@torch.no_grad()
def generate_block_train_data(
        model: PreTrainedModel,
        dataloader: DataLoader,
        out_dir: str,
):
    """
    第一层存储input,output
    第二层以及后面的层数仅存储output
    """
    device = next(model.parameters()).device
    layers = dict(model.named_modules()).get("model.layers",None)
    assert layers is not None, "model.layers not found"
    for idx in range(len(layers)):
        layers[idx] = Catcher(layers[idx])
        if idx == 0:
            layers[idx].set_store_state(store_input=True, store_output=True)
        else:
            layers[idx].set_store_state(store_input=False, store_output=True)   
        layers[idx].setup_executor(2)
        layers[idx].set_store_dir(out_dir)
        layers[idx].set_layer_idx(idx)

    print(f"len of dataloader: {len(dataloader)}")
    for idx, batch in tqdm(enumerate(dataloader)):
        inp, tar = batch
        model(inp.to(device))
    for idx in range(len(layers)):
        layers[idx] = layers[idx].module


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
