from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np
import torch
import random
from tqdm import tqdm
import torch.nn as nn

import torch
from torch.utils.data import Dataset
import os

def get_wikitext2(tokenizer, train_size, val_size, seed, seqlen, test_only):
    print("get_wikitext2")
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    if test_only:
        return testenc
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')

    
    random.seed(seed)
    trainloader = []
    val_sample_ratio = 0.9  # sample train from [0:0.9] and val from [0.9:1.0] to avoid overlap
    for _ in range(train_size):
        i = random.randint(0, int(trainenc.input_ids.shape[1]*val_sample_ratio) - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    valloader = []
    for _ in range(val_size):
        i = random.randint(int(trainenc.input_ids.shape[1]*val_sample_ratio) - seqlen - 1, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))
    return trainloader, valloader


def get_c4(tokenizer, train_size, val_size, seed, seqlen, test_only):
    print("get_c4")
    try:
        # set local path for faster loading
        traindata = load_dataset("arrow",
                    data_files={
                        "train": "/cpfs01/user/chenmengzhao/huggingface/datasets/allenai___json/allenai--c4-6fbe877195f42de5/0.0.0/0f7e3662623656454fcd2b650f34e886a7db4b9104504885bd462096cc7a9f51/json-train-00000-of-00002.arrow",
                        "validation": "/cpfs01/user/chenmengzhao/huggingface/datasets/allenai___json/allenai--c4-efc3d4f4606f44bd/0.0.0/fe5dd6ea2639a6df622901539cb550cf8797e5a6b2dd7af1cf934bed8e233e6e/json-validation.arrow",
                    },split='train'
                    )
        valdata = load_dataset("arrow",
                    data_files={
                        "validation": "/cpfs01/user/chenmengzhao/huggingface/datasets/allenai___json/allenai--c4-efc3d4f4606f44bd/0.0.0/fe5dd6ea2639a6df622901539cb550cf8797e5a6b2dd7af1cf934bed8e233e6e/json-validation.arrow",
                    },split='validation'
                    )
    except:
        traindata = load_dataset(
            'allenai/c4',  data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train'
        )
        valdata = load_dataset(
            'allenai/c4',  data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation'
        )

    random.seed(0)
    valenc = []
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
        return valenc 

    random.seed(seed)
    trainloader = []
    val_sample_ratio = 0.9  # sample train from [0:0.9] and val from [0.9:1.0] to avoid overlap
    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata)*val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen+1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    
    valloader = []
    for _ in range(val_size):
        while True:
            i = random.randint(int(len(traindata)*val_sample_ratio),len(traindata)-1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen+1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))



    return trainloader, valloader 

def get_redpajama(tokenizer, train_size, val_size, seed, seqlen):
    print("get_redpajama")
    try:
        loacal_dataset = "/cpfs01/user/chenmengzhao/huggingface/datasets/togethercomputer___red_pajama-data-1_t-sample"
        traindata = load_dataset(loacal_dataset,split='train')   
    except:
        traindata = load_dataset("togethercomputer/RedPajama-Data-1T-Sample",split='train')   
    random.seed(seed)
    traindata = traindata.shuffle(seed=seed) 
    trainloader = []
    val_sample_ratio = 0.9
    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata)*val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen+1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valloader = []
    for _ in range(val_size):
        while True:
            i = random.randint(int(len(traindata)*val_sample_ratio),len(traindata)-1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen+1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))
    return trainloader, valloader



def get_loaders(
    name, tokenizer, train_size=128, val_size=64,seed=0, seqlen=2048, test_only=False
):
    if 'wikitext2' in name:
        return get_wikitext2(tokenizer,train_size,val_size,seed,seqlen,test_only)
    elif 'c4' in name:
        return get_c4(tokenizer,train_size,val_size,seed,seqlen,test_only)
    elif 'redpajama' in name:
        return get_redpajama(tokenizer,train_size,val_size,seed,seqlen)
    else:
        raise NotImplementedError



@torch.no_grad()
def test_ppl(model, tokenizer, datasets=['wikitext2'],ppl_seqlen=2048):
    results = {}
    for dataset in datasets:
        testloader = get_loaders(
            dataset,
            tokenizer,
            seed=0,
            seqlen=ppl_seqlen,
            test_only=True
        )
        if "c4" in dataset:
            testenc = testloader
        else:
            testenc = testloader.input_ids

        seqlen = ppl_seqlen
        nsamples = testenc.numel() // seqlen
        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()
        nlls = []
        if hasattr(model,'lm_head') and isinstance(model.lm_head, nn.Linear):
            classifier = model.lm_head
        elif hasattr(model.model,'lm_head'):
            # for gptqmodels
            classifier = None
        elif hasattr(model,'output'):
            # for internlm
            classifier = model.output
        else:
            raise NotImplementedError
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(model.device)
            outputs = model.model(batch)
            if classifier is not None:
                hidden_states = outputs[0]
                logits = classifier(hidden_states.to(classifier.weight.dtype))
            else:
                logits = outputs[0]
            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][
                :, 1:
            ].to(shift_logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)


        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        print(f'{dataset}:{ppl}')
        results[dataset] = ppl.item()
    model.config.use_cache = use_cache
    return results

class BlockTrainDataset(Dataset):
    def __init__(self, size, seqlen, hidden_size, batch_size, dtype, cache_path='./cache/block_training_data', off_load_to_disk=False):
        self.size = size
        self.seqlen = seqlen
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.cache_path = cache_path
        self.off_load_to_disk = off_load_to_disk
        self.batch_size = batch_size
        assert size%batch_size == 0
         
        if self.off_load_to_disk:
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
                self._initialize_data_on_disk()
        else:
            self.data = torch.zeros((self.size//self.batch_size, self.batch_size, self.seqlen, self.hidden_size), dtype=self.dtype)

    def _initialize_data_on_disk(self):
        for idx in range(self.size//self.batch_size):
            tensor = torch.zeros((self.batch_size, self.seqlen, self.hidden_size), dtype=self.dtype)
            filepath = self._get_file_path(idx)
            torch.save(tensor, filepath)

    def _get_file_path(self, idx):
        return os.path.join(self.cache_path, f"data_{idx}.pt")

    def __len__(self):
        return self.size//self.batch_size

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("Index out of range")
        if self.off_load_to_disk:
            filepath = self._get_file_path(idx)
            tensor = torch.load(filepath)
        else:
            tensor = self.data[idx]
        return tensor

    def update_data(self, idx, new_data):
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
