import os
import sys
import random
import yaml
import time
from tqdm import tqdm
from pathlib import Path

from easydict import EasyDict 
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from accelerate import infer_auto_device_map, dispatch_model


from . import utils
from .datautils_block import get_loaders, test_ppl
from .quantize.int_linear_real import load_quantized_model
from .quantize.block_ap import block_ap
from .quantize.crossblockquant import cross_block_quantization
from .quantize.greedy_trainer import greedy_local_train


amp_enabled = os.environ.get("AMP_ENABLED", "False").lower() == "true"
torch.backends.cudnn.benchmark = True

@torch.no_grad()
def evaluate(model, tokenizer, args, logger=None):
    '''
    Note: evaluation simply move model to single GPU. 
    Therefor, to evaluate large model such as Llama-2-70B on single A100-80GB,
    please activate '--real_quant'.
    '''
    # import pdb;pdb.set_trace()
    block_class_name = model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(model, max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, no_split_module_classes=[block_class_name])
    if logger is not None:
        logger.info(f"device_map: {device_map}")
    else:
        print(f"device_map: {device_map}")
    model = dispatch_model(model, device_map=device_map)
    results = {}

    if args.eval_ppl:
        datasets = ["wikitext2", "c4"]
        ppl_results = test_ppl(model, tokenizer, datasets, args.ppl_seqlen)
        for dataset in ppl_results:
            if logger is not None:
                logger.info(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')
            else: 
                print(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')

    if args.eval_tasks != "":
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table
        task_list = args.eval_tasks.split(',')
        model = HFLM(pretrained=model, batch_size=args.eval_batch_size)
        task_manager = lm_eval.tasks.TaskManager()
        results = lm_eval.simple_evaluate(
        model=model,
        tasks=task_list,
        num_fewshot=0,
        task_manager=task_manager,
        )
        if logger is not None:
            logger.info(make_table(results))
        else:
            print(make_table(results))
        total_acc = 0
        for task in task_list:
            total_acc += results['results'][task]['acc,none']
        if logger is not None:
            logger.info(f'Average Acc: {total_acc/len(task_list)*100:.2f}%')
        else:
            print(f'Average Acc: {total_acc/len(task_list)*100:.2f}%')
    return results

def load_config(yaml_path):
    """
    从 YAML 文件加载配置
    """
    if yaml_path is None:
        return {}
    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)
    return EasyDict(config)

def update_config_with_args(config, args):
    """
    用 argparse 的参数更新配置，支持嵌套的参数更新，优先使用 config 中的值
    """
    for key, value in vars(args).items():
        # 只有当命令行提供了非 None 参数时才更新配置
        if value is not None:
            if '.' in key:  # 处理嵌套的配置
                top_key, sub_key = key.split('.')
                if top_key in config and isinstance(config[top_key], dict):
                    # 如果命令行中提供了非 None 的值，更新配置
                    config[top_key][sub_key] = value
            else:
                if not config.get(key):
                    # 如果命令行中提供了非 None 的值，更新配置
                    config[key] = value
        # 如果命令行参数存在且为 None，保持 config 中的默认值
        elif value is None and key not in config:
            # 如果 config 中没有该 key，则在 config 中加入该 key 的 None 值
            config[key] = value
    if "cuda_ids" in config:
        os.environ["CUDA_VISIBLE_DEVICES"]=config.cuda_ids
    return config



def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="path of config file")
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="direction of cached dataset, leading to faster debug")
    parser.add_argument("--output_dir", default="./log/", type=str, help="direction of logging file")
    parser.add_argument("--save_quant_dir", default=None, type=str, help="direction for saving quantization model")
    parser.add_argument("--real_quant", default=False, action="store_true",
                        help="use real quantization instead of fake quantization, can reduce memory footprint")
    parser.add_argument("--resume_quant", type=str, default=None,  help="model path of resumed quantized model")
    parser.add_argument("--calib_dataset",type=str,default="redpajama",
        choices=["wikitext2", "ptb", "c4", "mix", "redpajama"],
        help="Where to extract calibration data from.")
    parser.add_argument("--train_size", type=int, default=4096, help="Number of training data samples.")
    parser.add_argument("--val_size", type=int, default=64, help="Number of validation data samples.")
    parser.add_argument("--training_seqlen", type=int, default=2048, help="lenth of the training sequence.")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--ppl_seqlen", type=int, default=2048, help="input sequence length for evaluating perplexity")
    parser.add_argument("--seed", type=int, default=2, help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true",help="evaluate perplexity on wikitext2 and c4")
    parser.add_argument("--eval_tasks", type=str,default="", help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--wbits", type=int, default=4, help="weights quantization bits")
    parser.add_argument("--group_size", type=int, default=128, help="weights quantization group size")
    parser.add_argument("--quant_lr", type=float, default=1e-4, help="lr of quantization parameters (s and z)")
    parser.add_argument("--weight_lr", type=float, default=1e-5, help="lr of full-precision weights")
    parser.add_argument("--min_lr_factor", type=float, default=20, help="min_lr = lr/min_lr_factor")
    parser.add_argument("--clip_grad", type=float, default=0.3)
    parser.add_argument("--wd", type=float, default=0,help="weight decay")
    parser.add_argument("--net", type=str, default=None,help="model (family) name, for the easier saving of data cache")
    parser.add_argument("--max_memory", type=str, default="70GiB",help="The maximum memory of each GPU")
    parser.add_argument("--early_stop", type=int, default=0,help="early stoping after validation loss do not decrease")
    parser.add_argument("--off_load_to_disk", action="store_true", default=False, help="save training dataset to disk, saving CPU memory but may reduce training speed")
    parser.add_argument("--log_loss" , type=str, default=None , help="log loss path")
    parser.add_argument("--loss_func", type=str,
                        choices=["MSE", "FKLD" , "RKLD", "FKLD_RKLD" ,"MSE_FKLD", "MSE_RKLD", "MSE_FKLD_RKLD"],
                          default="MSE", help="loss function for training")
    
    parser.add_argument("--clamp_method", type=str, default="STE", help="clamp method for training")
    parser.add_argument("--quant_shedule_type", type=str, default="partial", help="quantization shedule type")
    parser.add_argument("--train_shedule_type", type=str, default="start2end", help="train shedule type")

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()
    config = load_config(args.config_path)
    args = update_config_with_args(config, args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

        
    # init logger
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_quant_dir:
        Path(args.save_quant_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)
    
    if args.net is None:
        args.net = args.model.split('/')[-1]
        logger.info(f"net is None, setting as {args.net}")
    if args.resume_quant:
        # directly load quantized model for evaluation
        model, tokenizer = load_quantized_model(args.resume_quant,args.wbits, args.group_size)
        logger.info(f"memory footprint after loading quantized model: {torch.cuda.max_memory_allocated('cuda') / 1024**3:.2f}GiB")
    else:
        # load fp quantized model
        config = AutoConfig.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False,legacy=False)
        model = AutoModelForCausalLM.from_pretrained(args.model,
                                        config=config,
                                        device_map='cpu',
                                        torch_dtype=torch.float16 if amp_enabled else torch.float32)
        for param in model.parameters():
            param.requires_grad = False

        # quantization
        if args.wbits < 16:
            logger.info("=== start quantization ===")
            tick = time.time()     
            # load calibration dataset
            cache_trainloader = f'{args.cache_dir}/dataloader_{args.net}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_train.cache'
            cache_valloader = f'{args.cache_dir}/dataloader_{args.net}_{args.calib_dataset}_{args.train_size}_{args.val_size}_{args.training_seqlen}_val.cache'
            if os.path.exists(cache_trainloader) and os.path.exists(cache_valloader):
                trainloader = torch.load(cache_trainloader,weights_only=True)
                logger.info(f"load trainloader from {cache_trainloader}")
                valloader = torch.load(cache_valloader,weights_only=True)
                logger.info(f"load valloader from {cache_valloader}")
            else:
                trainloader, valloader = get_loaders(
                    args.calib_dataset,
                    tokenizer,
                    args.train_size,
                    args.val_size,
                    seed=args.seed,
                    seqlen=args.training_seqlen,
                )
                torch.save(trainloader, cache_trainloader)    
                torch.save(valloader, cache_valloader)    
            # cross_block_quantization(
            # block_ap(
            greedy_local_train(
                model,
                args,
                trainloader,
                valloader,
                logger,
            )
            logger.info(time.time() - tick)
    torch.cuda.empty_cache()
    if args.save_quant_dir:
        logger.info("start saving model")
        model.save_pretrained(args.save_quant_dir)  
        tokenizer.save_pretrained(args.save_quant_dir) 
        logger.info("save model success")
    evaluate(model, tokenizer, args,logger)



if __name__ == "__main__":
    # print(sys.argv)
    main()
