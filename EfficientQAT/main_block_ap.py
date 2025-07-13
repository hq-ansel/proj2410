import os
import sys
import random
import yaml
import time
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.configuration_auto import AutoConfig
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from accelerate import infer_auto_device_map, dispatch_model
from . import utils
from .datautils_block import get_loaders, test_ppl
from .quantize.int_linear_real import load_quantized_model  # 假设修复了模块
from .quantize.block_ap import block_ap
from .quantize.crossblockquant import cross_block_quantization
from .quantize.greedy_trainer import greedy_local_train, timer

amp_enabled = os.environ.get("AMP_ENABLED", "False").lower() == "true"
torch.set_float32_matmul_precision('high')

@torch.no_grad()
def evaluate(model: Any, tokenizer: AutoTokenizer, config: Dict[str, Any], logger: Optional[Any] = None) -> Dict[str, Any]:
    block_class_name = model.model.layers[0].__class__.__name__
    results = {}
    if config.get('eval_ppl', False):
        datasets = ["wikitext2", "c4"]
        ppl_results = test_ppl(model, tokenizer, datasets, config.get('ppl_seqlen', 2048))
        for dataset in ppl_results:
            if logger is not None:
                logger.info(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')
            else:
                print(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')
    if config.get('eval_tasks', "") != "":
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table
        task_list = config['eval_tasks'].split(',')
        model_eval = HFLM(pretrained=model, batch_size=config.get('eval_batch_size', 16))
        print(f"Evaluating on tasks: {task_list}")
        task_manager = lm_eval.tasks.TaskManager()
        eval_results = lm_eval.simple_evaluate(
            model=model_eval,
            tasks=task_list,
            num_fewshot=0,
            task_manager=task_manager,
        )
        if logger is not None:
            logger.info(make_table(eval_results))
        else:
            print(make_table(eval_results))
        total_acc = 0
        for task in task_list:
            if eval_results and 'results' in eval_results and task in eval_results['results']:
                total_acc += eval_results['results'][task].get('acc,none', 0)
        avg_acc = total_acc / len(task_list) * 100 if task_list else 0
        if logger is not None:
            logger.info(f'Average Acc: {avg_acc:.2f}%')
        else:
            print(f'Average Acc: {avg_acc:.2f}%')
    return results

def load_config(yaml_path: str) -> Dict[str, Any]:
    if yaml_path is None:
        return {}
    with open(yaml_path, 'r') as file:
        return yaml.safe_load(file)

def update_config_with_args(config: Dict[str, Any], args: Any) -> Dict[str, Any]:
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    return config

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="path of config file", required=True)
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--cache_dir", default="/home/ubuntu/data/exp/proj2410/cache", type=str, help="direction of cached dataset, leading to faster debug")
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
    parser.add_argument("--seed", type=int, default=42, help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true",help="evaluate perplexity on wikitext2 and c4")
    parser.add_argument("--eval_tasks", type=str,default="", help="exampe:piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--wbits", type=int, default=4, help="weights quantization bits")
    parser.add_argument("--group_size", type=int, default=128, help="weights quantization group size")
    parser.add_argument("--quant_lr", type=float, default=1e-4, help="lr of quantization parameters (s and z)")
    parser.add_argument("--weight_lr", type=float, default=1e-5, help="lr of full-precision weights")
    parser.add_argument("--min_lr_factor", type=float, default=10, help="min_lr = lr/min_lr_factor")
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
    parser.add_argument("--with_catcher", action="store_true", default=False, help="use catcher for training saving memory")
    parser.add_argument("--quant_method", type=str, default="block_ap", help="quantization method")

    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()
    config = load_config(args.config_path)
    config = update_config_with_args(config, args)
    random.seed(config.get('seed', 42))
    np.random.seed(config.get('seed', 42))
    torch.manual_seed(config.get('seed', 42))
    torch.cuda.manual_seed(config.get('seed', 42))
    torch.cuda.manual_seed_all(config.get('seed', 42))
    os.environ['PYTHONHASHSEED'] = str(config.get('seed', 42))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if config.get('output_dir'):
        Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    if config.get('cache_dir'):
        Path(config['cache_dir']).mkdir(parents=True, exist_ok=True)
    if config.get('save_quant_dir'):
        Path(config['save_quant_dir']).mkdir(parents=True, exist_ok=True)
    output_dir = Path(config.get('output_dir', './log/'))
    logger = utils.create_logger(output_dir)
    config['logger'] = logger
    logger.info(config)
    if config.get('net') is None:
        config['net'] = config['model'].split('/')[-1]
        logger.info(f"net is None, setting as {config['net']}")
    if config.get('resume_quant'):
        model, tokenizer = load_quantized_model(config['resume_quant'], config['wbits'], config['group_size'])
        logger.info(f"memory footprint after loading quantized model: {torch.cuda.max_memory_allocated('cuda') / 1024**3:.2f}GiB")
    else:
        model_config = AutoConfig.from_pretrained(config['model'])
        tokenizer = AutoTokenizer.from_pretrained(config['model'], use_fast=False, legacy=False)
        model = AutoModelForCausalLM.from_pretrained(
            config['model'],
            attn_implementation="eager",
            config=model_config,
            device_map='cpu',
            torch_dtype=torch.float16 if amp_enabled else torch.float32
        )
        for param in model.parameters():
            param.requires_grad = False
        if config['wbits'] < 16:
            logger.info("=== start quantization ===")
            tick = time.time()
            cache_trainloader = f"{config['cache_dir']}/dataloader_{config['net']}_{config['calib_dataset']}_{config['train_size']}_{config['val_size']}_{config['training_seqlen']}_train.cache"
            cache_valloader = f"{config['cache_dir']}/dataloader_{config['net']}_{config['calib_dataset']}_{config['train_size']}_{config['val_size']}_{config['training_seqlen']}_val.cache"
            if os.path.exists(cache_trainloader) and os.path.exists(cache_valloader):
                trainloader = torch.load(cache_trainloader, weights_only=True)
                logger.info(f"load trainloader from {cache_trainloader}")
                valloader = torch.load(cache_valloader, weights_only=True)
                logger.info(f"load valloader from {cache_valloader}")
            else:
                trainloader, valloader = get_loaders(
                    config['calib_dataset'],
                    tokenizer,
                    config['train_size'],
                    config['val_size'],
                    seed=config['seed'],
                    seqlen=config['training_seqlen'],
                )
                torch.save(trainloader, cache_trainloader)
                torch.save(valloader, cache_valloader)
            if config['quant_method'] == "block_ap":
                greedy_local_train(model, config, trainloader, valloader, logger)
            elif config['quant_method'] == "awq":
                from .quantize.awq_pipeline import awq_pipline
                awq_pipline(model, trainloader, config)
            elif config['quant_method'] == "gptq":
                from .quantize.gptq_pipeline import gptq_pipeline
                model = gptq_pipeline(model, trainloader, config)
            elif config['quant_method'] == "aqlm":
                from .quantize.aqlm_pipeline import aqlm_pipeline
                _, model = aqlm_pipeline(model, trainloader, valloader, config)
            else:
                raise NotImplementedError(f"quantization method {config['quant_method']} not implemented")
            logger.info(time.time() - tick)
    torch.cuda.empty_cache()
    if config.get('save_quant_dir'):
        logger.info("start saving model")
        if config['quant_method'] == "gptq":
            model.save(config['save_quant_dir'])
        else:
            model.save_pretrained(config['save_quant_dir'])
        tokenizer.save_pretrained(config['save_quant_dir'])
        logger.info("save model success")
    model.to(config['cuda'][0])
    evaluate(model, tokenizer, config, logger)

if __name__ == "__main__":
    main()
