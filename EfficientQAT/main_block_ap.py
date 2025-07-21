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
from .liger_kernel_utils import apply_liger_kernel

amp_enabled = os.environ.get("AMP_ENABLED", "False").lower() == "true"
torch.set_float32_matmul_precision('high')

@torch.no_grad()
def evaluate(model: Any, tokenizer: AutoTokenizer, config: Dict[str, Any], logger: Optional[Any] = None) -> Dict[str, Any]:
    block_class_name = model.model.layers[0].__class__.__name__
    results = {}
    if config["train_param_settings"]["eval_ppl"]:
        datasets = ["wikitext2", "c4"]
        ppl_results = test_ppl(model, tokenizer, datasets, config.get('ppl_seqlen', 2048))
        for dataset in ppl_results:
            if logger is not None:
                logger.info(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')
            else:
                print(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')
    if config['train_param_settings'].get('eval_tasks', "") != "":
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table
        task_list = config['train_param_settings']['eval_tasks'].split(',')
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

def set_rng_env(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="path of config file", required=True)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parser.parse_args()
    config = load_config(args.config_path)
    config = update_config_with_args(config, args)
    set_rng_env(config.get('seed', 42))
    if config.get('output_dir'):
        Path(config['train_param_settings']['output_dir']).mkdir(parents=True, exist_ok=True)
    if config.get('cache_dir'):
        Path(config['train_param_settings']['cache_dir']).mkdir(parents=True, exist_ok=True)
    if config.get('save_dir'):
        Path(config['save_dir']).mkdir(parents=True, exist_ok=True)
    if config['train_param_settings']['cache_dir']:
        Path(config['train_param_settings']['cache_dir']).mkdir(parents=True, exist_ok=True)
    output_dir = Path(config.get('output_dir', './log/'))
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    logger = utils.create_logger(output_dir)
    config['logger'] = logger
    logger.info(config)
    if config.get('net') is None:
        config['model_settings']['net'] = config['model_settings']['model'].split('/')[-1]
        logger.info(f"net is None, setting as {config['model_settings']['net']}")
    if config.get('resume_quant'):
        model, tokenizer = load_quantized_model(config['train_param_settings']['resume_quant'], config['hyperparam_settings']['wbits'], config['hyperparam_settings']['group_size'])
        logger.info(f"memory footprint after loading quantized model: {torch.cuda.max_memory_allocated('cuda') / 1024**3:.2f}GiB")
    else:
        model_config = AutoConfig.from_pretrained(config['model_settings']['model'])
        tokenizer = AutoTokenizer.from_pretrained(config['model_settings']['model'], use_fast=False, legacy=False)
        # apply_liger_kernel(
        #     config=model_config, 
        #     require_logits=True)
        model = AutoModelForCausalLM.from_pretrained(
            config['model_settings']['model'],
            attn_implementation="eager",
            config=model_config,
            device_map='cpu',
            torch_dtype=torch.float16 
        )
        for param in model.parameters():
            param.requires_grad = False
        if config['hyperparam_settings']['wbits'] < 16:
            logger.info("=== start quantization ===")
            tick = time.time()
            cache_trainloader = f"{config['train_param_settings']['cache_dir']}/dataloader_{config['model_settings']['net']}_{config['train_param_settings']['calib_dataset']}_{config['train_param_settings']['train_size']}_{config['train_param_settings']['val_size']}_{config['train_param_settings']['training_seqlen']}_train.cache"
            cache_valloader = f"{config['train_param_settings']['cache_dir']}/dataloader_{config['model_settings']['net']}_{config['train_param_settings']['calib_dataset']}_{config['train_param_settings']['train_size']}_{config['train_param_settings']['val_size']}_{config['train_param_settings']['training_seqlen']}_val.cache"
            if os.path.exists(cache_trainloader) and os.path.exists(cache_valloader):
                trainloader = torch.load(cache_trainloader, weights_only=True)
                logger.info(f"load trainloader from {cache_trainloader}")
                valloader = torch.load(cache_valloader, weights_only=True)
                logger.info(f"load valloader from {cache_valloader}")
            else:
                trainloader, valloader = get_loaders(
                    config['train_param_settings']['calib_dataset'],
                    tokenizer,
                    config['train_param_settings']['train_size'],
                    config['train_param_settings']['val_size'],
                    seed=config['train_param_settings']['seed'],
                    seqlen=config['train_param_settings']['training_seqlen'],
                )
                torch.save(trainloader, cache_trainloader)
                torch.save(valloader, cache_valloader)
            if config['train_param_settings']['quant_method'] == "block_ap":
                greedy_local_train(model, config, trainloader, valloader, logger)
            elif config['train_param_settings']['quant_method'] == "awq":
                from .quantize.awq_pipeline import awq_pipline
                awq_pipline(model, trainloader, config)
            elif config['train_param_settings']['quant_method'] == "gptq":
                from .quantize.gptq_pipeline import gptq_pipeline
                model = gptq_pipeline(model, trainloader, config)
            elif config['train_param_settings']['quant_method'] == "aqlm":
                from .quantize.aqlm_pipeline import aqlm_pipeline
                _, model = aqlm_pipeline(model, trainloader, valloader, config)
            else:
                raise NotImplementedError(f"quantization method {config['train_param_settings']['quant_method']} not implemented")
            logger.info(time.time() - tick)
    torch.cuda.empty_cache()
    if config['save_dir'] is not None:
        logger.info("start saving model")
        if config['train_param_settings']['quant_method'] == "gptq":
            model.save(config['save_dir'])
        else:
            model.save_pretrained(config['save_dir'])
        tokenizer.save_pretrained(config['save_dir'])
        logger.info("save model success")
    model.to(config["cluster_settings"]['cuda_ids'][0])
    evaluate(model, tokenizer, config, logger)

if __name__ == "__main__":
    main()
