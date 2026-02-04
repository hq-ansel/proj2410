import os
import sys
import json
import re
from pathlib import Path
# 将VeOmni的读取代码加入路径
REPO_ROOT = Path(__file__).resolve().parents[1]
QUANT_TASKS_PATH = REPO_ROOT / "VeOmni" / "tasks" / "quantize"
if str(QUANT_TASKS_PATH) not in sys.path:
    sys.path.append(str(QUANT_TASKS_PATH))
HF_HOME = REPO_ROOT / "hf_home"
os.environ.setdefault("HF_HOME", str(HF_HOME))

import accelerate
import torch
import wandb
from easydict import EasyDict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from EfficientQAT.main_block_ap import evaluate

import load_tritonv2_quant  # noqa: E402


print("HF_HOME:", HF_HOME)

def load_quantized_model(path: str):
    model, tokenizer = load_tritonv2_quant.load_tritonv2_quantized_model(
        model_dir=path,
        device="cuda",
        dtype="float16",
    )
    return model, tokenizer


def _load_quantize_config(path: str):
    cand = Path(path) / "quantize_config.json"
    if cand.is_file():
        with open(cand, "r", encoding="utf-8") as f:
            return json.load(f)
    # If path is hf_ckpt or nested, try to find sibling checkpoints/out
    for parent in Path(path).parents:
        alt = parent / "out" / "quantize_config.json"
        if alt.is_file():
            with open(alt, "r", encoding="utf-8") as f:
                return json.load(f)
    return None


def _infer_bits_group_from_path(path: str):
    bits = None
    group_size = None
    m = re.search(r"w(\d+)g(\d+)", path)
    if m:
        bits = int(m.group(1))
        group_size = int(m.group(2))
    if bits is None:
        m = re.search(r"int(\d+)", path)
        if m:
            bits = int(m.group(1))
    if bits is None:
        bits = 8
    if group_size is None:
        group_size = 128
    return bits, group_size


def _find_latest_hf_ckpt(quant_path: str):
    p = Path(quant_path).resolve()
    checkpoints_dir = None
    for parent in p.parents:
        if parent.name == "checkpoints":
            checkpoints_dir = parent
            break
    if checkpoints_dir is None or not checkpoints_dir.is_dir():
        return None
    candidates = []
    for child in checkpoints_dir.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith("global_step_"):
            continue
        try:
            step = int(child.name.split("_")[-1])
        except ValueError:
            continue
        hf = child / "hf_ckpt"
        if hf.is_dir():
            candidates.append((step, hf))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return str(candidates[-1][1])


def _convert_linear_with_skip(module, prefix, config, skip_names):
    from EfficientQAT.core.linear.int_quant_linear import IntQuantLinear
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        if child_prefix in skip_names or child_prefix.endswith(tuple(f".{n}" for n in skip_names)):
            continue
        if isinstance(child, torch.nn.Linear) and not isinstance(child, IntQuantLinear):
            setattr(module, name, IntQuantLinear.from_float(child_prefix, child, config))
        else:
            _convert_linear_with_skip(child, child_prefix, config, skip_names)


def load_qat_hf_ckpt_model(quant_path: str, qcfg: dict | None):
    from EfficientQAT.core.quantizer.config import QuantConfig as EQuantConfig
    from EfficientQAT.core.linear.int_quant_linear import set_quant_state

    bits = None
    group_size = None
    sym = False
    if qcfg:
        bits = qcfg.get("bits")
        group_size = qcfg.get("group_size")
        sym = bool(qcfg.get("sym", False))
    if bits is None or group_size is None:
        bits, group_size = _infer_bits_group_from_path(quant_path)

    qconfig = EQuantConfig(
        quant_type="uniform_affine",
        n_bits=int(bits),
        group_size=int(group_size),
        symmetric=bool(sym),
    )
    skip_names = {"lm_head"}

    with accelerate.init_empty_weights():
        config = AutoConfig.from_pretrained(
            quant_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        try:
            model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    _convert_linear_with_skip(model, prefix="", config=qconfig, skip_names=skip_names)
    set_quant_state(model, weight_quant=True)
    model.tie_weights()

    device_map = {"": "cuda"}
    try:
        accelerate.load_checkpoint_in_model(
            model,
            checkpoint=quant_path,
            device_map=device_map,
            dtype=torch.float16,
            offload_state_dict=True,
        )
    except TypeError:
        accelerate.load_checkpoint_in_model(
            model,
            checkpoint=quant_path,
            device_map=device_map,
            offload_state_dict=True,
        )
    model = accelerate.dispatch_model(model, device_map=device_map)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(quant_path, local_files_only=True, trust_remote_code=True)
    return model, tokenizer


def qtip_model_from_hf_path(path, max_mem_ratio=0.7, device_map=None):
    from opponent.qtip.model.llama import LlamaForCausalLM

    if device_map is None:
        mmap = {
            i: f"{torch.cuda.mem_get_info(i)[1] * max_mem_ratio / (1 << 30)}GiB"
            for i in range(torch.cuda.device_count())
        }
        model = LlamaForCausalLM.from_pretrained(
            path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2",
        )
        device_map = accelerate.infer_auto_device_map(
            model,
            no_split_module_classes=["LlamaDecoderLayer"],
            max_memory=mmap,
        )
    model = LlamaForCausalLM.from_pretrained(
        path,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        device_map=device_map,
    )
    return model


def gptq_model_from_path(path, _wbit=2):
    from gptqmodel import GPTQModel
    model = GPTQModel.load(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer


# CUDA_VISIBLE_DEVICES=0,1,2 python -m test.eval_batch
# CUDA_VISIBLE_DEVICES=0,1,2,3  python -m test.eval_batch > exp.logs
# CUDA_VISIBLE_DEVICES=0  python -m test.eval_batch > expfp16.logs
# CUDA_VISIBLE_DEVICES=0 python -m test.eval_batch >> exp_res.logs


def build_eval_args():
    args = EasyDict()
    args["train_param_settings"] = {}
    train_params = args["train_param_settings"]
    train_params["eval_ppl"] = True
    train_params["max_memory"] = "24GB"
    train_params["ppl_seqlen"] = 2048
    train_params["batch_size"] = 1
    train_params["calib_dataset"] = "redpajama"
    train_params["train_size"] = 1
    train_params["val_size"] = 1
    train_params["seed"] = 42
    train_params["eval_tasks"] = "mmlu"
    train_params["num_fewshot"] = 5
    args.eval_batch_size = 8
    args.training_seqlen = 2048
    return args


def load_model_and_tokenizer(quant_path):
    quant_path = str(quant_path)
    if os.path.isfile(os.path.join(quant_path, "quantize_config.json")):
        qcfg = _load_quantize_config(quant_path)
        converted = qcfg.get("converted_modules") if qcfg else None
        if converted:
            return load_quantized_model(quant_path)
        hf_ckpt = _find_latest_hf_ckpt(quant_path)
        if hf_ckpt:
            print(
                "Warning: quantize_config.json has no converted modules; "
                f"falling back to QAT hf_ckpt at {hf_ckpt}"
            )
            return load_qat_hf_ckpt_model(hf_ckpt, qcfg)
        print(
            "Warning: quantize_config.json has no converted modules and no hf_ckpt found; "
            "falling back to fp16 HF load."
        )
    if "hf_ckpt" in Path(quant_path).parts:
        qcfg = _load_quantize_config(quant_path)
        return load_qat_hf_ckpt_model(quant_path, qcfg)
    if "qtip" in quant_path:
        model = qtip_model_from_hf_path(quant_path)
        tokenizer = AutoTokenizer.from_pretrained(quant_path)
        return model, tokenizer
    if "gptq" in quant_path:
        return gptq_model_from_path(quant_path)
    print("Loading model from", quant_path)
    model = AutoModelForCausalLM.from_pretrained(
        quant_path,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(quant_path)
    return model, tokenizer


def get_quant_paths():
    env_paths = os.environ.get("EVAL_QUANT_PATHS", "")
    if not env_paths:
        raise ValueError("EVAL_QUANT_PATHS is not set. Provide model paths in script/eval.sh.")
    paths = [p.strip() for p in env_paths.split(",") if p.strip()]
    if not paths:
        raise ValueError("EVAL_QUANT_PATHS is empty after parsing.")
    return paths


def main():
    for quant_path in get_quant_paths():
        run_name = Path(quant_path).name
        model_name = None
        
        # If the path ends in 'out', take the parent directory name for better context
        if run_name == "out":
            run_name = Path(quant_path).parent.parent.name 
        elif run_name == "checkpoints":
             run_name = Path(quant_path).parent.name
        
        # Try to extract a more descriptive name if possible, e.g. "w2g128-gradual-kd"
        # Assuming path structure like .../ModelName/EfficientQAT/w2g128-gradual-kd/checkpoints/out
        try:
             parts = Path(quant_path).parts
             if "checkpoints" in parts:
                 idx = parts.index("checkpoints")
                 if idx > 0:
                     run_name = parts[idx-1]
                 # Extract model name (e.g., "Llama2-7B", "Qwen2.5-3B")
                 if idx >= 3 and parts[idx-2] == "EfficientQAT":
                     model_name = parts[idx-3]
        except (ValueError, IndexError):
            pass
        
        # Build wandb run name with model name if available
        if model_name:
            wandb_name = f"eval-{model_name}-{run_name}"
        else:
            wandb_name = f"eval-{run_name}"

        wandb.init(
            project="EfficientQAT-Eval",
            name=wandb_name,
            config={"quant_path": quant_path},
            reinit=True
        )
        
        quant_model, tokenizer = load_model_and_tokenizer(quant_path)
        args = build_eval_args()
        result = evaluate(quant_model, tokenizer, args)
        print(f"quant_path: {quant_path}, result: {result}")
        
        # Flatten and log results to wandb
        log_data = {}
        if 'ppl_results' in result:
            for ds, ppl in result['ppl_results'].items():
                log_data[f"ppl/{ds}"] = ppl
        
        if 'eval_summary' in result:
            summary = result['eval_summary']
            if 'avg_acc' in summary:
                log_data["acc/avg"] = summary['avg_acc']
            if 'task_accuracies' in summary:
                for task, acc in summary['task_accuracies'].items():
                    log_data[f"acc/{task}"] = acc
        
        wandb.log(log_data)
        wandb.finish()

        torch.cuda.synchronize()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
