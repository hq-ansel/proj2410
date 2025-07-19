from typing import List, Tuple, Dict, Any
import os
import yaml
from itertools import product
import re


def list_yaml_files(folder: str) -> List[str]:
    """
    列出指定文件夹下所有的yaml文件。
    """
    return [f for f in os.listdir(folder) if f.endswith('.yaml')]


def get_template_name(template_path: str) -> str:
    """
    从模板文件路径中提取模板名称（不含扩展名）。
    """
    return os.path.splitext(os.path.basename(template_path))[0]


def parse_setting(setting_value: str, base_dir: str) -> Tuple[bool, List[str]]:
    """
    判断配置项是文件夹还是具体yaml文件。
    """
    if os.path.isdir(os.path.join(base_dir, setting_value)):
        folder = os.path.join(base_dir, setting_value)
        files = list_yaml_files(folder)
        return True, [os.path.join(setting_value, f) for f in files]
    elif isinstance(setting_value, str) and setting_value.endswith('.yaml'):
        return False, [setting_value]
    else:
        return False, [setting_value]


def extract_settings(template_dict: Dict[str, Any]) -> Dict[str, str]:
    """
    从模板dict对象中提取各项设置。
    """
    settings = {}
    valid_keys = ['cluster_settings', 'model_settings', 'hyperparam_settings', 'train_param_settings']
    for key in valid_keys:
        if key in template_dict:
            settings[key] = template_dict[key]
    return settings


def generate_output_name(combo: Tuple[str, str, str, str], template_name: str) -> str:
    c, m, h, t = combo
    return (f"{os.path.splitext(os.path.basename(c))[0]}-"
            f"{os.path.splitext(os.path.basename(m))[0]}-"
            f"{os.path.splitext(os.path.basename(h))[0]}-"
            f"{os.path.splitext(os.path.basename(t))[0]}-"
            f"{template_name}.yaml")


def load_yaml_with_vars(path: str) -> Dict[str, Any]:
    """
    加载yaml文件内容为dict。
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归合并两个dict。
    """
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            d[k] = deep_update(d[k], v)
        else:
            d[k] = v
    return d


def get_by_path(context: Dict[str, Any], path: str) -> Any:
    """
    支持跨嵌套dict的变量查找，如 model_settings.save_parent_dir
    """
    parts = path.split('.')
    val = context
    for p in parts:
        if isinstance(val, dict) and p in val:
            val = val[p]
        else:
            return None
    return val


def resolve_vars(obj: Any, context: Dict[str, Any]) -> Any:
    """
    递归替换obj中的{var}变量，context为变量上下文。
    """
    if isinstance(obj, dict):
        return {k: resolve_vars(v, context) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [resolve_vars(i, context) for i in obj]
    elif isinstance(obj, str):
        # 递归替换直到没有变量为止
        prev = None
        result = obj
        pattern = re.compile(r'\{([\w\.]+)\}')
        while prev != result:
            prev = result
            def repl(match):
                expr = match.group(1)
                val = get_by_path(context, expr)
                return str(val) if val is not None else match.group(0)
            result = pattern.sub(repl, result)
        return result
    else:
        return obj


def build_config(template_path: str, combo_paths: Dict[str, str], base_dir: str) -> Dict[str, Any]:
    """
    读取模板，展开所有引用内容，递归替换变量，返回完整配置dict。
    """
    with open(template_path, 'r') as f:
        template_yaml = yaml.safe_load(f)
    # 先展开各settings
    context = {}
    for k, v in combo_paths.items():
        full_path = os.path.join(base_dir, v)
        context[k] = load_yaml_with_vars(full_path)
    # 合并到模板
    config = template_yaml.copy() if template_yaml else {}
    for k in context:
        config[k] = context[k]
    # 先把一级变量（如 method）展开
    config = resolve_vars(config, config)
    # 再递归替换所有变量
    config = resolve_vars(config, config)
    return config


def main():
    """
    主函数：读取模板，解析设置，生成所有组合的完整嵌套配置文件。
    """
    base_dir = 'yaml'
    # 除了baseline.yaml，还有 gradual_quant.yaml
    template_path = os.path.join(base_dir, 'template', 'gradual_quant.yaml')
    output_dir = os.path.join(base_dir, 'generate')
    os.makedirs(output_dir, exist_ok=True)
    with open(template_path, 'r') as f:
        template_yaml = yaml.safe_load(f)
    settings = extract_settings(template_yaml)
    candidates = {}
    for k, v in settings.items():
        _, items = parse_setting(v, base_dir)
        candidates[k] = items
    keys = ['cluster_settings', 'model_settings', 'hyperparam_settings', 'train_param_settings']
    all_combos = list(product(*(candidates[k] for k in keys)))
    template_name = get_template_name(template_path)
    for combo in all_combos:
        combo_paths = dict(zip(keys, combo))
        config = build_config(template_path, combo_paths, base_dir)
        out_name = generate_output_name(combo, template_name)
        out_path = os.path.join(output_dir, out_name)
        with open(out_path, 'w') as f:
            yaml.dump(config, f, allow_unicode=True, sort_keys=False)
        print(f"生成: {out_path}")


if __name__ == '__main__':
    main()