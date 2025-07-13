import glob
import os
import subprocess
from typing import List, Dict
import yaml

WORKDIR: str = '/home/ubuntu/data/exp/proj2410/'
PYTHONPATH: str = '/home/ubuntu/data/exp/proj2410/EfficientQAT'
HF_HOME: str = '/home/ubuntu/data/exp/proj2410/hf_home'
LOGDIR: str = 'logs'
os.makedirs(LOGDIR, exist_ok=True)

def load_yaml(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_model_path(config: Dict) -> str:
    return config.get('model_settings', {}).get('model', '/home/ubuntu/data/exp/proj2410/model/unknown')

def get_save_path(config: Dict) -> str:
    return config.get('save_dir', '/home/ubuntu/data/exp/proj2410/quant_model/EfficientQAT/unknown')

def main() -> None:
    config_files: List[str] = glob.glob('yaml/generate/*.yaml')
    for config_path in config_files:
        config: Dict = load_yaml(config_path)
        model_path: str = get_model_path(config)
        save_path: str = get_save_path(config)
        env: Dict[str, str] = os.environ.copy()
        env['HF_HOME'] = HF_HOME
        env['PYTHONPATH'] = env.get('PYTHONPATH', '') + ':' + PYTHONPATH
        env['MODEL_PATH'] = model_path
        env['SAVE_PATH'] = save_path
        cmd: List[str] = [
            'python', '-m', 'EfficientQAT.main_block_ap',
            '--config_path', config_path,
            '--wbits', str(config.get('hyperparam_settings', {}).get('wbits', 4)),
            '--group_size', str(config.get('hyperparam_settings', {}).get('group_size', 128)),
            '--quant_lr', str(config.get('train_param_settings', {}).get('quant_lr', 1e-5)),
            '--weight_lr', str(config.get('train_param_settings', {}).get('weight_lr', 1e-5)),
            '--batch_size', str(config.get('train_param_settings', {}).get('batch_size', 8)),
            '--eval_ppl', '--real_quant', '--epochs', str(config.get('train_param_settings', {}).get('epochs', 2)),
            '--save_quant_dir', save_path
        ]
        log_file: str = os.path.join(LOGDIR, os.path.basename(config_path) + '.log')
        print(f'Running: {" ".join(cmd)}')
        with open(log_file, 'w') as logf:
            subprocess.run(cmd, env=env, cwd=WORKDIR, stdout=logf, stderr=subprocess.STDOUT)
        print(f'Finished: {config_path}')

if __name__ == '__main__':
    main() 