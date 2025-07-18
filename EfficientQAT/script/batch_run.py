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
    # 安装字符排序
    config_files.sort()
    for config_path in config_files:
        env: Dict[str, str] = os.environ.copy()
        env['HF_HOME'] = HF_HOME
        env['PYTHONPATH'] = env.get('PYTHONPATH', '') + ':' + PYTHONPATH
        # env["http_proxy"] = "172.18.166.139:7890"
        # env['https_proxy'] = "172.18.166.139:7890"
        # env['HF_ENDPOINT'] = "https://hf-mirror.com"
        cmd: List[str] = [
            'python', '-m', 'EfficientQAT.main_block_ap',
            '--config_path', "EfficientQAT/"+config_path,
        ]
        log_file: str = os.path.join(LOGDIR, os.path.basename(config_path) + '.log')
        print(f'Running: {" ".join(cmd)}')
        with open(log_file, 'w') as logf:
            subprocess.run(cmd, env=env, cwd=WORKDIR, stdout=logf, stderr=subprocess.STDOUT)
        print(f'Finished: {config_path}')

if __name__ == '__main__':
    main() 