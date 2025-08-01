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

import glob
import os
import subprocess
from multiprocessing import Pool, cpu_count
from typing import List, Dict

env: Dict[str, str] = os.environ.copy()
env['HF_HOME'] = HF_HOME
env['PYTHONPATH'] = env.get('PYTHONPATH', '') + ':' + PYTHONPATH
def run_command(cmd):
    """运行单个命令，并捕获输出和错误"""
    log_file = os.path.join(LOGDIR, os.path.basename(cmd[4].replace("EfficientQAT/","")) + '.log') # 修正日志文件路径
    print(f'Running: {" ".join(cmd)}')
    with open(log_file, 'w') as logf:
        result = subprocess.run(cmd, env=env, cwd=WORKDIR, stdout=logf, stderr=subprocess.STDOUT)
    print(f'Finished: {cmd[4].replace("EfficientQAT/","")}')
    return result.returncode


def main() -> None:
    batchsize = 1
    config_files: List[str] = glob.glob('yaml/generate/*.yaml')
    config_files.sort()
    
    commands = []
    for config_path in config_files:
        cmd: List[str] = [
            'python', '-m', 'EfficientQAT.main_block_ap',
            '--config_path', "EfficientQAT/"+config_path,
        ]
        commands.append(cmd)

    with Pool(processes=min(batchsize, len(commands), cpu_count())) as pool: # 使用进程池，限制进程数量
        results = pool.map(run_command, commands)

    # 检查是否有任何子进程失败
    if any(result != 0 for result in results):
        print("Some subprocesses failed.")


#  你需要在调用 main 函数时，传入 batchsize 参数，例如：
# main(batchsize=1)  # 一次跑一个subprocess
# main(batchsize=2)  # 一次跑两个subprocess
# main(batchsize=4) # 一次跑四个subprocess (如果你的机器支持)
if __name__ == '__main__':
    main() 