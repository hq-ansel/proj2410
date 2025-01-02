# proj2410


EfficientQAT 是主要代码实现的地方

通过执行 bash EfficientQAT/script/qwen/baseline.sh 即可执行Efficient QAT 进行2bit group size为128粒度的量化


deita_dataset是deita-6k 使用监督模型时需要的包，属于原EfficientQAT项目的一部分，基本不用

examples 是EfficientQAT项目的案例代码，可以作为启动参考

model_transfer 主要包含将EfficientQAT特定的格式的模型转换为gptqmodel可以执行的模型，或者转换为bitblas 可以执行的模型格式

quantize包括逐层量化的主要部分(block_ap,greedy_trainer)，以及量化器(quantizer,quantizerv2 代表使用高精度零点，quantizerv3 代表LSQ格式)

quantize/int_linear_fake 是进行模拟量化(全fp32)的线性类

quantize/int_linear_real 是进行真实量化(权重是对应量化数值类型)的线性类

quantize/trition_utils 是用triton实现的一个简单的反量化核

quantize/utils 包括quantize使用时的一些工具函数

script包含了执行2bit group size为128粒度的量化的脚本
yaml则是配置文件

datautils_block是每层更新数据集处理的相关类与load text数据集的函数

datautils_e2e 则是EfficientQAT的的qp部分所需的数据类

loss_utils 仅包含了loss的相关函数

utils 包含了一些amp相关类，记录中间属性的函数

main_block_ap.py 是block_ap量化的主函数入口

main_e2e_qp.py 是EfficientQAT的qp部分的主函数入口


必要条件执行

安装依赖包
pip install -r EfficientQAT/requirements.txt

下载必要模型