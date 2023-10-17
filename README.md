# Ant Multi-Modal-Framework (AntMMF)
蚂蚁自研多模态视频预训练框架。

## Introduction （介绍）

该论文的代码库简称为AntMMF，用于多模态视频预训练。

AntMMF的第一级文件目录如下所示：
- LEGAL.md 					# 合法性声明
- README.md					# 使用指南
- antmmf						# 核心代码库
- docker						# docker环境说明（后续补充）
- prj								# 工程项目库 （主要代码）
- requirements.txt	# 依赖包
-tests							# 本地测试脚本及数据

## Lincense （使用协议）

协议为CC BY 4.0(https://creativecommons.org/licenses/by/4.0/)

Please first LICENSE.txt. You must not use the content in this project if you do not agree to the terms, legal disclaimer, and license outlined in these files.

## Installation （安装指南）

- AntMMF的安装步骤如下所示：
```
# 新建环境（Python版本3.8）
conda create -n antmmf python=3.8
source activate antmmf

# clone项目地址
cd /YourPath/
git clone -b dxn_0709_open https://code.alipay.com/multimodal/antmmf.git

# 安装包依赖
cd antmmf
pip install -r requirements.txt
```

- AntMMF支持通过docker启动，具体详见`\docker`文档。

`TODO`：docker文档待后续dockerfile对外披露后，再补充。

## Dataset (数据集)

AntMMF支持在以下的公开数据集上进行预训练或微调操作：
- 预训练数据集：
  - 图文数据集，支持`COCO`，`VG`，`CC3M`等数据集；
  - 视频文本数据集，支持`WebVid-2M`，`Howto100M`，`CNVid-3.5M`（中文）等数据集。
- 微调数据集：
  - 跨模态检索数据集，支持`MSRVTT`，`DiDemo`，`MSVD`，`VATEX`等数据集；
  - 视频问答数据集，支持`MSRVTT-QA`，`MSVD-QA`等数据集；
  - 多选项视频问答数据集，支持`MSRVTT-MC-QA`等数据集。

## Performance Results （结果指标）

AntMMF在多个公开视频理解数据集上的结果如下所示：

`TODO`：结果指标待后续模型对外披露后，再补充。

## Quick Start （快速启动）

AntMMF提供了本地测试脚本，可以快速进行安装正确性验证：
```
# 终端运行
sh tests/scripts/local_test/coco_vg.local.sh
```

## Pre-Training （预训练）

AntMMF提供了多个数据集上的预训练脚本，具体详见`tests/scripts/pretrain`。

预训练时，可以通过 1）命令行，2）sh脚本，3）yml文件 这三种方式控制预训练进程，其优先级顺序为：命令行 > sh脚本 > yml文件。

下面以在COCO+VG图文数据集上预训练为例，展示AntMMF的预训练流程：

- 下载COCO+VG数据集
- 修改prj/video_text_pretrain/configs/univl/video/pretrain/coco_vg_videoswin.yml中的`data_root_dir`字段
- 运行tests/scripts/pretrain/coco_vg_videoswin.sh脚本，其中一些重要字段的含义是：

```
python -m antmmf.utils.launch \
    --nproc_per_node=8 --master_port=12371  prj/video_text_pretrain/run.py \
    --config ${CONFIG} \                                # config文件路径
    training_parameters.distributed True \              # 是否进行分布式数据读取和训练
    training_parameters.run_type train \                # 当前运行状态（train->训练，predict->测试）    
    training_parameters.restart True \                  # 是否重新开始训练（False的话会重置训练轮数）
    training_parameters.batch_size 128 \                # 训练size
    training_parameters.test_batch_size 64 \            # 测试size
    optimizer_attributes.params.lr 5e-5 \               # 学习率
    optimizer_attributes.params.weight_decay 1e-3 \     # 学习率衰减率
    training_parameters.enable_amp True \               # 是否开启混合精度训练
    training_parameters.save_dir ${SAVE_DIR}/test       # 训练结果保存地址
```

## Fine-Tuning （微调）

AntMMF提供了多个数据集上的微调脚本，具体详见`tests/scripts/finetune`。

微调的流程和逻辑与预训练类似，但需要注意为`training_parameters.resume_file`赋值，该字段会读取对应checkpoint的文件参数。

## Inference （推理）

AntMMF支持使用已训练/微调好的模型进行测试，测试的脚本可类比`tests/scripts/finetune/mcvqa_msr_vtt_mc_qa_videoswin.sh`文件。

注意在测试时，须将`training_parameters.run_type`字段置为`predict`，
并且`training_parameters.resume_file`须指向一个已充分收敛的模型。

## FAQ （问答）

## Citations （引用）

