# Ant Multi-Modal-Framework (AntMMF)
蚂蚁自研多模态视频预训练框架。

<p align="center">
  
<p align="center">
    👋 团队目前深耕自研多模态大模型，已有相关成熟的经验和产品。欢迎感兴趣，有能力的小伙伴加入我们！
</p>

</p>

*Read this in [English](README_en.md).*

## News （最近更新2023/12/29）

本项目作为底层代码库，是如下项目的底层依赖，包括：

- SNP-S3: 多模态预训练模型【TCSVT，CCF-B】
- CNVid-3.5M: 中文多模态预训练模型 & 中文视频文本数据集 【CVPR-23，CCF-A】
- DMAE: 双模态注意力增强的文本视频检索 & 三元偏序对比学习 【ACM MM-23，CCF-A】

## Introduction （介绍）

该论文的代码库简称为AntMMF，用于多模态视频预训练。

AntMMF的第一级文件目录如下所示：
- antmmf						# 核心代码库
- prj								# 工程项目库 （主要代码）
- tests							# 本地测试脚本及数据
- LEGAL.md 					# 合法性声明
- README.md					# 使用指南
- README-CN.md			# 使用指南（中文）
- requirements.txt	# 依赖包


## Lincense （使用协议）

协议为CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

使用本项目前，请先阅读LICENSE.txt。如果您不同意该使用协议中列出的条款、法律免责声明和许可，您将不得使用本项目中的这些内容。

## Installation （安装指南）

- AntMMF的安装步骤如下所示：
```
# 新建环境（Python版本3.8）
conda create -n antmmf python=3.8
source activate antmmf

# clone项目地址
cd /YourPath/
git clone https://github.com/alipay/Ant-Multi-Modal-Framework


# 安装包依赖
cd antmmf
pip install -r requirements.txt
```

- AntMMF支持通过docker启动，具体详见`\docker`文档。

`TODO`：docker文档和相关环境整理中，后续会对外进行发布。

## Quick Start （快速启动）

AntMMF提供了本地测试脚本，可以快速进行安装正确性验证：
```
# 终端运行
sh prj/base_vtp/scripts/local_test/coco_vg.local.sh
```

## FAQ （问答）

## Citations （引用）

