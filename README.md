# 蚂蚁多模态框架
Read this in [English](https://github.com/alipay/Ant-Multi-Modal-Framework/blob/main/README_EN.md).

# 简介
本代码库包含蚂蚁多模态认知团队在AntMMF中集成的多模态方向研究代码。AntMMF多模态框架封装了包括数据集管理、数据处理、训练流程、模型和模块在内的标准多模态功能，同时支持这些组件的自定义扩展。


## News
- 2024.02: 开源中英双语多模态CLIP:[M2-Encoder](https://github.com/alipay/Ant-Multi-Modal-Framework/tree/main/prj/M2_Encoder), 使用大规模中英文数据进行训练（~60亿图文对）
- 2023.12: 开源以下论文代码 [SNP-S3](https://github.com/alipay/Ant-Multi-Modal-Framework/tree/main/prj/snps3_vtp), [DMAE](https://github.com/alipay/Ant-Multi-Modal-Framework/tree/main/prj/dmae_vtp), and [CNVid-3.5M](https://github.com/alipay/Ant-Multi-Modal-Framework/tree/main/prj/cnvid_vtp).
- 2023.06: [SNP-S3](https://ieeexplore.ieee.org/document/10214396) 被IEEE T-CSVT(Transactions on Circuits and Systems for Video Technology) 2023接收.
- 2023.05: [DMAE](https://arxiv.org/pdf/2309.11082.pdf) 被ACM MultiMedia 2023接收.
- 2023.03: [CNVid-3.5M](https://openaccess.thecvf.com/content/CVPR2023/papers/Gan_CNVid-3.5M_Build_Filter_and_Pre-Train_the_Large-Scale_Public_Chinese_Video-Text_CVPR_2023_paper.pdf) 被CVPR 2023接收.
 
## 研究方向

### 视频-文本预训练
- 数据集
  - [CNVid-3.5M](https://openaccess.thecvf.com/content/CVPR2023/papers/Gan_CNVid-3.5M_Build_Filter_and_Pre-Train_the_Large-Scale_Public_Chinese_Video-Text_CVPR_2023_paper.pdf) (CVPR-2023): 中文视频文本预训练数据集。
- 预训练方法及模型
  - [SNP-S3](https://ieeexplore.ieee.org/document/10214396) (IEEE T-CSVT 2023): 语义增强的视频预训练。

### 视频-文本检索 
- [DMAE](https://arxiv.org/pdf/2309.11082.pdf) (ACM MM-2023): 双模态注意力增强和偏序对比学习的视频文本检索。

### 视频编辑
- [EVE](https://arxiv.org/abs/2308.10648): 高效的零样本视频编辑方法。


## 环境设置

- 请按照以下步骤初始化AntMMF运行环境。
```
# 创建新环境
conda create -n antmmf python=3.8
source activate antmmf

# 克隆项目代码到本地
git clone https://github.com/alipay/Ant-Multi-Modal-Framework

# 安装项目依赖
cd antmmf
pip install -r requirements.txt
```

## Citations
如果您觉得AntMMF对您的工作有帮助，请考虑引用：
```
@misc{qp2023AntMMF,
  author =       {Qingpei, Guo and Xingning, Dong and Xiaopei, Wan and Xuzheng, Yu and Chen, Jiang and Xiangyuan, Ren and Kiasheng, Yao and Shiyu, Xuan},
  title =        {AntMMF: Ant Multi-Modal Framework},
  howpublished = {\url{https://github.com/alipay/Ant-Multi-Modal-Framework}},
  year =         {2023}
}
```

## License

本项目根据[Apache 2.0](https://github.com/apache/.github/blob/main/LICENSE) 授权，在正确引用出处的情况下，允许在任何媒介中无限制地使用、分发和复制。

## 致谢
我们的代码基于[FAIR mmf](https://github.com/facebookresearch/mmf)，感谢作者的重要开源贡献。

## 联系我们

:raising_hand: 如需帮助或解决与本代码库相关的问题，请提交issue。

:star: 我们正在招聘，如果您对我们的工作感兴趣，请通过`qingpei.gqp@antgroup.com`联系我们。

