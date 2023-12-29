# SNP-S3-VTP: Shared Network Pre-training and Significant Semantic Strengthening for Various Video-Text Tasks

This repository contains the codebase for our paper
[SNP-S3: Shared Network Pre-training and Significant Semantic Strengthening for Various Video-Text Tasks](https://ieeexplore.ieee.org/document/10214396), which has been accepted by TCSVT.

## What is SNP-S3?

SNP-S3 is a framework for learning cross-modal video representations by directly pre-training on raw data to facilitate various downstream video-text tasks. 

The main contributions of SNP-S3 lie in the pre-training framework and proxy tasks.

* SNP-S3 proposes Shared Network Pre-training (SNP). By employing one shared BERT-type network to refine textual and cross-modal features simultaneously, SNP is lightweight and could support various downstream applications.
* SNP-S3 proposes the Significant Semantic Strengthening (S3) strategy, which includes a novel masking and matching proxy task to promote the pre-training performance.
* Experiments conducted on three downstream video-text tasks and six datasets demonstrate that, SNP-S3 achieves a satisfactory balance between the pre-training efficiency and the fine-tuning performance.

## Codebase

Check [CODEBASE_cn.md (中文)](https://github.com/alipay/Ant-Multi-Modal-Framework/blob/main/prj/snps3_vtp/CODEBASE_cn.md) or 
[CODEBASE_en.md (English)](https://github.com/alipay/Ant-Multi-Modal-Framework/blob/main/prj/snps3_vtp/CODEBASE_en.md) for instructions of codebase downloading and model pre-training (SNP-S3).


## Citation

If you find SNP-S3 useful, please consider citing the following paper:

```
@ARTICLE{10214396,
  author={Dong, Xingning and Guo, Qingpei and Gan, Tian and Wang, Qing and Wu, Jianlong and Ren, Xiangyuan and Cheng, Yuan and Chu, Wei},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={SNP-S3: Shared Network Pre-training and Significant Semantic Strengthening for Various Video-Text Tasks}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2023.3303945}
}
```
