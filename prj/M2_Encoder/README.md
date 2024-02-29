# $M^2$-Encoder: Advancing Bilingual Image-Text Understanding by Large-scale Efficient Pretraining
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boldsymbol-m-2-encoder-advancing-bilingual/zero-shot-image-retrieval-on-coco-cn)](https://paperswithcode.com/sota/zero-shot-image-retrieval-on-coco-cn?p=boldsymbol-m-2-encoder-advancing-bilingual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boldsymbol-m-2-encoder-advancing-bilingual/zero-shot-cross-modal-retrieval-on-flickr30k)](https://paperswithcode.com/sota/zero-shot-cross-modal-retrieval-on-flickr30k?p=boldsymbol-m-2-encoder-advancing-bilingual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boldsymbol-m-2-encoder-advancing-bilingual/zero-shot-image-retrieval-on-flickr30k-cn)](https://paperswithcode.com/sota/zero-shot-image-retrieval-on-flickr30k-cn?p=boldsymbol-m-2-encoder-advancing-bilingual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boldsymbol-m-2-encoder-advancing-bilingual/zero-shot-transfer-image-classification-on-1)](https://paperswithcode.com/sota/zero-shot-transfer-image-classification-on-1?p=boldsymbol-m-2-encoder-advancing-bilingual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boldsymbol-m-2-encoder-advancing-bilingual/zero-shot-learning-on-imagenet-cn)](https://paperswithcode.com/sota/zero-shot-learning-on-imagenet-cn?p=boldsymbol-m-2-encoder-advancing-bilingual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boldsymbol-m-2-encoder-advancing-bilingual/zero-shot-text-to-image-retrieval-on-coco-cn)](https://paperswithcode.com/sota/zero-shot-text-to-image-retrieval-on-coco-cn?p=boldsymbol-m-2-encoder-advancing-bilingual)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/boldsymbol-m-2-encoder-advancing-bilingual/zero-shot-cross-modal-retrieval-on-coco-2014)](https://paperswithcode.com/sota/zero-shot-cross-modal-retrieval-on-coco-2014?p=boldsymbol-m-2-encoder-advancing-bilingual)

Official PyTorch implementation of the paper ["M2-Encoder: Advancing Bilingual Image-Text Understanding by Large-scale Efficient Pretraining"](https://arxiv.org/abs/2401.15896).


## What is $M^2$-Encoder?
Vision-language foundation models like CLIP have revolutionized the field of artificial intelligence, yet Chinese multimodal foundation models have lagged due to the relative scarcity of large-scale pretraining datasets. Toward this end, we introduce a comprehensive bilingual(Chinese-English) dataset BM-6B with over 6 billion image-text pairs, aimed at enhancing multimodal foundation models, especially for Chinese. To efficiently handle this dataset’s size, we also propose a novel grouped aggregation approach for image-text contrastive loss computation, which reduces the communication overhead and GPU memory demands, facilitating a 60% increase in training speed. We pretrain a series of bilingual image-text foundation models with an enhanced fine-grained understanding ability on BM-6B, the resulting models, dubbed as $M^2$-Encoders(pronounced “M-Square”), set new benchmarks in both languages for multimodal retrieval and classification tasks. Notably, Our largest $M^2$-Encoder-10B model achieves top-1 accuracies of 88.5% on ImageNet and 80.7% on ImageNet-CN under a zero-shot classification setting, surpassing previously reported SoTA methods by 2.2% and 21.1%, respectively. We believe our $M^2$-Encoder series represents one of the most comprehensive bilingual image-text foundation models to date, and we are making it available to the research community for further exploration and development.

## What can $M^2$-Encoder do ?
image-text cross-modal retrieval and zero-shot image classification

## Schedule
- [x] Release $M^2$-Encoder code
- [x] Release Weights on Modelscope
- [ ] Release Demos on Modelscope

## Weights on Modelscope 
https://www.modelscope.cn/models/M2Cognition/M2-Encoder/summary

## Quick Start 
```
# 新建环境（Python版本3.8）
conda create -n m2-encoder python=3.8
source activate m2-encoder

# clone项目地址
cd /YourPath/
git clone https://github.com/alipay/Ant-Multi-Modal-Framework

# 安装包依赖
cd ./Ant-Multi-Modal-Framework/prj/M2_Encoder/
pip install -r requirements.txt

# 运行demo, 会自动下载modelscope上的权重
python run.py
```



## Evaluation Results
An overview of existing multimodal models on zero-shot classification and retrieval performance. The top-1 accuracy on (a) ImageNet-CN and (b) ImageNet. The retrieval MR on (c) Flicker30K-CN and (d) Flicker30K. Our $M^2$-Encoders excel compared to models with a similar number of parameters.
![](https://github.com/alipay/Ant-Multi-Modal-Framework/blob/main/prj/M2_Encoder/pics/effect.png)


## Citations
If you find $M^2$-Encoder useful for your work, please consider citing:
```
@misc{guo2024m2encoder,
      title={M2-Encoder: Advancing Bilingual Image-Text Understanding by Large-scale Efficient Pretraining}, 
      author={Qingpei Guo and Furong Xu and Hanxiao Zhang and Wang Ren and Ziping Ma and Lin Ju and Jian Wang and Jingdong Chen and Ming Yang},
      year={2024},
      url={https://arxiv.org/abs/2401.15896},
}
```

## Acknowledgement
$M^2$-Encoder is built with reference to the code of the following projects: [BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3), [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP). Thanks for their awesome work!


