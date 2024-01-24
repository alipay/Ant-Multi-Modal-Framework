# M2-Encoder: Advancing Bilingual Image-Text Understanding by Large-scale Efficient Pretraining

Codebase for M2-Encoder, models will be released soon.

## Abstract
Vision-language foundation models like CLIP
have revolutionized the field of artificial intelligence,
yet Chinese multimodal foundation
models have lagged due to the relative
scarcity of large-scale pretraining datasets. Toward
this end, we introduce a comprehensive
bilingual(Chinese-English) dataset BM-
6B with over 6 billion image-text pairs, aimed
at enhancing multimodal foundation models,
especially for Chinese. To efficiently handle
this dataset’s size, we also propose a novel
grouped aggregation approach for image-text
contrastive loss computation, which reduces
the communication overhead and GPU memory
demands, facilitating a 60% increase in
training speed. We pretrain a series of bilingual
image-text foundation models with an enhanced
fine-grained understanding ability on
BM-6B, the resulting models, dubbed as M2-
Encoders(pronounced “M-Square”), set new
benchmarks in both languages for multimodal
retrieval and classification tasks. Notably, Our
largest M2-Encoder-10B model achieves top-1
accuracies of 88.5% on ImageNet and 80.7%
on ImageNet-CN under a zero-shot classification
setting, surpassing previously reported
SoTA methods by 2.2% and 21.1%, respectively.
We believe our M2-Encoder series represents
one of the most comprehensive bilingual
image-text foundation models to date, and we
are making it available to the research community
for further exploration and development.

## Overall Results
An overview of existing multimodal models
on zero-shot classification and retrieval performance.
The top-1 accuracy on (a) ImageNet-CN and (b) ImageNet.
The retrieval MR on (c) Flicker30K-CN and
(d) Flicker30K. Our M2-Encoders excel compared to
models with a similar number of parameters.
![](pics/effect.png)


## Benchmark Evaluation
### Zero-shot Evaluation on Chinese benchmark
- Imagenet Classification
![](pics/cn_imagenet_cls.jpg)

- Image-text Retrieval
![](pics/cn_retrieval.jpg)

### Zero-shot Evaluation on English benchmark
- Imagenet Classification
![](pics/en_imagenet_cls.jpg)

- Image-text Retrieval
![](pics/en_retrieval.jpg)

## Zero-shot Evaluation on fine-grained benchmark
![](pics/fine-grained.jpg)


