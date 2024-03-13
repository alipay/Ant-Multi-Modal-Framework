# AntMMF: Ant Multi-Modal Framework

## Introduction
 
This repository contains codes for multi-modality learning from the Multimodal Cognition group of Ant Group that have been integrated into AntMMF. AntMMF encapsulates standard multimodal functionalities including dataset management, data processing, training workflows, models, and modules, while also enabling custom extensions of these components.


## News
- February, 2024: release the code of bilingual multimodal CLIP-[M2-Encoder](https://github.com/alipay/Ant-Multi-Modal-Framework/tree/main/prj/M2_Encoder), which was trained on our BM-6B bilingual dataset.
- December, 2023: release the code of [SNP-S3](https://github.com/alipay/Ant-Multi-Modal-Framework/tree/main/prj/snps3_vtp), [DMAE](https://github.com/alipay/Ant-Multi-Modal-Framework/tree/main/prj/dmae_vtp), and [CNVid-3.5M](https://github.com/alipay/Ant-Multi-Modal-Framework/tree/main/prj/cnvid_vtp).
- June, 2023: [SNP-S3](https://ieeexplore.ieee.org/document/10214396) was accepted by IEEE T-CSVT 2023.
- May, 2023: [DMAE](https://arxiv.org/pdf/2309.11082.pdf) was accepted by ACM MultiMedia 2023.
- March, 2023: [CNVid-3.5M](https://openaccess.thecvf.com/content/CVPR2023/papers/Gan_CNVid-3.5M_Build_Filter_and_Pre-Train_the_Large-Scale_Public_Chinese_Video-Text_CVPR_2023_paper.pdf) was accepted by CVPR 2023.
 

## Focus Areas

### Video & Text Pretraining
- Dataset
  - [CNVid-3.5M](https://openaccess.thecvf.com/content/CVPR2023/papers/Gan_CNVid-3.5M_Build_Filter_and_Pre-Train_the_Large-Scale_Public_Chinese_Video-Text_CVPR_2023_paper.pdf) (CVPR-2023): A large-scale public Chinese video-text pretraining dataset.
- Pretraining Methods
  - [SNP-S3](https://ieeexplore.ieee.org/document/10214396) (IEEE T-CSVT 2023): Semantic enhancement for video pretraining.

### Video & Text Retrieval 
- [DMAE](https://arxiv.org/pdf/2309.11082.pdf) (ACM MM-2023): Dual-Modal attention-enhanced Text-Video Retrieval with triplet partial margin contrastive learning.

### Video Editing
- [EVE](https://arxiv.org/abs/2308.10648): Efficient zero-shot video editing.


## Environmental Setup

- Please follow the steps below to initialize the environment of the AntMMF.
```
# Build a new environment.
conda create -n antmmf python=3.8
source activate antmmf

# Clone this project.
cd /YourPath/
git clone https://github.com/alipay/Ant-Multi-Modal-Framework

# Install the required packages.
cd antmmf
pip install -r requirements.txt
```

## Citations
If you find AntMMF useful for your work, please consider citing:
```
@misc{qp2023AntMMF,
  author = {Qingpei, Guo and Xingning, Dong and Xiaopei, Wan and Xuzheng, Yu and Chen, Jiang and Xiangyuan, Ren and Kiasheng, Yao and Shiyu, Xuan},
  title =        {AntMMF: Ant Multi-Modal Framework},
  howpublished = {\url{https://github.com/alipay/Ant-Multi-Modal-Framework}},
  year =         {2023}
}
```

## License
This project is licensed under the [Apache 2.0](https://github.com/apache/.github/blob/main/LICENSE) license, which permits unrestricted use, distribution, and reproduction in any medium, provided the original work is properly cited.

## Acknowledgments
Our code is based on [FAIR mmf](https://github.com/facebookresearch/mmf). We thank the authors for their wonderful open-source efforts.


## Contact Information
:raising_hand: For help or issues with this codebase, please submit an issue.

:star: We are hiring, if you are interested in our work, please feel free to contact  Qingpei Guo(qingpei.gqp@antgroup.com).
