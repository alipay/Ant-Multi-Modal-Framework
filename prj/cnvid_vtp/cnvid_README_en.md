# Ant Multi-Modal-Framework (AntMMF)

This codebase is now mainly used for multi-modal image/video pre-training.

## Lincense

The lincense of the AntMMF project is CC BY 4.0  (https://creativecommons.org/licenses/by/4.0/).

Please first LICENSE.txt. You must not use the content in this project if you do not agree to the terms, legal disclaimer, and license outlined in these files.

## Installation

- Please follow the forward steps to initialize the environment of the AntMMF.
```
# Build a new environment.
conda create -n antmmf python=3.8
source activate antmmf

# Clone this project.
cd /YourPath/
git clone -b dxn_0709_open https://code.alipay.com/multimodal/antmmf.git

# Install the required packages.
cd antmmf
pip install -r requirements.txt
```

## Dataset

AntMMF supports the following public datasets for multi-modal pre-training and fine-tuning.
- Pre-training datasets：
  - Image-text datasets: *e.g.*, `COCO`，`VG`，and `CC3M`.
  - Video-text datasets: *e.g.*, `WebVid-2M`，`Howto100M`，and `CNVid-3.5M`(Chinese).
- Fine-tuning datasets：
  - Cross-modal retrieval datasets: *e.g.*, `MSRVTT`，`DiDemo`，`MSVD`，and `VATEX`；
  - Video qusetion-answering datasets: *e.g.*, `MSRVTT-QA`，and `MSVD-QA`；
  - Multi-choice video qusetion-answering datasets: *e.g.*, `MSRVTT-MC-QA`.

## Performance Results

`TODO`：Performance results are coming soon.

## Quick Start

AntMMF provides the following script for local test.
```
sh tests/scripts/local_test/coco_vg.local.sh
```

## Pre-Training

AntMMF provides various pre-training scripts, please follow `tests/scripts/pretrain` for more information.

The following shell command is an example to start the image-text pre-training on COCO-VG datasets.

```
python -m antmmf.utils.launch \
    --nproc_per_node=8 --master_port=12371  prj/base_vtp/run.py \
    --config ${CONFIG} \                                
    training_parameters.distributed True \              
    training_parameters.run_type train \                
    training_parameters.restart True \                  
    training_parameters.batch_size 128 \                
    training_parameters.test_batch_size 64 \            
    optimizer_attributes.params.lr 5e-5 \               
    optimizer_attributes.params.weight_decay 1e-3 \     
    training_parameters.enable_amp True \               
    training_parameters.save_dir ${SAVE_DIR}/test      
```

## Fine-Tuning

AntMMF provides various pre-training scripts, please follow `tests/scripts/finetune` for more information.

The pipeline of fine-tuning is the same as pre-training.

## Inference

AntMMF supports inference with a well pre-trained model, please follow `tests/scripts/finetune/mcvqa_msr_vtt_mc_qa_videoswin.sh` for more information.

## FAQ

## Citations

