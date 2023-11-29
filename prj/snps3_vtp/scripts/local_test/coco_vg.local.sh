export PYTHONPATH=`pwd`:$PYTHONPATH
export PYTORCH_TRANSFORMERS_CACHE='/YourPath/'
export TORCH_HOME='/YourPath/'

CONFIG=prj/base_vtp/configs/univl/video/pretrain/coco_vg.local.yml
TEST_FINETUNE_SAVE_DIR=/YourPath/

export CUDA_VISIBLE_DEVICES=0,1
python -m antmmf.utils.launch \
    --nproc_per_node=2 --master_port=12391 prj/base_vtp/run.py \
    --config ${CONFIG} \
    training_parameters.distributed True \
    training_parameters.run_type train \
    training_parameters.restart True \
    training_parameters.find_unused_parameters True \
    training_parameters.num_workers 10 \
    training_parameters.batch_size 2 \
    optimizer_attributes.params.lr 1e-5 \
    training_parameters.save_dir ${TEST_FINETUNE_SAVE_DIR}/test \
    training_parameters.log_interval 20