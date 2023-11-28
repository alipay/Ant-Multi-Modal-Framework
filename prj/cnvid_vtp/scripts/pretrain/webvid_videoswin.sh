export PYTHONPATH=`pwd`:$PYTHONPATH
export PYTORCH_TRANSFORMERS_CACHE='/YourPath/'
export TORCH_HOME='/YourPath/'

CONFIG=prj/base_vtp/configs/univl/video/pretrain/webvid_videoswin.yml
RESUME_DIR=/YourCheckpointPath/
SAVE_DIR=/YourPath/

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m antmmf.utils.launch \
    --nproc_per_node=8 --master_port=12372  prj/base_vtp/run.py \
    --config ${CONFIG} \
    training_parameters.distributed True \
    training_parameters.run_type train \
    training_parameters.find_unused_parameters True \
    training_parameters.num_workers 10 \
    training_parameters.restart True  \
    training_parameters.batch_size 128 \
    training_parameters.test_batch_size 64 \
    optimizer_attributes.params.lr 5e-5 \
    optimizer_attributes.params.weight_decay 1e-3 \
    training_parameters.enable_amp True \
    training_parameters.save_dir ${SAVE_DIR}/test \
    training_parameters.resume_file ${RESUME_DIR}/YourCheckpointPath/xxx.ckpt \
    model_attributes.univl.hard_example_mining True