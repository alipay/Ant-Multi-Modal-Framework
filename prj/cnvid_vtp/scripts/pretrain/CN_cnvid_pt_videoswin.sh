export PYTHONPATH=`pwd`:$PYTHONPATH
export PYTORCH_TRANSFORMERS_CACHE='/YourPath/'
export TORCH_HOME='/YourPath/'

CONFIG=prj/cnvid_vtp/configs/univl/video/pretrain/CN_video_videoswin.yml
RESUME_DIR=/YourCheckpointPath/
SAVE_DIR=/YourPath/

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m antmmf.utils.launch \
    --nproc_per_node=8 --master_port=12372  prj/cnvid_vtp/run.py \
    --config ${CONFIG} \
    training_parameters.distributed True \
    training_parameters.run_type train \
    training_parameters.restart True \
    training_parameters.find_unused_parameters True \
    task_attributes.univl_task.dataset_attributes.video_text_pretrain.train_ensemble_n_clips 4 \
    task_attributes.univl_task.dataset_attributes.video_text_pretrain.test_ensembel_n_clips 4 \
    training_parameters.num_workers 20 \
    training_parameters.batch_size 128 \
    training_parameters.test_batch_size 64 \
    optimizer_attributes.params.lr 5e-5 \
    optimizer_attributes.params.weight_decay 1e-3 \
    training_parameters.enable_amp False \
    training_parameters.save_dir ${SAVE_DIR}/test \
    model_attributes.univl.hard_example_mining True