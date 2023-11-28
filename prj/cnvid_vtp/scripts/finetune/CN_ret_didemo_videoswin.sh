export PYTHONPATH=`pwd`:$PYTHONPATH
export PYTORCH_TRANSFORMERS_CACHE='/YourPath/'
export TORCH_HOME='/YourPath/'

CONFIG=prj/cnvid_vtp/configs/univl/video/finetune_retrieval/CN_didemo_videoswin.yml
PRETRAINED_SAVE_DIR=/YourCheckpointPath/
DIDEMO_CN_FINETUNE_SAVE_DIR=/YourPath/

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m antmmf.utils.launch \
    --nproc_per_node=8 --master_port=12392 prj/cnvid_vtp/run.py \
    --config ${CONFIG} \
    training_parameters.distributed True \
    training_parameters.run_type train \
    training_parameters.restart True \
    training_parameters.find_unused_parameters True \
    training_parameters.num_workers 10 \
    task_attributes.univl_task.dataset_attributes.video_text_retrieval.train_ensemble_n_clips 16 \
    task_attributes.univl_task.dataset_attributes.video_text_retrieval.test_ensembel_n_clips 16 \
    training_parameters.resume_file ${PRETRAINED_SAVE_DIR}/YourPath/xxx.ckpt \
    training_parameters.save_dir ${DIDEMO_CN_FINETUNE_SAVE_DIR}/test \
    training_parameters.batch_size 32 \
    training_parameters.test_batch_size 32 \
    model_attributes.univl.encoder_lr_decay 1.0 \
    optimizer_attributes.params.lr 1e-5