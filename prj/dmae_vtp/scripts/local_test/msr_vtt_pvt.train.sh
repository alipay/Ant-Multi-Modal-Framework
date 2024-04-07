export PYTHONPATH=`pwd`/../../:$PYTHONPATH
export PYTORCH_TRANSFORMERS_CACHE='/YourPath/'
export TORCH_HOME='/YourPath/'

CONFIG=configs/univl/video/finetune_retrieval/msr_vtt_pvt.yml 
# finetune:  retrieval stage1
SAVE_DIR=/YourPath1/
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m antmmf.utils.launch \
    --nproc_per_node=8 --master_port=12582 run.py \
    --config ${CONFIG} \
    training_parameters.distributed True \
    training_parameters.run_type train \
    training_parameters.restart True \
    training_parameters.find_unused_parameters True \
    training_parameters.num_workers 10 \
    task_attributes.univl_task.dataset_attributes.video_text_retrieval.train_ensemble_n_clips 1 \
    task_attributes.univl_task.dataset_attributes.video_text_retrieval.test_ensembel_n_clips 1 \
    training_parameters.save_dir ${SAVE_DIR} \
    training_parameters.batch_size 160 \
    training_parameters.monitored_metric val/l1_simi_t2v-mean_recall \
    model_attributes.univl.encoder_lr_decay 1.0 \
    model_attributes.univl.training_stage stage1+stage2+stage3 \
    optimizer_attributes.params.lr 1e-5
    
if false;then
# finetune:  retrieval stage2
PRETRAINED_CKPT=/YourPath1/
SAVE_DIR=/YourPath2/
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m antmmf.utils.launch \
    --nproc_per_node=8 --master_port=12591 run.py \
    --config ${CONFIG} \
    training_parameters.distributed True \
    training_parameters.run_type train \
    training_parameters.restart True \
    training_parameters.find_unused_parameters True \
    training_parameters.num_workers 10 \
    task_attributes.univl_task.dataset_attributes.video_text_retrieval.train_ensemble_n_clips 10 \
    task_attributes.univl_task.dataset_attributes.video_text_retrieval.test_ensembel_n_clips 10 \
    training_parameters.resume_file ${PRETRAINED_CKPT} \
    training_parameters.save_dir ${SAVE_DIR} \
    training_parameters.batch_size 16 \
    training_parameters.test_batch_size 96 \
    training_parameters.load_pretrained True \
    training_parameters.monitored_metric val/l3_simi_t2v-mean_recall \
    model_attributes.univl.encoder_lr_decay 1.0 \
    model_attributes.univl.training_stage stage1+stage2+stage3 \
    optimizer_attributes.params.lr 1e-5 \
    model_attributes.univl.l3_with_nfc True \
    model_attributes.univl.l3_loss_type negNCE \
    model_attributes.univl.l3_interaction att_ti \
    task_attributes.univl_task.dataset_attributes.video_text_retrieval.l3_use_twm True  \
fi
