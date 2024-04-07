export PYTHONPATH=`pwd`/../../:$PYTHONPATH
export PYTORCH_TRANSFORMERS_CACHE='/YourPath/'
export TORCH_HOME='/YourPath/'

# eval: retrieval stage2
MSRVTT_FINETUNE_SAVE_DIR=/YourCheckpointPath/
ROOT_DIR=/YourPath/

CUDA_VISIBLE_DEVICES=1 \
python -m antmmf.utils.launch --nproc_per_node=1 --master_port=12201 prj/dmae_vtp/run.py \
  --config ${CONFIG} \
  training_parameters.resume_file ${MSRVTT_FINETUNE_SAVE_DIR}/YourPath/xxx.ckpt \
  training_parameters.save_dir ${ROOT_DIR}/test \
  training_parameters.distributed True \
  training_parameters.run_type predict \
  training_parameters.restart True \
  training_parameters.find_unused_parameters True \
  training_parameters.num_workers 2 \
  training_parameters.test_batch_size 1
  task_attributes.univl_task.dataset_attributes.video_text_retrieval.test_ensembel_n_clips 10 \
  training_parameters.load_pretrained True \
  training_parameters.monitored_metric val/l3_simi_t2v-mean_recall \
  model_attributes.univl.training_stage stage1+stage2+stage3 \
  model_attributes.univl.l3_with_nfc True \
  model_attributes.univl.l3_loss_type negNCE \
  model_attributes.univl.l3_interaction att_ti \
  task_attributes.univl_task.dataset_attributes.video_text_retrieval.l3_use_twm True  \
