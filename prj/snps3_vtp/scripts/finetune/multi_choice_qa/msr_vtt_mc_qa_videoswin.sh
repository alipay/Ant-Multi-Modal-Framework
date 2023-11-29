export PYTHONPATH=`pwd`:$PYTHONPATH
export PYTORCH_TRANSFORMERS_CACHE='/YourPath/'
export TORCH_HOME='/YourPath/'

CONFIG=prj/snps3_vtp/configs/univl/video/finetune_multi_choice_qa/msr_vtt_mc_qa_videoswin.yml
MSRVTT_FINETUNE_SAVE_DIR=/YourCheckpointPath/
ROOT_DIR=/YourPath/

CUDA_VISIBLE_DEVICES=1 \
python -m antmmf.utils.launch --nproc_per_node=1 --master_port=12201 prj/snps3_vtp/run.py \
  --config ${CONFIG} \
  training_parameters.resume_file ${MSRVTT_FINETUNE_SAVE_DIR}/YourPath/xxx.ckpt \
  training_parameters.save_dir ${ROOT_DIR}/test \
  training_parameters.distributed True \
  training_parameters.run_type predict \
  training_parameters.restart True \
  training_parameters.find_unused_parameters True \
  training_parameters.num_workers 2 \
  training_parameters.test_batch_size 1