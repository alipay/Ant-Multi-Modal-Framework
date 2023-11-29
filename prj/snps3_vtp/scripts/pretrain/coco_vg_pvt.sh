export PYTHONPATH=`pwd`:$PYTHONPATH
export PYTORCH_TRANSFORMERS_CACHE='/YourPath/'
export TORCH_HOME='/YourPath/'

CONFIG=prj/snps3_vtp/configs/univl/video/pretrain/coco_vg_pvt.yml
SAVE_DIR=/YourPath/

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m antmmf.utils.launch \
    --nproc_per_node=8 --master_port=12371  prj/snps3_vtp/run.py \
    --config ${CONFIG} \
    training_parameters.distributed True \
    training_parameters.run_type train \
    training_parameters.restart True \
    training_parameters.find_unused_parameters True \
    training_parameters.num_workers 6 \
    training_parameters.batch_size 128 \
    training_parameters.test_batch_size 64 \
    optimizer_attributes.params.lr 5e-5 \
    optimizer_attributes.params.weight_decay 1e-3 \
    training_parameters.enable_amp True \
    training_parameters.save_dir ${SAVE_DIR}/test \
    task_attributes.univl_task.dataset_attributes.video_text_pretrain.processors.caption_processor.params.intra_VTM.IW_MLM True \
    model_attributes.univl.pretraining_heads.Vision_Word_Matching True \
    model_attributes.univl.pretraining_heads.MASK_All_IWords_info.VWM_count_stage "after" \
    model_attributes.univl.pretraining_heads.MASK_All_IWords_info.Word_Chosen_Num 3
