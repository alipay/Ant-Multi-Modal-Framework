export PYTHONPATH=$PYTHONPATH:./

output_dir=dir_need_to_change
if [ -d ${output_dir} ];then
    echo "dir already exists"
else
    mkdir ${output_dir}
fi

llama_path=path_to_llama_weight
llava_cc3m_pretrain_data_path=./
llava_cc3m_pretrain_base_path=./
llava_cc3m_pretrain_image_folder=./

# stage 1 multi-modal align

OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 --master_port=25002 \
    llava/train/train.py \
    --model_name_or_path ${llama_path} \
    --llama_path ${llama_path} \
    --dataset_name LLaVACaptionDataset \
    --data_path ${llava_cc3m_pretrain_data_path} \
    --image_folder ${llava_cc3m_pretrain_image_folder} \
    --base_path ${llava_cc3m_pretrain_base_path} \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --conversation_template llamav2 \
    --freeze_llm True \
    --llm_adapter_enable False \
    --visual_adapter_enable False \
    --freeze_vit True \
    --bf16 True \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400000 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --dataloader_num_workers 4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --report_to tensorboard
