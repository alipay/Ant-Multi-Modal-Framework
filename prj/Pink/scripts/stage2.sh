export PYTHONPATH=$PYTHONPATH:./

output_dir=path_to_stage1

output_sft_dir=path_to_sft
if [ -d ${output_sft_dir} ];then
    echo "dir already exists"
else
    mkdir ${output_sft_dir}
    cp $0 ${output_sft_dir}/$(basename "$0")
fi
llama_path=path_to_llama_weight

base_path=path_to_dataset_dir
visualgenome_dataset_path=./
coco_dataset_path=./
flickr_dataset_path=./

visualgenome_region_descriptions_path=./
LLaVA-115K_path=./
vqav2_path=./
a-okvqa_path=./
flickr_path=./

OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 --master_port=25000 \
    llava/train/train.py \
    --dataset_name VisualGenomeDataset@InstructCaptionDataset@VQAv2Dataset@AOKVQADataset@FlickrEntityDataset \
    --model_name_or_path ${output_dir} \
    --llama_path ${llama_path} \
    --data_path ${visualgenome_region_descriptions_path}@${LLaVA-115K_path}@${vqav2_path}@${a-okvqa_path}@${flickr_path} \
    --image_folder ${visualgenome_dataset_path}@${coco_dataset_path}/train2017@${coco_dataset_path}@${coco_dataset_path}/train2017@${flickr_dataset_path} \
    --base_path ${path_to_base_path} \
    --vision_tower openai/clip-vit-large-patch14 \
    --task_pool RegionGroundingCaption,VisualGrounding,Relation,Counting,CoarseLocation,Detection \
    --conversation_template llamav2 \
    --add_mark_tokens True \
    --mm_vision_select_layer -2 \
    --tune_mm_mlp_adapter True \
    --freeze_llm False \
    --llm_adapter_enable True \
    --freeze_vit False \
    --visual_adapter_enable True \
    --bf16 True \
    --output_dir ${output_sft_dir} \
    --num_train_epochs 6 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --weight_decay 0.02 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --dataloader_num_workers 4 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --report_to tensorboard
