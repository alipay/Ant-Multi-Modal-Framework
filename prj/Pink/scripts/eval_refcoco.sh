export PYTHONPATH=$PYTHONPATH:./

model_name=./
question_file=./
image_folder=./
output_path=./

CUDA_VISIBLE_DEVICES=0 python pink/eval/model_vg_base_batch.py \
    --model-name ${model_name} \
    --question-file ${question_file} \
    --image-folder ${image_folder} \
    --answers-file ${output_path} \
    --conv-mode 'llamav2'