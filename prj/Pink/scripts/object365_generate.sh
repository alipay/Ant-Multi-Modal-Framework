export PYTHONPATH=$PYTHONPATH:./

output_vg_dir=path_to_model_after_stage2
question_file=./
image-folder=path_to_object365
CHUNKS=8
for IDX in {0..7}; do
    let idx_id=${IDX}
    CUDA_VISIBLE_DEVICES=${IDX} python pink/eval/model_object365.py \
        --model-name ${output_vg_dir} \
        --question-file ${question_file} \
        --image-folder ${image_folder} \
        --answers-file ./object365_${idx_id}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx ${idx_id} \
        --conv-mode 'llamav2'
done