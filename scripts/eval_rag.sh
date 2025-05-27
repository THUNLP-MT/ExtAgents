#!/bin/bash

default_target_dir=./results_rag

if [ $# -ge 1 ]; then
    target_dir=$1
else
    target_dir=$default_target_dir
fi

python src/eval/generate_pred.py --final_preds_file ${target_dir}/final_preds.jsonl --output_file ${target_dir}/predictions.json
python src/eval/hotpot_evaluate_v1.py ${target_dir}/predictions.json ./data/sampled_hotpot_questions.json
