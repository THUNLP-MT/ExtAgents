#!/bin/bash

default_target_dir=./results_en

if [ $# -ge 1 ]; then
    target_dir=$1
else
    target_dir=$default_target_dir
fi

python src/eval/compute_scores_partial-enhanced.py --task longbook_qa_eng --preds_path ${target_dir}/final_preds.jsonl
