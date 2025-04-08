cd ..
set -e
conda activate base


MODEL_LIST="deepseek-vl2"

DATASET_LIST="dysca-clean pope openchair hqh dysca-attack dysca-corruption dysca-typographic  vlbold-ri vlbold-r vlbold-g vlbold-p-emo vlbold-p-nohuman vlbbq m3oralbench-classification m3oralbench-judge m3oralbench-choose_image toxicity-b toxicity-pr toxicity-po figstep mmsafetybench vpr p2-insensitive p2-sensitive vispr_leakage"

python run_calculation.py --model_list $MODEL_LIST --dataset_list $DATASET_LIST

