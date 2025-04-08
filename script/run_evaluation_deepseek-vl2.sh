cd ..
set -e
conda activate deepseek_vl2

INST_PRE="Please use only 'yes' or 'no' to answer the following question: "


# Perception
DATASET_LIST="dysca-clean"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST

# Hallucination
DATASET_LIST="pope"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST --cfg_options "{\"inst_pre\": \"${INST_PRE}\"}"

DATASET_LIST="openchair"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST

DATASET_LIST="hqh"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST

# Robustness
DATASET_LIST="dysca-attack"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST

DATASET_LIST="dysca-corruption"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST

DATASET_LIST="dysca-typographic"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST

# Bias
DATASET_LIST="vlbold-ri"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST

DATASET_LIST="vlbold-r"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST

DATASET_LIST="vlbold-g"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST

DATASET_LIST="vlbold-p-emo"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST

DATASET_LIST="vlbold-p-nohuman"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST

DATASET_LIST="vlbbq"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST


# Morality
DATASET_LIST="m3oralbench-classification"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST

DATASET_LIST="m3oralbench-judge"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST

DATASET_LIST="m3oralbench-choose_image"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST


# Toxicity
DATASET_LIST="toxicity-b"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST

DATASET_LIST="toxicity-pr"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST

DATASET_LIST="toxicity-po"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST

# Jailbreak
DATASET_LIST="figstep"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST

DATASET_LIST="mmsafetybench"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST


# Privacy Awareness
DATASET_LIST="vpr"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST

DATASET_LIST="p2-insensitive"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST

DATASET_LIST="p2-sensitive"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST

# Privacy Leakage
DATASET_LIST="vispr_leakage"
CUDA_VISIBLE_DEVICES=0,1,2,3,4 python run_evaluation.py --model_name 'deepseek-vl2' --dataset_list $DATASET_LIST