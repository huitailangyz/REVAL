# REVAL: A Comprehension Evaluation on Reliability and Values of Large Vision-Language Models
This repository is an official implementation of the paper [REVAL](https://arxiv.org/abs/2503.16566).

<div align="center">
<a href='https://arxiv.org/abs/2503.16566'><img src='https://img.shields.io/badge/Paper-Arxiv-blue'></a>
<a href='https://github.com/huitailangyz/REVAL'><img src='https://img.shields.io/badge/Github-Page-Green'></a>
<a href='https://huggingface.co/datasets/huitailang/REVAL'><img src='https://img.shields.io/badge/Hugging%20Face-Datasets-red'></a>
</div>



## Introduction

The rapid evolution of Large Vision-Language Models (LVLMs) has highlighted the necessity for comprehensive evaluation frameworks that assess these models across diverse dimensions. While existing benchmarks focus on specific aspects such as perceptual abilities, cognitive capabilities, and safety against adversarial attacks, they often lack the breadth and depth required to provide a holistic understanding of LVLMs' strengths and limitations. To address this gap, we introduce REVAL, a comprehensive benchmark designed to evaluate the **RE**liability and **VAL**ue of LVLMs. REVAL encompasses over 144K image-text Visual Question Answering (VQA) samples, structured into two primary sections: Reliability, which assesses truthfulness (e.g., perceptual accuracy and hallucination tendencies) and robustness (e.g., resilience to adversarial attacks, typographic attacks, and image corruption), and Values, which evaluates ethical concerns (e.g., bias and moral understanding), safety issues (e.g., toxicity and jailbreak vulnerabilities), and privacy problems (e.g., privacy awareness and privacy leakage). We evaluate 26 models, including mainstream open-source LVLMs and prominent closed-source models like GPT-4o and Gemini-1.5-Pro. Our findings reveal that while current LVLMs excel in perceptual tasks and toxicity avoidance, they exhibit significant vulnerabilities in adversarial scenarios, privacy preservation, and ethical reasoning. These insights underscore critical areas for future improvements, guiding the development of more secure, reliable, and ethically aligned LVLMs. REVAL provides a robust framework for researchers to systematically assess and compare LVLMs, fostering advancements in the field.
<div align="center">
    <img src="figures/framework.jpg" alt="framework" style="width: 100%;"> 
</div>

## Dataset

|                 | Topic            |                    | Dataset          | # Sample  | Question Type  |
|-----------------|------------------|--------------------|------------------|-----------|----------------|
| **Reliability** | **Truthfulness** | Perception         | Dysca            | 28,217    | TF/MC/FF       |
|                 |                  | Hallucination      | POPE             | 3,000     | TF             |
|                 |                  |                    | OpenCHAIR        | 2,000     | FF             |
|                 |                  |                    | HQH              | 4,000     | FF             |
|                 | **Robustness**   | Adversarial Attack | Dysca            | 28,217    | TF/MC/FF       |
|                 |                  | Typographic Attack | Dysca            | 26,379    | TF/MC/FF       |
|                 |                  | Image Corruption   | Dysca            | 28,217    | TF/MC/FF       |
| **Values**      | **Ethics**       | Bias               | VLBiasBench      | 12,636    | FF             |
|                 |                  | Morality           | M$^3$oralBench   | 4,640     | MC             |
|                 | **Safety**       | Toxicity           | ToViLaG          | 2,157     | FF             |
|                 |                  | Jailbreak          | FigStep          | 500       | FF             |
|                 |                  |                    | MMSafetyBench    | 1,680     | FF             |
|                 | **Privacy**      | Privacy Awareness  | Self-constructed | 2,182     | TF             |
|                 |                  | Privacy Leakage    | Self-constructed | 586       | FF             |
|                 |                  | **Total**          |                  | 144,411   |                |

* `TF` represents the true-or-false question, `MC` represents the multi-choice question, `FF` represents the free-form question


## Main Result
<div align="center">
    <img src="figures/result.jpg" alt="result" style="width: 100%;"> 
    <img src="figures/reliability.jpg" alt="reliability" style="width: 100%;"> 
    <img src="figures/value.jpg" alt="value" style="width: 100%;"> 
</div>

## Usage
### Dataset Preparation
Please get the REVAL dataset at [here](https://huggingface.co/datasets/huitailang/REVAL) and place it in the `./data` path

### Model Preparation
Please refer to the installation instructions of each model and place the model file under the `./models/XXX` folder.

Below we list the github of the models we evaluated, as well as the download addresses of the models.

| Model Name                |      Github     | Download  | 
|---------------------------|-----------|----------------|
| MiniGPT4-Vicuna-7B        | [MiniGPT4-Vicuna-7B](https://github.com/Vision-CAIR/MiniGPT-4) | [MiniGPT4-Vicuna-7B](https://drive.google.com/file/d/1RY9jV0dyqLX-o38LrumkKRh6Jtaop58R/view) |
| MiniGPT4-Vicuna-13B       | [MiniGPT4-Vicuna-13B](https://github.com/Vision-CAIR/MiniGPT-4) | [MiniGPT4-Vicuna-13B](https://drive.google.com/file/d/1a4zLvaiDBr-36pasffmgpvH5P7CKmpze/view) |
| MiniGPT4-Llama2           | [MiniGPT4-Llama2](https://github.com/Vision-CAIR/MiniGPT-4) | [MiniGPT4-Llama2](https://drive.google.com/file/d/11nAPjEok8eAGGEG1N2vXo3kBLCg0WgUk/view) |
| MiniGPT-v2                | [MiniGPT-v2](https://github.com/Vision-CAIR/MiniGPT-4) | [MiniGPT-v2](https://drive.google.com/file/d/1HkoUUrjzFGn33cSiUkI-KcT-zysCynAz/view) |
| BLIP2-Flan-T5-XL          | [BLIP2-Flan-T5-XL](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) | [BLIP2-Flan-T5-XL](https://huggingface.co/Salesforce/blip2-flan-t5-xl) |
| BLIP2-OPT-3B              | [BLIP2-OPT-3B](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) | [BLIP2-OPT-3B](https://huggingface.co/Salesforce/blip2-opt-2.7b) |
| BLIP2-OPT-7B              | [BLIP2-OPT-7B](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) | [BLIP2-OPT-7B](https://huggingface.co/Salesforce/blip2-opt-6.7b) |
| InstructBLIP-Vicuna-7B    | [InstructBLIP-Vicuna-7B](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) | [InstructBLIP-Vicuna-7B](https://huggingface.co/Salesforce/instructblip-vicuna-7b) |
| InstructBLIP-Vicuna-13B   | [InstructBLIP-Vicuna-13B](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) | [InstructBLIP-Vicuna-13B](https://huggingface.co/Salesforce/instructblip-vicuna-13b) |
| InstructBLIP-Flan-T5-XL   | [InstructBLIP-Flan-T5-XL](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) | [InstructBLIP-Flan-T5-XL](https://huggingface.co/Salesforce/instructblip-flan-t5-xl) |
| InstructBLIP-Flan-T5-XXL  | [InstructBLIP-Flan-T5-XXL](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) | [InstructBLIP-Flan-T5-XXL](https://huggingface.co/Salesforce/instructblip-flan-t5-xxl) |
| LLaVA-1.5-7B              | [LLaVA-1.5-7B](https://github.com/haotian-liu/LLaVA) | [LLaVA-1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b) |
| LLaVA-1.5-13B             | [LLaVA-1.5-13B](https://github.com/haotian-liu/LLaVA) | [LLaVA-1.5-13B](https://huggingface.co/liuhaotian/llava-v1.5-13b) |
| Otter                     | [Otter](https://github.com/Luodian/Otter) | [Otter](https://huggingface.co/luodian/OTTER-Image-MPT7B) |
| Shikra-7B                 | [Shikra-7B](https://github.com/shikras/shikra) | [Shikra-7B](https://huggingface.co/shikras/shikra-7b-delta-v1) |
| InternLM-XComposer-VL-7B  | [InternLM-XComposer-VL-7B](https://github.com/InternLM/InternLM-XComposer) | [InternLM-XComposer-VL-7B](https://huggingface.co/internlm/internlm-xcomposer-vl-7b) |
| InternLM-XComposer2-VL-7B | [InternLM-XComposer2-VL-7B](https://github.com/InternLM/InternLM-XComposer) | [InternLM-XComposer2-VL-7B](https://huggingface.co/internlm/internlm-xcomposer2-vl-7b) |
| Qwen-VL-Chat              | [Qwen-VL-Chat](https://github.com/QwenLM/Qwen-VL) | [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat) |
| Emu2-Chat                 | [Emu2-Chat](https://github.com/baaivision/Emu/tree/main/Emu2) | [Emu2-Chat](https://huggingface.co/BAAI/Emu2-Chat) |
| GLM-4V-9B                 | [GLM-4V-9B](https://github.com/THUDM/GLM-4) | [GLM-4V-9B](https://huggingface.co/THUDM/glm-4v-9b) |
| MiniCPM-Llama3-v2.5       | [MiniCPM-Llama3-v2.5](https://github.com/OpenBMB/MiniCPM-V) | [MiniCPM-Llama3-v2.5](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5) |
| Yi-VL                     | [Yi-VL](https://github.com/01-ai/Yi/tree/main/VL) | [Yi-VL](https://huggingface.co/01-ai/Yi-VL-6B) |
| mPLUG-Owl2                | [mPLUG-Owl2](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2) | [mPLUG-Owl2](https://huggingface.co/MAGAer13/mplug-owl2-llama2-7b) |
| Phi-3-Vision              | [Phi-3-Vision](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) | [Phi-3-Vision](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct) |




If you need to add a new model for evaluation, prepare the `XXX_load.py` file under the `./models` folder to import the model.

### Script
After downloading the REVAL dataset and installing the model, run the evaluation script under the `./script` folder.
We use the DeepSeek-VL2 model as an example to illustrate this process.

```shell
#!/bin/bash
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
```

After completing the inference of the model, further use the script to calculate the scores of different evaluation dimensions.

```shell
#!/bin/bash
cd ..
set -e
conda activate base

MODEL_LIST="deepseek-vl2"
DATASET_LIST="dysca-clean pope openchair hqh dysca-attack dysca-corruption dysca-typographic  vlbold-ri vlbold-r vlbold-g vlbold-p-emo vlbold-p-nohuman vlbbq m3oralbench-classification m3oralbench-judge m3oralbench-choose_image toxicity-b toxicity-pr toxicity-po figstep mmsafetybench vpr p2-insensitive p2-sensitive vispr_leakage"

python run_calculation.py --model_list $MODEL_LIST --dataset_list $DATASET_LIST
```


## Citation
If you find our work helpful for your research, please consider citing our work.

```bibtex
@misc{zhang2025reval,
    title        = {REVAL: A Comprehension Evaluation on Reliability and Values of Large Vision-Language Models}, 
    author       = {Jie Zhang and Zheng Yuan and Zhongqi Wang and Bei Yan and Sibo Wang and Xiangkui Cao and Zonghui Guo and Shiguang Shan and Xilin Chen},
    year         = {2025},
    eprint       = {2503.16566},
    primaryClass = {cs.CV},
    url          = {https://arxiv.org/abs/2503.16566}, 
}
```