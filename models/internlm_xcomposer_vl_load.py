import sys
import argparse
import torch
from transformers import AutoModel, AutoTokenizer

from PIL import Image 
import torch 
from accelerate import dispatch_model

from .base_model import VLM_BaseModel
sys.path.append("./models/InternLM-XComposer/checkpoints/internlm-xcomposer2-vl-7b")
from configuration_internlm_xcomposer2 import InternLMXcomposer2Config
from modeling_internlm_xcomposer2 import InternLMXComposer2ForCausalLM


model_path = {
    'internlm-xcomposer-vl-7b': './models/InternLM-XComposer/checkpoints/internlm-xcomposer-vl-7b',
    'internlm-xcomposer2-vl-7b': './models/InternLM-XComposer/checkpoints/internlm-xcomposer2-vl-7b'
}

class InternLM_XComposer_VL_7B(VLM_BaseModel):
    def __init__(self, model_path=model_path, model_name='internlm-xcomposer-vl-7b', **kwargs):
        super().__init__(model_name, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_path[model_name], trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path[model_name], trust_remote_code=True, torch_dtype=torch.bfloat16).cuda().eval()
        # device_map = self.auto_configure_device_map(2)
        # self.model = dispatch_model(model, device_map=device_map)
        self.model = model
        self.model.tokenizer = tokenizer

    def auto_configure_device_map(self, num_gpus):
        # visual_encoder 算4层
        # internlm_model.model.embed_tokens 占用1层
        # norm 和 lm_head 占用1层
        # transformer.layers 占用 32 层
        # 总共34层分配到num_gpus张卡上
        num_trans_layers = 32
        per_gpu_layers = 38 / num_gpus

        device_map = {
            'visual_encoder': 0,
            'ln_vision': 0,
            'Qformer': 0,
            'internlm_model.model.embed_tokens': 0,
            'internlm_model.model.norm': 0,
            'internlm_model.lm_head': 0,
            'query_tokens': 0,
            'flag_image_start': 0,
            'flag_image_end': 0,
            'internlm_proj.weight': 0,
            'internlm_proj.bias': 0,
        }
        used = 6
        gpu_target = 0
        for i in range(num_trans_layers):
            if used >= per_gpu_layers:
                gpu_target += 1
                used = 0
            assert gpu_target < num_gpus
            device_map[f'internlm_model.model.layers.{i}'] = gpu_target
            used += 1
        return device_map

    def generate(self, instruction, images, interleaved=False):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        # internlm_xcomposer_vl only supports single image
        instruction = self.inst_pre + instruction + self.inst_suff
        with torch.no_grad():
            pred = self.model.generate(
                text=instruction,
                image=images[0],
                num_beams=self.config.get('num_beams', 5),
                do_sample=True,
                min_length=1,
                repetition_penalty=1.5,
                length_penalty=self.config.get('length_penalty', 1.0),
                temperature=self.config.get('temperature', 1.0),
                top_p=self.config.get('top_p', 1.0),
                max_new_tokens=self.config.get('max_new_tokens', 500),
            )
        return pred


class InternLM_XComposer2_VL_7B(VLM_BaseModel):
    def __init__(self, model_path=model_path, model_name='internlm-xcomposer2-vl-7b', **kwargs):
        super().__init__(model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path[model_name], trust_remote_code=True)
        config = InternLMXcomposer2Config.from_pretrained(model_path[model_name])
        self.model = InternLMXComposer2ForCausalLM.from_pretrained(model_path[model_name], config=config, torch_dtype=torch.bfloat16).cuda().eval()
        self.inst_pre = '<ImageHere>' + self.inst_pre

    def auto_configure_device_map(self, num_gpus):
        # visual_encoder 算4层
        # internlm_model.model.embed_tokens 占用1层
        # norm 和 lm_head 占用1层
        # transformer.layers 占用 32 层
        # 总共34层分配到num_gpus张卡上
        num_trans_layers = 32
        per_gpu_layers = 38 / num_gpus

        device_map = {
            'visual_encoder': 0,
            'ln_vision': 0,
            'Qformer': 0,
            'internlm_model.model.embed_tokens': 0,
            'internlm_model.model.norm': 0,
            'internlm_model.lm_head': 0,
            'query_tokens': 0,
            'flag_image_start': 0,
            'flag_image_end': 0,
            'internlm_proj.weight': 0,
            'internlm_proj.bias': 0,
        }
        used = 6
        gpu_target = 0
        for i in range(num_trans_layers):
            if used >= per_gpu_layers:
                gpu_target += 1
                used = 0
            assert gpu_target < num_gpus
            device_map[f'internlm_model.model.layers.{i}'] = gpu_target
            used += 1
        return device_map

    def generate(self, instruction, images, interleaved=False):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        instruction = self.inst_pre + instruction + self.inst_suff
        with torch.cuda.amp.autocast():
            response, _ = self.model.chat(self.tokenizer, query=instruction, image=images[0], history=[], do_sample=False, max_new_tokens=self.config.get('max_new_tokens', 512))
        return response
