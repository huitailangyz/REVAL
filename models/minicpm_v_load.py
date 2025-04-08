import sys
import argparse
import torch
torch.backends.cudnn.enabled = False
from transformers import AutoModel, AutoTokenizer
from PIL import Image 

from .base_model import VLM_BaseModel


model_path = './models/MiniCPM-V/checkpoints/MiniCPM-Llama3-V-2_5'

class MiniCPM_Llama3_V_2_5(VLM_BaseModel):
    def __init__(self, model_path=model_path, model_name='minicpm-llama3-v-2_5', **kwargs):
        super().__init__(model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()

    def generate(self, instruction, images, interleaved=False):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        instruction = self.inst_pre + instruction + self.inst_suff
        image = Image.open(images[0]).convert('RGB')
        msgs = [{'role': 'user', 'content': instruction}]
        pred = self.model.chat(
            image=image,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=True, # if sampling=False, beam_search will be used by default
            temperature=self.config.get('temperature', 0.7),
            max_new_tokens=self.config.get('max_new_tokens', 512),
        )
        return pred
