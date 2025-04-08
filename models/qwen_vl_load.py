from PIL import Image 
import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base_model import VLM_BaseModel


model_path = {
    'qwen-vl': './models/Qwen-VL/checkpoints/Qwen-VL',
    'qwen-vl-chat': './models/Qwen-VL/checkpoints/Qwen-VL-Chat'
}

class Qwen_VL_Chat(VLM_BaseModel):
    def __init__(self, model_path=model_path, model_name='qwen-vl-chat', **kwargs):
        super().__init__(model_name, **kwargs)
        self.inst_suff = self.inst_suff + ' Answer:'
        self.tokenizer = AutoTokenizer.from_pretrained(model_path[model_name], trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path[model_name], device_map="cuda", trust_remote_code=True).eval()


    def generate(self, instruction, images, interleaved=False):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """

        instruction = self.inst_pre + instruction + self.inst_suff
        history = None
        query = self.tokenizer.from_list_format([
            {'image': images[0]}, # Either a local path or an url
            {'text': instruction},
        ])
        with torch.no_grad():
            response, history = self.model.chat(self.tokenizer, query=query, history=history, max_new_tokens=self.config.get('max_new_tokens', 512))
            
        return response
