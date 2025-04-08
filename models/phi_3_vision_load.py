import sys
import argparse
import torch
torch.backends.cudnn.enabled = False
from PIL import Image 
from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

from .base_model import VLM_BaseModel


model_path = './models/Phi-3-vision-128k-instruct'

class Phi_3_vision(VLM_BaseModel):
    def __init__(self, model_path=model_path, model_name='phi-3-vision-128k-instruct', **kwargs):
        super().__init__(model_name, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='eager') # use _attn_implementation='eager' to disable flash attention
        # self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='eager').to(dtype=torch.float16) # use _attn_implementation='eager' to disable flash attention
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.inst_pre = "<|image_1|>\n" + self.inst_pre

    def generate(self, instruction, images, interleaved=False):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        instruction = self.inst_pre + instruction + self.inst_suff
        image = Image.open(images[0]).convert('RGB')

        messages = [{'role': 'user', 'content': instruction}]
        
        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(prompt, [image], return_tensors="pt").to("cuda") 
        with torch.inference_mode():
            generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, 
                                               num_beams=self.config.get('num_beams', 3),
                                               temperature=self.config.get('temperature', 0.7),
                                               max_new_tokens=self.config.get('max_new_tokens', 512)
                                            ) 

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

        return response
