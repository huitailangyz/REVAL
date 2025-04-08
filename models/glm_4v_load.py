import torch
torch.backends.cudnn.enabled = False
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch

from .base_model import VLM_BaseModel


model_path = './models/GLM-4/checkpoints/glm-4v-9b'


class GLM_4V(VLM_BaseModel):
    def __init__(self, model_path=model_path, model_name='glm-4v-9b', **kwargs):
        super().__init__(model_name, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True)
            device_map = infer_auto_device_map(model, max_memory={0:'16GiB', 1:'16GiB',}, no_split_module_classes=['GLMBlock'])
            self.model = load_checkpoint_and_dispatch(model, model_path, device_map=device_map).eval()

    def generate(self, instruction, images, interleaved=False):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """

        instruction = self.inst_pre + instruction + self.inst_suff
        image = Image.open(images[0]).convert('RGB')
        inputs = self.tokenizer.apply_chat_template([{"role": "user", "image": image, "content": instruction}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True)
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                max_length = self.config.get('max_length', 512),
                top_k=self.config.get('top_k', 1),
            )
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(outputs[0])
            response = response[:response.find("<|endoftext|>")]
        return response