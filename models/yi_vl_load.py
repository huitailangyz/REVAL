import sys
import argparse
import torch
torch.backends.cudnn.enabled = False
from PIL import Image 

from .base_model import VLM_BaseModel

sys.path.append("./models/Yi/VL")
from llava.conversation import conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    expand2square,
    get_model_name_from_path,
    load_pretrained_model,
    tokenizer_image_token,
)
from llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info


model_path = './models/Yi/checkpoints/Yi-VL-6B'

class Yi_VL(VLM_BaseModel):
    def __init__(self, model_path=model_path, model_name='yi-vl', **kwargs):
        super().__init__(model_name, **kwargs)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(model_path)
        self.model = self.model.to(dtype=torch.bfloat16).cuda()
        


    def generate(self, instruction, images, interleaved=False):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        instruction = self.inst_pre + instruction + self.inst_suff
        image = Image.open(images[0]).convert('RGB')
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        instruction = DEFAULT_IMAGE_TOKEN + "\n" + instruction
        conv = conv_templates["mm_default"].copy()
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") .unsqueeze(0).cuda())

        stop_str = conv.sep
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda(),
                do_sample=True,
                num_beams=self.config.get('num_beams', 3),
                temperature=self.config.get('temperature', 0.7),
                top_p=self.config.get('top_p', 1.0),
                max_new_tokens=self.config.get('max_new_tokens', 512),
                stopping_criteria=[stopping_criteria],
                use_cache=True,
            )
        input_token_len = input_ids.shape[1]
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()

        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        return outputs
