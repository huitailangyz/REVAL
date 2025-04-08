import torch
from transformers import AutoModelForCausalLM
from .base_model import VLM_BaseModel

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


model_path = "./models/DeepSeek-VL2/checkpoints/deepseek-vl2"


def split_model():
    device_map = {}
    num_layers_per_gpu = [2, 7, 7, 7, 7]
    num_layers =  sum(num_layers_per_gpu)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision'] = 0
    device_map['projector'] = 0
    device_map['image_newline'] = 0
    device_map['view_seperator'] = 0
    device_map['language.model.embed_tokens'] = 0
    device_map['language.model.norm'] = 0
    device_map['language.lm_head'] = 0
    device_map[f'language.model.layers.{num_layers - 1}'] = 0
    return device_map



class DeepSeek_VL2(VLM_BaseModel):
    def __init__(self, model_path=model_path, model_name='deepseek-vl2', **kwargs):
        super().__init__(model_name, **kwargs)
        # specify the path to the model
        self.vl_chat_processor = DeepseekVLV2Processor.from_pretrained(model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        device_map = split_model()
        self.vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,torch_dtype=torch.bfloat16, device_map=device_map).eval()

    def generate(self, instruction, images, interleaved=False):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        ## single image conversation example
        instruction = self.inst_pre + "<image>\n" + instruction + self.inst_suff

        conversation = [
            {
                "role": "<|User|>",
                "content": instruction,
                "images": images,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        # load images and prepare for inputs
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True,
            system_prompt=""
        ).to(self.vl_gpt.device)

        # run image encoder to get the image embeddings
        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

        # run the model to get the response
        outputs = self.vl_gpt.language.generate(
            inputs_embeds=inputs_embeds,
            input_ids = prepare_inputs["input_ids"],
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=self.config.get('max_new_tokens', 512),
            do_sample=False,
            use_cache=True
        )

        generated_tokens = outputs[0][len(prepare_inputs["input_ids"][0]):]  # Skip the input tokens
        answer = self.tokenizer.decode(generated_tokens.cpu().tolist(), skip_special_tokens=True)
        return answer
