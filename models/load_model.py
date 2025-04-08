def load_model(model_name, **kwargs):
    if model_name == 'minigpt4_vicuna-7b':
        from .minigpt4_load import MiniGPT4_Vicuna_7B
        model = MiniGPT4_Vicuna_7B(**kwargs)
    elif model_name == 'minigpt4_vicuna-13b':
        from .minigpt4_load import MiniGPT4_Vicuna_13B
        model = MiniGPT4_Vicuna_13B(**kwargs)
    elif model_name == 'minigpt4_llama_2':
        from .minigpt4_load import MiniGPT4_LLaMA_2
        model = MiniGPT4_LLaMA_2(**kwargs)
    elif model_name == 'minigpt_v2':
        from .minigpt_v2_load import MiniGPT_v2
        model = MiniGPT_v2(**kwargs)
    elif model_name == 'blip2_flan-t5-xl':
        from .blip2_load import BLIP2_Flan_T5_XL
        model = BLIP2_Flan_T5_XL(**kwargs)
    elif model_name == 'blip2-opt-3b':
        from .blip2_load import BLIP2_OPT_3B
        model = BLIP2_OPT_3B(**kwargs)
    elif model_name == 'blip2-opt-7b':
        from .blip2_load import BLIP2_OPT_7B
        model = BLIP2_OPT_7B(**kwargs)
    elif model_name == 'instructblip_vicuna-7b':
        from .instructblip_load import InstructBLIP_Vicuna_7B
        model = InstructBLIP_Vicuna_7B(**kwargs)
    elif model_name == 'instructblip_vicuna-13b':
        from .instructblip_load import InstructBLIP_Vicuna_13B
        model = InstructBLIP_Vicuna_13B(**kwargs)
    elif model_name == 'instructblip_flan-t5-xl':
        from .instructblip_load import InstructBLIP_Flan_T5_XL
        model = InstructBLIP_Flan_T5_XL(**kwargs)
    elif model_name == 'instructblip_flan-t5-xxl':
        from .instructblip_load import InstructBLIP_Flan_T5_XXL
        model = InstructBLIP_Flan_T5_XXL(**kwargs)
    elif model_name == 'llava_1.5-7b':
        from .llava_load import LLaVA_1_5_7B
        model = LLaVA_1_5_7B(**kwargs)
    elif model_name == 'llava_1.5-13b':
        from .llava_load import LLaVA_1_5_13B
        model = LLaVA_1_5_13B(**kwargs)
    elif model_name == 'otter':
        from .otter_load import Otter
        model = Otter(**kwargs)
    elif model_name == 'shikra-7b':
        from .shikra_load import Shikra_7B
        model = Shikra_7B(**kwargs)
    elif model_name == 'qwen-vl-chat':
        from .qwen_vl_load import Qwen_VL_Chat
        model = Qwen_VL_Chat(**kwargs)
    elif model_name == 'internlm-xcomposer-vl-7b':
        from .internlm_xcomposer_vl_load import InternLM_XComposer_VL_7B
        model = InternLM_XComposer_VL_7B(**kwargs)
    elif model_name == 'internlm-xcomposer2-vl-7b':
        from .internlm_xcomposer_vl_load import InternLM_XComposer2_VL_7B
        model = InternLM_XComposer2_VL_7B(**kwargs)
    elif model_name == 'emu2-chat':
        from .emu2_chat_load import Emu2_Chat
        model = Emu2_Chat(**kwargs)
    elif model_name == 'glm-4v-9b':
        from .glm_4v_load import GLM_4V
        model = GLM_4V(**kwargs)
    elif model_name == 'minicpm-llama2-v2.5':
        from .minicpm_v_load import MiniCPM_Llama3_V_2_5
        model = MiniCPM_Llama3_V_2_5(**kwargs)
    elif model_name == 'yi-vl':
        from .yi_vl_load import Yi_VL
        model = Yi_VL(**kwargs)
    elif model_name == 'mplug-owl2':
        from .mplug_owl2_load import mPLUG_Owl2
        model = mPLUG_Owl2(**kwargs)
    elif model_name == 'phi-3-vision':
        from .phi_3_vision_load import Phi_3_vision
        model = Phi_3_vision(**kwargs)
    elif model_name == 'deepseek-vl2':
        from .deepseek_vl2_load import DeepSeek_VL2
        model = DeepSeek_VL2(**kwargs)
    else:
        raise NotImplementedError(f"{model_name} not implemented")
    return model