from vlm import *
from vlm.api import *
from functools import partial
import os

PandaGPT_ROOT = None
MiniGPT4_ROOT = None
TransCore_ROOT = None
Yi_ROOT = None
OmniLMM_ROOT = None
Mini_Gemini_ROOT = None
VXVERSE_ROOT = None
VideoChat2_ROOT = None
VideoChatGPT_ROOT = None
PLLaVA_ROOT = None
RBDash_ROOT = None
VITA_ROOT = None
LLAVA_V1_7B_MODEL_PTH = "Please set your local path to LLaVA-7B-v1.1 here, the model weight is obtained by merging LLaVA delta weight based on vicuna-7b-v1.1 in https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md with vicuna-7b-v1.1. "



o1_key = 'XXX'  # noqa: E501
o1_apis = {
    'o1': partial(
        GPT4V,
        model="o1-2024-12-17",
        key=o1_key,
        api_base='OFFICIAL', 
        temperature=0,
        img_detail='high',
        retry=10,
        verbose=False,
    ),
}

api_models = {
    # GPT
    "GPT4V": partial(
        GPT4V,
        model="gpt-4-1106-vision-preview",
        temperature=0,
        img_size=512,
        img_detail="low",
        retry=10,
        verbose=False,
    ),
    "GPT4V_HIGH": partial(
        GPT4V,
        model="gpt-4-1106-vision-preview",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4V_20240409": partial(
        GPT4V,
        model="gpt-4-turbo-2024-04-09",
        temperature=0,
        img_size=512,
        img_detail="low",
        retry=10,
        verbose=False,
    ),
    "GPT4V_20240409_HIGH": partial(
        GPT4V,
        model="gpt-4-turbo-2024-04-09",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4o": partial(
        GPT4V,
        model="gpt-4o-2024-05-13",
        temperature=0,
        img_size=512,
        img_detail="low",
        retry=10,
        verbose=False,
    ),
    "GPT4o_HIGH": partial(
        GPT4V,
        model="gpt-4o-2024-05-13",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4o_20240806": partial(
        GPT4V,
        model="gpt-4o-2024-08-06",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4o_20241120": partial(
        GPT4V,
        model="gpt-4o-2024-11-20",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "ChatGPT4o": partial(
        GPT4V,
        model="chatgpt-4o-latest",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4o_MINI": partial(
        GPT4V,
        model="gpt-4o-mini-2024-07-18",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "GPT4.5": partial(
        GPT4V, 
        model='gpt-4.5-preview-2025-02-27',
        temperature=0, 
        timeout=600,
        img_size=-1, 
        img_detail='high', 
        retry=10, 
        verbose=False,
    ),
    "gpt-4.1-2025-04-14": partial(
        GPT4V,
        model="gpt-4.1-2025-04-14",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "gpt-4.1-mini-2025-04-14": partial(
        GPT4V,
        model="gpt-4.1-mini-2025-04-14",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    "gpt-4.1-nano-2025-04-14": partial(
        GPT4V,
        model="gpt-4.1-nano-2025-04-14",
        temperature=0,
        img_size=-1,
        img_detail="high",
        retry=10,
        verbose=False,
    ),
    # Gemini
    "GeminiPro1-0": partial(
        Gemini, model="gemini-1.0-pro", temperature=0, retry=10
    ),  # now GeminiPro1-0 is only supported by vertex backend
    "GeminiPro1-5": partial(
        Gemini, model="gemini-1.5-pro", temperature=0, retry=10
    ),
    "GeminiFlash1-5": partial(
        Gemini, model="gemini-1.5-flash", temperature=0, retry=10
    ),
    "GeminiPro1-5-002": partial(
        GPT4V, model="gemini-1.5-pro-002", temperature=0, retry=10
    ),  # Internal Use Only
    "GeminiFlash1-5-002": partial(
        GPT4V, model="gemini-1.5-flash-002", temperature=0, retry=10
    ),  # Internal Use Only
    "GeminiFlash2-0": partial(
        Gemini, model="gemini-2.0-flash", temperature=0, retry=10
    ),
    "GeminiFlashLite2-0": partial(
        Gemini, model="gemini-2.0-flash-lite", temperature=0, retry=10
    ),
    "GeminiFlash2-5": partial(
        GPT4V, model="gemini-2.5-flash", temperature=0, retry=10, timeout=1800
    ),
    "GeminiPro2-5": partial(
        GPT4V, model="gemini-2.5-pro", temperature=0, retry=10, timeout=1800
    ),
    
   
    "Claude3V_Opus": partial(
        Claude3V, model="claude-3-opus-20240229", temperature=0, retry=10, verbose=False
    ),
    "Claude3V_Sonnet": partial(
        Claude3V,
        model="claude-3-sonnet-20240229",
        temperature=0,
        retry=10,
        verbose=False,
    ),
    "Claude3V_Haiku": partial(
        Claude3V,
        model="claude-3-haiku-20240307",
        temperature=0,
        retry=10,
        verbose=False,
    ),
    "Claude3-5V_Sonnet": partial(
        Claude3V,
        model="claude-3-5-sonnet-20240620",
        temperature=0,
        retry=10,
        verbose=False,
    ),
    "Claude3-5V_Sonnet_20241022": partial(
        Claude3V,
        model="claude-3-5-sonnet-20241022",
        temperature=0,
        retry=10,
        verbose=False,
    ),
    "Claude3-7V_Sonnet": partial(
        Claude3V,
        model="claude-3-7-sonnet-20250219",
        temperature=0,
        retry=10,
        verbose=False,
    ),
    "Claude4_Opus": partial(
        Claude3V,
        model="claude-4-opus-20250514",
        temperature=0,
        retry=10,
        verbose=False,
        timeout=1800
    ),
    "Claude4_Sonnet": partial(
        Claude3V,
        model="claude-4-sonnet-20250514",
        temperature=0,
        retry=10,
        verbose=False,
        timeout=1800
    ),
    
    "grok-vision-beta": partial(
        GPT4V,
        model="grok-vision-beta",
        api_base="https://api.x.ai/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "grok-2-vision-1212": partial(
        GPT4V,
        model="grok-2-vision",
        api_base="https://api.x.ai/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    # kimi
    "moonshot-v1-8k": partial(
        GPT4V,
        model="moonshot-v1-8k-vision-preview",
        api_base="https://api.moonshot.cn/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "moonshot-v1-32k": partial(
        GPT4V,
        model="moonshot-v1-32k-vision-preview",
        api_base="https://api.moonshot.cn/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
    "moonshot-v1-128k": partial(
        GPT4V,
        model="moonshot-v1-128k-vision-preview",
        api_base="https://api.moonshot.cn/v1/chat/completions",
        temperature=0,
        retry=10,
    ),
}


xtuner_series = {
    "llava-internlm2-7b": partial(
        LLaVA_XTuner,
        llm_path="internlm/internlm2-chat-7b",
        llava_path="xtuner/llava-internlm2-7b",
        visual_select_layer=-2,
        prompt_template="internlm2_chat",
    ),
    "llava-internlm2-20b": partial(
        LLaVA_XTuner,
        llm_path="internlm/internlm2-chat-20b",
        llava_path="xtuner/llava-internlm2-20b",
        visual_select_layer=-2,
        prompt_template="internlm2_chat",
    ),
    "llava-internlm-7b": partial(
        LLaVA_XTuner,
        llm_path="internlm/internlm-chat-7b",
        llava_path="xtuner/llava-internlm-7b",
        visual_select_layer=-2,
        prompt_template="internlm_chat",
    ),
    "llava-v1.5-7b-xtuner": partial(
        LLaVA_XTuner,
        llm_path="lmsys/vicuna-7b-v1.5",
        llava_path="xtuner/llava-v1.5-7b-xtuner",
        visual_select_layer=-2,
        prompt_template="vicuna",
    ),
    "llava-v1.5-13b-xtuner": partial(
        LLaVA_XTuner,
        llm_path="lmsys/vicuna-13b-v1.5",
        llava_path="xtuner/llava-v1.5-13b-xtuner",
        visual_select_layer=-2,
        prompt_template="vicuna",
    ),
    "llava-llama-3-8b": partial(
        LLaVA_XTuner,
        llm_path="xtuner/llava-llama-3-8b-v1_1",
        llava_path="xtuner/llava-llama-3-8b-v1_1",
        visual_select_layer=-2,
        prompt_template="llama3_chat",
    ),
}


llava_series = {
    "llava_v1.5_7b": partial(LLaVA, model_path="liuhaotian/llava-v1.5-7b"),
    "llava_v1.5_13b": partial(LLaVA, model_path="liuhaotian/llava-v1.5-13b"),
    "llava_v1_7b": partial(LLaVA, model_path=LLAVA_V1_7B_MODEL_PTH),
    "sharegpt4v_7b": partial(LLaVA, model_path="Lin-Chen/ShareGPT4V-7B"),
    "sharegpt4v_13b": partial(LLaVA, model_path="Lin-Chen/ShareGPT4V-13B"),
    "llava_next_vicuna_7b": partial(
        LLaVA_Next, model_path="llava-hf/llava-v1.6-vicuna-7b-hf"
    ),
    "llava_next_vicuna_13b": partial(
        LLaVA_Next, model_path="llava-hf/llava-v1.6-vicuna-13b-hf"
    ),
    "llava_next_mistral_7b": partial(
        LLaVA_Next, model_path="llava-hf/llava-v1.6-mistral-7b-hf"
    ),
    "llava_next_yi_34b": partial(LLaVA_Next, model_path="llava-hf/llava-v1.6-34b-hf"),
    "llava_next_llama3": partial(
        LLaVA_Next, model_path="llava-hf/llama3-llava-next-8b-hf"
    ),
    "llava_next_72b": partial(LLaVA_Next, model_path="llava-hf/llava-next-72b-hf"),
    "llava_next_110b": partial(LLaVA_Next, model_path="llava-hf/llava-next-110b-hf"),
    "llava_next_qwen_32b": partial(
        LLaVA_Next2, model_path="lmms-lab/llava-next-qwen-32b"
    ),
    "llava_next_interleave_7b": partial(
        LLaVA_Next, model_path="llava-hf/llava-interleave-qwen-7b-hf"
    ),
    "llava_next_interleave_7b_dpo": partial(
        LLaVA_Next, model_path="llava-hf/llava-interleave-qwen-7b-dpo-hf"
    ),
    "llava-onevision-qwen2-0.5b-ov-hf": partial(
        LLaVA_OneVision_HF, model_path="llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
    ),
    "llava-onevision-qwen2-0.5b-si-hf": partial(
        LLaVA_OneVision_HF, model_path="llava-hf/llava-onevision-qwen2-0.5b-si-hf"
    ),
    "llava-onevision-qwen2-7b-ov-hf": partial(
        LLaVA_OneVision_HF, model_path="llava-hf/llava-onevision-qwen2-7b-ov-hf"
    ),
    "llava-onevision-qwen2-7b-si-hf": partial(
        LLaVA_OneVision_HF, model_path="llava-hf/llava-onevision-qwen2-7b-si-hf"
    ),
    "llava_onevision_qwen2_0.5b_si": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-0.5b-si"
    ),
    "llava_onevision_qwen2_7b_si": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-7b-si"
    ),
    "llava_onevision_qwen2_72b_si": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-72b-si"
    ),
    "llava_onevision_qwen2_0.5b_ov": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-0.5b-ov"
    ),
    "llava_onevision_qwen2_7b_ov": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-7b-ov-chat"
    ),
    "llava_onevision_qwen2_72b_ov": partial(
        LLaVA_OneVision, model_path="lmms-lab/llava-onevision-qwen2-72b-ov-sft"
    ),
    "Aquila-VL-2B": partial(LLaVA_OneVision, model_path="BAAI/Aquila-VL-2B-llava-qwen"),
    "llava_video_qwen2_7b": partial(
        LLaVA_OneVision, model_path="lmms-lab/LLaVA-Video-7B-Qwen2"
    ),
    "llava_video_qwen2_72b": partial(
        LLaVA_OneVision, model_path="lmms-lab/LLaVA-Video-72B-Qwen2"
    ),
    "varco-vision-hf": partial(
        LLaVA_OneVision_HF, model_path="NCSOFT/VARCO-VISION-14B-HF"
    ),
}


internvl = {
    "InternVL-Chat-V1-1": partial(
        InternVLChat, model_path="OpenGVLab/InternVL-Chat-V1-1", version="V1.1"
    ),
    "InternVL-Chat-V1-2": partial(
        InternVLChat, model_path="OpenGVLab/InternVL-Chat-V1-2", version="V1.2"
    ),
    "InternVL-Chat-V1-2-Plus": partial(
        InternVLChat, model_path="OpenGVLab/InternVL-Chat-V1-2-Plus", version="V1.2"
    ),
    "InternVL-Chat-V1-5": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL-Chat-V1-5",
        version="V1.5",
    )
}

mini_internvl = {
    "Mini-InternVL-Chat-2B-V1-5": partial(
        InternVLChat, model_path="OpenGVLab/Mini-InternVL-Chat-2B-V1-5", version="V1.5"
    ),
    "Mini-InternVL-Chat-4B-V1-5": partial(
        InternVLChat, model_path="OpenGVLab/Mini-InternVL-Chat-4B-V1-5", version="V1.5"
    ),
}

internvl2 = {
    "InternVL2-1B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-1B", version="V2.0"
    ),
    "InternVL2-2B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-2B", version="V2.0"
    ),
    "InternVL2-4B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-4B", version="V2.0"
    ),
    "InternVL2-8B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-8B", version="V2.0"
    ),
    "InternVL2-26B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-26B", version="V2.0"
    ),
    "InternVL2-40B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-40B", version="V2.0"
    ),
    "InternVL2-76B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-Llama3-76B", version="V2.0"
    ),
    "InternVL2-8B-MPO": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2-8B-MPO", version="V2.0"
    ),
    "InternVL2-8B-MPO-CoT": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2-8B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
}

internvl2_5 = {
    "InternVL2_5-1B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-1B", version="V2.0"
    ),
    "InternVL2_5-2B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-2B", version="V2.0"
    ),
    "QTuneVL1-2B": partial(
        InternVLChat, model_path="hanchaow/QTuneVL1-2B", version="V2.0"
    ),
    "InternVL2_5-4B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-4B", version="V2.0"
    ),
    "InternVL2_5-8B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-8B", version="V2.0"
    ),
    "InternVL2_5-26B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-26B", version="V2.0"
    ),
    "InternVL2_5-38B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-38B", version="V2.0"
    ),
    "InternVL2_5-78B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-78B", version="V2.0"
    ),
    # InternVL2.5 series with Best-of-N evaluation
    "InternVL2_5-8B-BoN-8": partial(
        InternVLChat, model_path="OpenGVLab/InternVL2_5-8B", version="V2.0",
        best_of_n=8, reward_model_path="OpenGVLab/VisualPRM-8B",
    ),
}

internvl2_5_mpo = {
    "InternVL2_5-1B-MPO": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2_5-1B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-2B-MPO": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2_5-2B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-4B-MPO": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2_5-4B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-8B-MPO": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2_5-8B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-26B-MPO": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2_5-26B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-38B-MPO": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2_5-38B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-78B-MPO": partial(
        InternVLChat,
        model_path="OpenGVLab/InternVL2_5-78B-MPO",
        version="V2.0",
        use_mpo_prompt=True,
    ),
    "InternVL2_5-8B-GUI": partial(
        InternVLChat,
        model_path="/fs-computility/mllm1/shared/zhaoxiangyu/models/internvl2_5_8b_internlm2_5_7b_dynamic_res_stage1", 
        version="V2.0", 
        max_new_tokens=512,
        screen_parse=False,
    ),
     "InternVL3-7B-GUI": partial(
        InternVLChat,
        model_path="/fs-computility/mllm1/shared/zhaoxiangyu/GUI/checkpoints/internvl3_7b_dynamic_res_stage1_56/", 
        version="V2.0", 
        max_new_tokens=512,
        screen_parse=False,
    ),
}

internvl3 = {
    "InternVL3-1B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3-1B", version="V2.0"
    ),
    "InternVL3-2B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3-2B", version="V2.0"
    ),
    "InternVL3-8B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3-8B", version="V2.0",
    ),
    "InternVL3-9B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3-9B", version="V2.0"
    ),
    "InternVL3-14B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3-14B", version="V2.0"
    ),
    "InternVL3-38B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3-38B", version="V2.0"
    ),
    "InternVL3-78B": partial(
        InternVLChat, model_path="OpenGVLab/InternVL3-78B", version="V2.0"
    ),
}



qwen2vl_series = {
    "QVQ-72B-Preview": partial(
        Qwen2VLChat,
        model_path="Qwen/QVQ-72B-Preview",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        system_prompt="You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step.",
        max_new_tokens=8192,
        post_process=False,
    ),
    "Qwen2-VL-72B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-72B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-7B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-7B-Instruct-AWQ": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-7B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-7B-Instruct-GPTQ-Int4": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-7B-Instruct-GPTQ-Int8": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-2B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-2B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-2B-Instruct-AWQ": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-2B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-2B-Instruct-GPTQ-Int4": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2-VL-2B-Instruct-GPTQ-Int8": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "XinYuan-VL-2B-Instruct": partial(
        Qwen2VLChat,
        model_path="Cylingo/Xinyuan-VL-2B",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
    ),
    "Qwen2.5-VL-3B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-3B-Instruct-AWQ": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-7B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-7B-Instruct-ForVideo": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-7B-Instruct",
        min_pixels=128 * 28 * 28,
        max_pixels=768 * 28 * 28,
        total_pixels=24576 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-7B-Instruct-AWQ": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-32B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-32B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-72B-Instruct": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-72B-Instruct",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "MiMo-VL-7B-SFT": partial(
        Qwen2VLChat,
        model_path="XiaomiMiMo/MiMo-VL-7B-SFT",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "MiMo-VL-7B-RL": partial(
        Qwen2VLChat,
        model_path="XiaomiMiMo/MiMo-VL-7B-RL",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-72B-Instruct-ForVideo": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-72B-Instruct",
        min_pixels=128 * 28 * 28,
        max_pixels=768 * 28 * 28,
        total_pixels=24576 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-VL-72B-Instruct-AWQ": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
    "Qwen2.5-Omni-7B-ForVideo": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-Omni-7B",
        min_pixels=128 * 28 * 28,
        max_pixels=768 * 28 * 28,
        total_pixels=24576 * 28 * 28,
        use_custom_prompt=False,
        use_audio_in_video=True, # set use audio in video
    ),
    "Qwen2.5-Omni-7B": partial(
        Qwen2VLChat,
        model_path="Qwen/Qwen2.5-Omni-7B",
        min_pixels=1280 * 28 * 28,
        max_pixels=16384 * 28 * 28,
        use_custom_prompt=False,
    ),
}



internvl_groups = [
    internvl, internvl2, internvl2_5, mini_internvl, internvl2_5_mpo, 
    internvl3,
]
internvl_series = {}
for group in internvl_groups:
    internvl_series.update(group)

supported_VLM = {}

model_groups = [
     o1_apis, api_models, xtuner_series,  llava_series, 
    internvl_series, qwen2vl_series,
]

for grp in model_groups:
    supported_VLM.update(grp)