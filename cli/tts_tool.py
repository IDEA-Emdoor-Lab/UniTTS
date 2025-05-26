
import sys
import os
import traceback
from tqdm import tqdm
import soundfile as sf
import librosa
import argparse
from vllm import LLM, SamplingParams

tts_prompt_ref_text = """<|im_start|>system
### 你输出的音色要求：
当处于必须进行语音回复的情形下，你应该以如下示例音频的声音种类来予以回复，示例音频的声音如下：
示例音频对应的文字:{example_text}
示例音频：{example_voice}

### 你的输出模式：
你的输出模式：【语音或者音频】
### 用户输入模式：
【文本】是用户当前的输入模式。<|im_end|><|endoftext|><|im_start|>user
你要把【{content}】这句话转为语音。<|im_end|><|endoftext|><|im_start|>assistant
"""

long_cot_prompt_template = """<|im_start|>system
你是一个人工智能助手，你在回答用户问题时候需要根据输出模式回答。
如果用户指定输出模式为深度思考，先生成思考步骤，再回答问题，生成的深度思考内容需要放在<|cot_begin|>和<|cot_end|>中，再生成最终答案。
如果用户指定输出模式为直接回答答案，则直接回答用户问题，不需要生成思考过程。


### 你的输出模式：
【深度思考】

<|im_end|><|endoftext|><|im_start|>user
{question} <|im_end|><|endoftext|><|im_start|>assistant
"""

text_conversation_prompt_template = """<|im_start|>system
你是一位有帮助的人工智能助手，擅长回答用户问题。

### 你的输出模式：
你的输出模式：【文本】

### 用户输入模式：
【文本】是用户当前的输入模式。
<|im_end|><|endoftext|><|im_start|>user
{question}<|im_end|><|endoftext|><|im_start|>assistant
"""
 
def post_result(audio_token_list):
    """_summary_

    Args:
        audio_token_dict (_type_): _description_
    """
    audio_tokens = []

    for token_dict in audio_token_list:
        audio_tokens.append(token_dict['absolute_token_id'])
    if len(audio_tokens) == len(audio_token_list):
        return audio_tokens
    else:
        return []
    
def enocde_audio( codec, tokenizer, audio_data):
    # print(f'audio_data: {audio_data}', flush=True)
    audio_data, samplerate = sf.read(audio_data)
    # 目标采样率
    target_samplerate = 24000
    # 使用 librosa 进行重采样
    audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=target_samplerate)
    # 转换为单声道
    if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
    # print(f'audio_data.shape: {audio_data.shape}', flush=True)
    audio_tokens = codec.encode([[audio_data, samplerate]], enable_bfloat16=True, raw_audio=True)
    # print(f'audio_tokens: {audio_tokens}', flush=True)
    codes_list = audio_tokens[0].codes_list
    audio_datas = post_result(codes_list[0])

    audio_text = tokenizer.tokenizer.decode(audio_datas)
    audio_text = f'<|inter_audio_begin|>{audio_text}<|inter_audio_end|>'

    return audio_text

def format_prompt(codec, tokenizer, ref_text, ref_audio_path, text):      

    prompt_text = ref_text
    ref_audio_text = enocde_audio(codec, tokenizer, ref_audio_path)
    ref_audio_text = f'<|inter_audio_begin|>{ref_audio_text}<|inter_audio_end|>'
    prompt = tts_prompt_ref_text.format(content=text, example_voice=ref_audio_text, example_text=prompt_text)
    return prompt

def format_long_cot_prompt(prompt_text):      
    prompt = long_cot_prompt_template.format(question=prompt_text)
    return prompt

def format_text_prompt(prompt_text):      
    prompt = long_cot_prompt_template.format(question=prompt_text)
    return prompt

