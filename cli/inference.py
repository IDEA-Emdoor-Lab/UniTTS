import sys
import os
import traceback
from tqdm import tqdm
from tokenizer import QWenTokenizer
import soundfile as sf
import librosa
import argparse
from vllm import LLM, SamplingParams

sys.path.append('../../DistilCodec/')
from distil_codec import DistilCodec # type: ignore

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
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_dir", type=str, default='')
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt_config", type=str, default='')
    parser.add_argument("--model_config", type=str, default='')
    parser.add_argument("--model_name", type=str, default='')
    parser.add_argument("--inference_type", type=str, default='')
    parser.add_argument('--text', type=str, help='text for tts inference')
    parser.add_argument('--ref_text', type=str, help='text of ref audio')
    parser.add_argument('--ref_audio_path', type=str, help='text of ref audio')
    

    args = parser.parse_args()
    return args

class InferenceClient:
    def __init__(self, args) -> None:
        self.args = args    
    def post_result(self, audio_token_list):
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
        
    def enocde_audio(self, codec, tokenizer, audio_data):
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
        audio_datas = self.post_result(codes_list[0])

        audio_text = tokenizer.tokenizer.decode(audio_datas)
        audio_text = f'<|inter_audio_begin|>{audio_text}<|inter_audio_end|>'

        return audio_text

    def format_prompt(self, codec, tokenizer):      

        prompt_text = self.args.ref_text
        ref_audio_text = self.enocde_audio(codec, tokenizer, self.args.ref_audio_path)
        ref_audio_text = f'<|inter_audio_begin|>{ref_audio_text}<|inter_audio_end|>'
        prompt = tts_prompt_ref_text.format(content=self.args.text, example_voice=ref_audio_text, example_text=prompt_text)
        return prompt
    
    def load_model_and_codec(self):
        llm = LLM(model=self.args.model_name, dtype='auto', gpu_memory_utilization=0.8, seed=self.args.seed) # , seed=42    

        codec = DistilCodec.from_pretrained(
            config_path=self.args.model_config,
            model_path=self.args.ckpt_config,
            use_generator=True,
            is_debug=False,
            local_rank=0).eval()
        
        return llm, codec


    def inference_data(self):
        tokenizer: QWenTokenizer = QWenTokenizer(self.args.model_name)
        stop_tokens = ["<|endoftext|>"]
        stop_ids = tokenizer.tokenizer.convert_tokens_to_ids(stop_tokens)
        print(f'Stop IDs: {stop_ids}')
        # 定义停止 token ID 列表
        stop_tokens = ["<|endoftext|>", "<|endofaudio|>", "<|im_end|>"]
        stop_ids = tokenizer.tokenizer.convert_tokens_to_ids(stop_tokens)


        # 初始化model和codec
        llm, codec = self.load_model_and_codec()
        sampling_params = SamplingParams(temperature=self.args.temperature, top_p=self.args.top_p, stop_token_ids=stop_ids, max_tokens=6000)
    

        if not os.path.exists(self.args.output_dir):
            os.mkdir(self.args.output_dir)
        
        prompt = self.format_prompt(codec, tokenizer)

        output = llm.generate([prompt], sampling_params)
        tokens = tokenizer.tokenizer.encode(output[0].outputs[0].text)[1: -2]
        utt = 'infer'
        #decode_codec_and_save_audio([tokens], self.args.output_dir, utt, codec) 
        y_gen = codec.decode_from_codes(
            tokens, 
            minus_token_offset=True # if the 'plus_llm_offset' of method demo_for_generate_audio_codes is set to True, then minus_token_offset must be True.
        )
        codec.save_wav(
            audio_gen_batch=y_gen, 
            nhop_lengths=[y_gen.shape[-1]], 
            save_path=self.args.output_dir,
            name_tag=utt
        )
def main():
    args = get_args()
    print(args)
    
    client = InferenceClient(args)
    client.inference_data()

if __name__ == '__main__':
    # convert_token2audio()
    main()
