import sys

from fastapi import FastAPI
import gradio as gr
import uvicorn
from fastapi import Body
import numpy as np

from argparse import ArgumentParser
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

import torch
import soundfile as sf
import librosa

import sys
import os
import uuid
import re

from vllm import LLM, SamplingParams

from prompt_template import general_system, dialog_role_system, dialog_user_system, general_user_input

sys.path.append('/cognitive_comp/sunqianguo/workspace/audio_codec')
from aipal_codec import AIPalCodec, decode_codec_and_save_audio # type: ignore

# 自定义 StoppingCriteria，用于多个停止 ID
class MultiTokenStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_ids):
        super().__init__()
        self.stop_ids = stop_ids  # 停止的 token ID 列表

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        # 检查最后一个生成的 token 是否在 stop_ids 中
        return input_ids[0, -1].item() in self.stop_ids

class AudioWebDemo(object):
    """前端页面支持的功能如下：
            1. 支持输入音频文件，并生成音频文件
            2. 支持输入文本，并生成文本
            3. 支持输入文本和音频文件，并生成音频文件

    Args:
        object (_type_): _description_
    """
    def __init__(self, model_path, args) -> None:
        self.args = args
        
        # 加载模型和分词器
        if model_path is not None:
            self.device = torch.device(f'cuda:{0}')
            #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # {"": "cuda:0"}
            print(f'load model_path from: {model_path}')
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            # <|beginofaudio|>
            self.beginofaudio = self.tokenizer.encode('<|beginofaudio|>', add_special_tokens=False, padding='do_not_pad', return_tensors='pt').numpy().tolist()[0][0] # <|beginofaudio|>
            self.endofaudio = self.tokenizer.encode('<|endofaudio|>', add_special_tokens=False, padding='do_not_pad', return_tensors='pt').numpy().tolist()[0][0] # <|beginofaudio|>

            self.inter_audio_begin = self.tokenizer.encode('<|inter_audio_begin|>', add_special_tokens=False, padding='do_not_pad', return_tensors='pt').numpy().tolist()[0][0] # <|beginofaudio|>
            self.inter_end_audio = self.tokenizer.encode('<|inter_audio_end|>', add_special_tokens=False, padding='do_not_pad', return_tensors='pt').numpy().tolist()[0][0]

            self.endoftext_token = self.tokenizer.encode('<|endoftext|>', add_special_tokens=False, padding='do_not_pad', return_tensors='pt').numpy().tolist()[0][0]
            
            #  device_map="auto"
            if args.use_vllm:
                self.model = LLM(model=model_path, dtype='bfloat16', gpu_memory_utilization=0.8, device=self.device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_path).bfloat16().cuda() #.to("cuda:1") 
                self.model.eval()
            print(f'self.beginofaudio is:{self.beginofaudio}', flush=True)
            print(f'self.endofaudio is:{self.endofaudio}', flush=True)
            print(f'self.inter_audio_begin is:{self.inter_audio_begin}', flush=True)
            print(f'self.inter_end_audio is:{self.inter_end_audio}', flush=True)

            # 定义停止 token ID 列表
            stop_tokens = ["<|endoftext|>"]
            self.stop_ids = self.tokenizer.convert_tokens_to_ids(stop_tokens)

            # 创建 StoppingCriteriaList
            self.stopping_criteria = StoppingCriteriaList([MultiTokenStoppingCriteria(self.stop_ids)])
            
        
        if args.model_config is not None and args.ckpt_config is not None:
            print(f'load ckpt_config: {self.args.ckpt_config}')
            self.codec = AIPalCodec.from_pretrained(config_path=self.args.model_config,
                                        model_path=self.args.ckpt_config,
                                        use_generator=True,
                                        is_debug=False, local_rank=0).eval()

        self.offset = 152064
        self.endofaudio_index = 184832
        self.save_dir_curr = './log/webdemo/' #'/cognitive_comp/sunqianguo/tmp/audio/webdemo/'
        if not os.path.exists(self.save_dir_curr):
            os.makedirs(self.save_dir_curr, exist_ok=True)

        self.target_samplerate = 24000

        self.general_audio_index = 1
        self.user_audio_index = 1
        self.role_audio_index = 1

    def clear_fn(self):
        self.general_audio_index = 1
        self.user_audio_index = 1
        self.role_audio_index = 1
        return [], [], '', '', '', None, None
    
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

    def enocde_audio(self, audio_data):
        # print(f'audio_data: {audio_data}', flush=True)
        audio_data, samplerate = sf.read(audio_data)

        
        # 转换为单声道
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=self.target_samplerate)
        # print(f'audio_data.shape: {audio_data.shape}', flush=True)
        audio_tokens = self.codec.encode([[audio_data, self.target_samplerate]], enable_bfloat16=True, raw_audio=True)
        # print(f'audio_tokens: {audio_tokens}', flush=True)
        codes_list = audio_tokens[0].codes_list
        # print('=' * 60)
        # print(f'codes_list: {codes_list}', flush=True)
        audio_datas = self.post_result(codes_list[0])
        
        audio_text  =self.tokenizer.decode(audio_datas)

        return audio_text
    
    def format_system(self, system_prompt: str,
            task_mode,
            audio1,
            user_prompt,
            user_audio1,
            role_prompt,
            role_audio1):
        if task_mode == "general":
            system_str = system_prompt
            '''
            if audio1:
                audio_text1 = self.enocde_audio(audio1)
                system_str = system_str.replace('<audio1>', '<|inter_audio_begin|>' + audio_text1 + '<|inter_audio_end|>')
            
            if audio2:
                audio_text2 = self.enocde_audio(audio2)
                system_str = system_str.replace('<audio2>', '<|inter_audio_begin|>' + audio_text2 + '<|inter_audio_end|>')
            '''
            
            return system_str
        
        elif task_mode == "role_dialog":
            system_str = ''
            system_str += user_prompt

            system_str += role_prompt

            return system_str
    
    def task_mode_update_interface(self, task_mode):
        if task_mode == "general":
            # 当 task_mode 是 "general" 时，显示 general tab，隐藏 role_dialog tab
            return gr.Tab.update(visible=True), gr.Tab.update(visible=False)
        elif task_mode == "role_dialog":
            # 当 task_mode 是 "role_dialog" 时，显示 role_dialog tab，隐藏 general tab
            return gr.Tab.update(visible=False), gr.Tab.update(visible=True)
        
    def replace_inter_audio_content(self, text, replacement=""):
        """
        删除 <|inter_audio_begin|> 和 <|inter_audio_end|> 之间的内容，并替换为指定内容。

        :param text: 原始字符串
        :param replacement: 替换的内容，默认为空字符串
        :return: 处理后的字符串
        """
        # 正则表达式匹配 <|inter_audio_begin|> 和 <|inter_audio_end|> 之间的内容
        pattern = r'<\|inter_audio_begin\|>.*?<\|inter_audio_end\|>'
        # 使用 re.sub 替换匹配的内容
        result = re.sub(pattern, replacement, text, flags=re.DOTALL)
        return result
    def update_system_by_audio(self, audio_path: str | None, system_data):
        audio_index = 0
        audio_index = self.general_audio_index

        if f'<audio{audio_index}>' not in system_data  and '<|inter_audio_begin|>' in system_data:
            audio_text = self.enocde_audio(audio_path)
            system_data = self.replace_inter_audio_content(system_data, '<|inter_audio_begin|>' + audio_text + '<|inter_audio_end|>')
            return system_data

        if f'<audio{audio_index}>' not in system_data or audio_path is None:
            return system_data
    
        audio_text = self.enocde_audio(audio_path)
        system_data = system_data.replace(f'<audio{audio_index}>', '<|inter_audio_begin|>' + audio_text + '<|inter_audio_end|>')
        #self.general_audio_index += 1

        return system_data
    
    def update_system_by_user_audio(self, audio_path: str | None, system_data):
        audio_index = 0    
        audio_index = self.user_audio_index

        if f'<audio{audio_index}>' not in system_data  and '<|inter_audio_begin|>' in system_data:
            audio_text = self.enocde_audio(audio_path)
            system_data = self.replace_inter_audio_content(system_data, '<|inter_audio_begin|>' + audio_text + '<|inter_audio_end|>')
            return system_data


        if f'<user_audio{audio_index}>' not in system_data or audio_path is None:
            return system_data
        audio_text = self.enocde_audio(audio_path)
        system_data = system_data.replace(f'<user_audio{audio_index}>', '<|inter_audio_begin|>' + audio_text + '<|inter_audio_end|>')
        #self.user_audio_index += 1

        return system_data
    
    def update_system_by_role_audio(self, audio_path: str | None, system_data):
        audio_index = 0    
        audio_index = self.role_audio_index

        if f'<audio{audio_index}>' not in system_data  and '<|inter_audio_begin|>' in system_data:
            audio_text = self.enocde_audio(audio_path)
            system_data = self.replace_inter_audio_content(system_data, '<|inter_audio_begin|>' + audio_text + '<|inter_audio_end|>')
            return system_data

        if f'<role_audio{audio_index}>' not in system_data or audio_path is None:
            return system_data
        audio_text = self.enocde_audio(audio_path)
        system_data = system_data.replace(f'<role_audio{audio_index}>', '<|inter_audio_begin|>' + audio_text + '<|inter_audio_end|>')
        #self.role_audio_index += 1

        return system_data
    
    def decode_audio_tokens(self, audio_tokens):
        # audio_ids = [int(num) for num in audio_tokens.replace("\n", "").replace(" ", "").split(",")]
        audio_ids = self.tokenizer.encode(audio_tokens.replace("\n", "").replace(" ", ""))

        uuid_str = str(uuid.uuid4())

        save_path = os.path.join(self.save_dir_curr, f'{uuid_str}.wav')
        print(f"concat save_path is:{save_path}", flush=True)
        if len(audio_ids) > 0:
            decode_codec_and_save_audio([audio_ids], self.save_dir_curr, f'{uuid_str}', self.codec)
            # decode_codec_and_save_audio([audio_ids], self.save_dir_curr, f'{uuid_str}')
        
        return save_path
    
    def encode_audio_tokens(self, audio_path):
        audio_data, samplerate = sf.read(audio_path)

        
        # 转换为单声道
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        audio_data = librosa.resample(audio_data, orig_sr=samplerate, target_sr=self.target_samplerate)
        # print(f'audio_data.shape: {audio_data.shape}', flush=True)
        audio_tokens = self.codec.encode([[audio_data, self.target_samplerate]], enable_bfloat16=True, raw_audio=True)
        # print(f'audio_tokens: {audio_tokens}', flush=True)
        codes_list = audio_tokens[0].codes_list
        # print('=' * 60)
        # print(f'codes_list: {codes_list}', flush=True)
        audio_datas = self.post_result(codes_list[0])
        
        return audio_datas
    
    def inference_fn(self, 
            temperature: float,
            top_p: float,
            max_new_token: int,
            audio_path: str | None,
            input_text: str | None,
            history: list[dict],
            previous_input_tokens: str,
            previous_completion_tokens: str,
            system_prompt: str,
            task_mode,
            audio1,
            user_prompt,
            user_audio1,
            role_prompt,
            role_audio1,
    ):

        if audio_path is not None and len(input_text) == 0:
            print('enter audio mode.....', flush=True)
            assert audio_path is not None
            history.append({"role": "user", "content": {"path": audio_path}})
            # 将音频转成tokenizer之后转成text
            audio_text = self.enocde_audio(audio_path)
            user_input = '<|beginofaudio|>' + audio_text  + '<|endofaudio|>'
            #user_input = '<|inter_audio_begin|>' + audio_text  + '<|inter_audio_end|>'

        elif audio_path is not None and input_text is not None and len(input_text) > 0:
            assert audio_path is not None
            assert input_text is not None
            print('enter audio and text mode.....', flush=True)
            # 将音频的预处理
            history.append({"role": "user", "content": input_text})
            history.append({"role": "user", "content": {"path": audio_path}})

            audio_text = self.enocde_audio(audio_path)
            audio_text = '<|inter_audio_begin|>' + audio_text  + '<|inter_audio_end|>'
            
            input_text = input_text.replace('<audio>', audio_text)
            user_input = input_text
            
        else:
            print('enter text mode.....', flush=True)
            assert input_text is not None
            user_input = input_text
            history.append({"role": "user", "content": input_text})
        
            
        # Gather history
        inputs = previous_input_tokens + previous_completion_tokens
        inputs = inputs.strip()

        if "<|system|>" not in inputs:
            system_str = self.format_system(system_prompt, task_mode, audio1, user_prompt, user_audio1, role_prompt, role_audio1)
            inputs += f"<|system|>\n{system_str}<|endoftext|>\n"

        # 推理结果，并将结果返回给前端页面
        inputs += f"<|im_start|>user\n{user_input}<|im_end|><|endoftext|>\n<|im_start|>assistant\n"
        print(f'inputs is: {inputs}', flush=True)
        
        sample_ids = self.tokenizer.encode(inputs)
        input_ids = torch.tensor(sample_ids).to(self.device)
        input_ids = torch.unsqueeze(input_ids, dim=0)
        
        # '''
        # 生成续写
        if args.use_vllm:
            
            sampling_params = SamplingParams(temperature=temperature, top_p=top_p, stop_token_ids=self.stop_ids, max_tokens=max_new_token)
            output = self.model.generate([inputs], sampling_params)
            # print(output, flush=True)

            #tokens = self.tokenizer.tokenizer.encode(output[0].outputs[0].token_ids)[1: -2]
            output_ids = output[0].outputs[0].token_ids

        else:
            output = self.model.generate(input_ids, max_length=max_new_token,  do_sample = True, 
                top_p = top_p, 
                temperature = temperature, num_return_sequences=1,
                stopping_criteria=self.stopping_criteria)
        
            # 解码输出 只保留下来新生成的音频dialog
            output_ids = output[0][len(sample_ids):].cpu().numpy()
        total_generated_text = self.tokenizer.decode(output_ids)
        print(f'total_generated_text output is:{total_generated_text}', flush=True)
        
        # 音频和文本token分开进行保存，展示的时候，分别展开文本和音频方便用户去管理
        text_tokens = []
        for t in output_ids:
            if t < self.offset:
                text_tokens.append(t)
        generated_text = self.tokenizer.decode(text_tokens)
        '''
        if self.inter_audio_begin in output_ids:
            indices = np.where(output_ids == self.inter_audio_begin)

            # 过滤出来所有的音频token，将音频token组合起来
            audio_tokens = []

            for id, i in enumerate(indices[0]):
                output_ids_n = output_ids[i+1:]
                # output_ids_n = output_ids[indices[0][0]+1:]
                output_ids_decode = output_ids_n
                if self.inter_end_audio in output_ids_n:
                    indices_end = np.where(output_ids_n == self.inter_end_audio)
                    output_ids_decode = output_ids_n[:indices_end[0][0]]
                    audio_tokens.extend(output_ids_decode)

            # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            uuid = str(uuid.uuid4())
            decode_codec_and_save_audio([audio_tokens], self.save_dir_curr, f"{uuid}")
        '''
        audio_ids = []
        for t in output_ids:
            if t >= self.offset and t < self.endofaudio_index:
                audio_ids.append(t)
        uuid_str = str(uuid.uuid4())
        print(f'audio_ids is:{audio_ids}', flush=True)

        audio_text  =self.tokenizer.decode(audio_ids)
        print(f'audio_text is:{audio_text}', flush=True)
        save_path = os.path.join(self.save_dir_curr, f'{uuid_str}.wav')
        print(f"concat save_path is:{save_path}", flush=True)
        if len(audio_ids) > 0:
            decode_codec_and_save_audio([audio_ids], self.save_dir_curr, f'{uuid_str}')
            history.append({"role": "assistant", "content": {"path": save_path, "type": "audio/wav"}})

        history.append({"role": "assistant", "content": generated_text.replace('<|im_end|><|endoftext|>', '')})
        print(f"save_path is:{save_path}", flush=True)

        
        compelted_text = total_generated_text if task_mode == "general" else generated_text.strip('<|im_end|>').strip('<|endoftext|>')
        print(f'compelted_text is:{compelted_text}', flush=True)
        print(f'task_mode is:{task_mode}', flush=True)

        return history, inputs, compelted_text, '', None, save_path

    def create_demo(self):
        # Create the Gradio interface
        with gr.Blocks(title="UniTTS Inference WebDemo", fill_height=True) as demo:
            # 添加一个 Markdown 组件来显示居中的大标题
            gr.Markdown(
                """
                <div style="text-align: center;">
                    <h1>UniTTS Inference WebDemo</h1>
                </div>
                """
            )
            with gr.Row():
                with gr.Column(scale=1):
                    temperature = gr.Number(
                        label="Temperature",
                        value=0.9
                    )

                    top_p = gr.Number(
                        label="Top p",
                        value=0.9
                    )

                    max_new_token = gr.Number(
                        label="Max new tokens",
                        value=6000,
                    )
                    

                    task_mode = gr.Radio(["general", "role_dialog"], label="Task Mode", value="general")
                    with gr.Column(scale=2):
                        with gr.Tab('general') as general_tab:
                            system_prompt = gr.Textbox(label="system",
                                            value=general_system,
                                            lines=20)
                            audio1 = gr.Audio(label="Input audio", type='filepath', show_download_button=True, visible=True)
                            # audio2 = gr.Audio(label="Input audio", type='filepath', show_download_button=True, visible=True)
                        
                        with gr.Tab('role_dialog') as role_dialog_tab:
                            user_prompt = gr.Textbox(label="user system",
                                            value=dialog_user_system,
                                            lines=20)
                            user_audio1 = gr.Audio(label="Input audio1", type='filepath', show_download_button=True, visible=True)
                            #user_audio2 = gr.Audio(label="Input audio2", type='filepath', show_download_button=True, visible=True)
                            role_prompt = gr.Textbox(label="model system",
                                            value=dialog_role_system,
                                            lines=20)
                            role_audio1 = gr.Audio(label="role Input audio1", type='filepath', show_download_button=True, visible=True)
                            #role_audio2 = gr.Audio(label="role Input audio2", type='filepath', show_download_button=True, visible=True)

                    #system_prompt = gr.Textbox(label="system：",
                    #                              value='system',
                    #                              lines=6,
                    #                              interactive=True)

                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        elem_id="chatbot",
                        bubble_full_width=False,
                        type="messages",
                        scale=1,
                    )

                    #input_mode = gr.Radio(["audio", "text", "text_audio"], label="Input Mode", value="text")

                    text_input = gr.Textbox(label="Input text", value=general_user_input, lines=1, visible=True)
                    audio = gr.Audio(label="Input audio", type='filepath', show_download_button=True, visible=True)

                    submit_btn = gr.Button("Submit")
                    reset_btn = gr.Button("Clear")
                    output_audio = gr.Audio(label="Play", streaming=True,
                                            autoplay=True, show_download_button=False)
                    complete_audio = gr.Audio(label="Last Output Audio (If Any)", type="filepath", show_download_button=True)


            gr.Markdown("""## Debug Info""")
            with gr.Row():
                input_tokens = gr.Textbox(
                    label=f"Input Tokens",
                    interactive=False,
                )

                completion_tokens = gr.Textbox(
                    label=f"Completion Tokens",
                    interactive=False,
                )

            detailed_error = gr.Textbox(
                label=f"Detailed Error",
                interactive=False,
            )
            gr.Markdown("""## Tools""")
            with gr.Row():
                decode_audio = gr.Audio(label="deocdeaudio", type='filepath', show_download_button=True, visible=True)
                decode_audio_tokens = gr.Textbox(
                    label=f"Input Tokens"
                )
                decode_btn = gr.Button("decode button")
            
            with gr.Row():
                enocde_audio = gr.Audio(label="Input audio", type='filepath', show_download_button=True, visible=True)
                enocde_audio_tokens = gr.Textbox(
                    label=f"Input Tokens"
                )
                encode_btn = gr.Button("encode button")

            history_state = gr.State([])

            respond = submit_btn.click(
                self.inference_fn,
                inputs=[
                    temperature,
                    top_p,
                    max_new_token,
                    audio,
                    text_input,
                    history_state,
                    input_tokens,
                    completion_tokens,
                    system_prompt,
                    task_mode,
                    audio1,
                    user_prompt,
                    user_audio1,
                    role_prompt,
                    role_audio1,

                ],
                outputs=[history_state, input_tokens, completion_tokens, detailed_error, output_audio, complete_audio]
            )

            respond.then(lambda s: s, [history_state], chatbot)

            reset_btn.click(self.clear_fn, outputs=[chatbot, history_state, input_tokens, completion_tokens, detailed_error, output_audio, complete_audio])
            # input_mode.input(self.clear_fn, outputs=[chatbot, history_state, input_tokens, completion_tokens, detailed_error, output_audio, complete_audio]) #then(update_input_interface, inputs=[input_mode], outputs=[audio, text_input])
            task_mode.change(self.clear_fn, outputs=[chatbot, history_state, input_tokens, completion_tokens, detailed_error, output_audio, complete_audio]) #.then(self.task_mode_update_interface, inputs=[task_mode], outputs=[general_tab, role_dialog_tab])
            
            audio1.input(self.update_system_by_audio, inputs=[audio1, system_prompt], outputs=[system_prompt])
            user_audio1.input(self.update_system_by_user_audio, inputs=[user_audio1, user_prompt], outputs=[user_prompt])
            role_audio1.input(self.update_system_by_role_audio, inputs=[role_audio1, role_prompt], outputs=[role_prompt])

            decode_btn.click(
                self.decode_audio_tokens,
                inputs=[decode_audio_tokens],
                outputs=[decode_audio]
            )
            encode_btn.click(
                self.encode_audio_tokens,
                inputs=[enocde_audio],
                outputs=[enocde_audio_tokens]
            )
            return demo.queue()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default="8888")
    parser.add_argument("--demo_url", type=str)
    parser.add_argument("--api_url", type=str)

    parser.add_argument("--audio_model_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--ckpt_config", type=str)
    parser.add_argument("--use_vllm", type=bool, default=False)

    args = parser.parse_args()

    app = FastAPI()


    pmt_mode = None
    mind_bot = AudioWebDemo(args.model_path, args)


    
    mind_bot.create_demo()
    demo = mind_bot.create_demo()

    app = gr.mount_gradio_app(app, demo, path=args.demo_url)

    # ssl_certfile
    # ssl_keyfile
    
    #uvicorn.run(app, host=args.host, port=args.port)
    uvicorn.run(app, host=args.host, port=args.port,
                ssl_certfile="/cognitive_comp/sunqianguo/workspace/GLM-4-Voice/ssl/cert.pem", ssl_keyfile="/cognitive_comp/sunqianguo/workspace/GLM-4-Voice/ssl/key.pem")



    
  
    
