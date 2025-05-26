<div align="center">
    <h1>
    UniTTS
    </h1>
    <p>
    <b><em>UniTTS: An end-to-end TTS system without decoupling of acoustic and semantic information</em></b>
   </p>
    <p>
    </p>
    </p>
    <a href="https" style="color:red">Paper </a> |  
    <a href="https://huggingface.co/IDEA-Emdoor/UniTTS-mixed-v0.1" style="color:#FFD700">Hugging Face Model</a>
    <a href="https://github.com/IDEA-Emdoor-Lab/UniTTS?tab=readme-ov-file" style="color:gray">Code</a>
     <p>
        <img src="figures/idea_capture.png" alt="Institution 1" style="width: 200px; height: 60px;">
    </p>
     <p>
        <img src="figures/yidao_logo.png" alt="Institution 2" style="width: 200px; height: 60px;">
        <img src="figures/yijiayiban.png" alt="Institution 3" style="width: 200px; height: 60px;">
    </p>
</div>


## UniTTS

### Overview

we introduce [DistilCodec](https://github.com/IDEA-Emdoor-Lab/DistilCodec/tree/dev?tab=readme-ov-file) and UniTTS. DistilCodec is a single-codebook audio codec, which has 32768 codes, and the utilization of the codebook achieves nearly 100\%. UniTTS leverages DistilCodec for audio discretization, while its backbone network adopts Qwen2.5-7B to model relationships between audio tokens. 

Our main contributions are summarized as follows:

  - DistilCodec: We propose a training methodology that enables the distillation of multi-codebook NAC into single-codebook NAC. Through this approach, we have developed DistilCodec - a single-codebook NAC containing 32,768 codes that achieves 100\% utilization with balanced code distribution. Notably, DistilCodec employs universal audio data for training rather than being restricted to speech-specific datasets.
  - UniTTS: We present UniTTS, a novel TTS system trained on QWen2.5-7B and DistilCodec. Leveraging DistilCodec's comprehensive audio modeling capabilities, UniTTS achieves end-to-end speech synthesis with full-spectrum audio input/output. The system demonstrates enhanced naturalness in emotional expressiveness compared to conventional TTS systems, particularly in capturing subtle prosodic variations and affective nuances during audio generation. 
  - Novel Audio Language Model Paradigm: We establish a dual-phase Audio Language Model (ALM) training framework, which comprises (i) Audio Perceptual Modeling (DistilCodec) focusing purely on acoustic discretization, and (ii) Audio Cognitive Modeling (UniTTS) implemented via pretraining (incorporating universal audio autoregressive tasks), supervised fine-tuning (evaluating text-audio interleaved prompts' impact), and alignment (employing direct preference optimization for speech refinement) - enabled by UniTTS's complete end-to-end integration within the LLM.

The architecture of UniTTS is illustrated in the figure below. ![UniTTS](./figures/figure_1.jpg).

# Training data distribution and application scope
The model architecture was augmented with cross-lingual text-speech paired datasets (English and Chinese) alongside text-associated instruction corpora during pretraining. Subsequent SFT and alignment phases systematically incorporated three datasets: text instructions dataset, long-CoT dataset, and Chinese TTS dataset. Consequently, the model demonstrates robust capabilities in text-based conversational, long-CoT conversational, and Chinese TTS.

The distribution of the sft training data is as follows:

| Data Type                  | Number of Samples |
|----------------------------|-------------------|
| Text Data                  | 181K              |
| Long-cot Dataset           | 55K               |
| Chinese Text-Audio Alignment Data  | 401K              |
| Total                      | 637K              |

The distribution of the lpo training data is as follows:

| Data Type                  | Number of Samples |
|----------------------------|-------------------|
| General SFT Data           | 100K              |
| Long-cot Dataset           | 45K               |
| Chinese Text-Audio Alignment Data  | 300K              |
| Total                      | 445K              |

The proposed model supports the following capabilities

|     Application Type       | Support Status    |
|----------------------------|-------------------|
| Text conversation          | Supported           |
| Long-cot conversation      | Supported           |
| Chinese TTS                | Supported           |


## Install
**Clone and Install**

- Clone the repo
``` sh
git clone git@github.com:IDEA-Emdoor-Lab/UniTTS.git

git clone git@github.com:IDEA-Emdoor-Lab/DistilCodec.git

cd UniTTS
```

- Installation environment
``` sh
conda create -n unitts -y python=3.10
conda activate unitts
pip install -r requirements.txt
```



**Model Download**

Download via git clone:
```sh
mkdir -p pretrained_models

# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

# clone UniTTS model
git clone git@hf.co:IDEA-Emdoor/UniTTS-mixed-v0.1

# clone DistilCodec model
git clone git@hf.co:IDEA-Emdoor/DistilCodec-v1.0
```

**Training Usage**

We have open-sourced our three-stage training code, including pre-training, SFT, and LPO. Our training code is built upon the pai-megatron-patch with optimizations. For usage instructions on pre-training and SFT training, please refer to the following [README](pai-megatron-patch/examples/qwen2_5/README.md).


**Inference Usage**

Direct inference can be executed with the following script
``` sh
cd cli
sh run_evalation.sh
```
Or you can also run it directly using the following Python command
```
python inference.py \
    --model_config $model_config \
    --ckpt_config $ckpt_config \
    --model_name $model_path \
    --output_dir $output_dir \
    --temperature $temperature \
    --top_p $top_p \
    --seed $seed \
    --text $text \
    --ref_text $ref_text \
    --ref_audio_path $ref_audio_path \
```

## **Demos**

Our model can generate audio that maintains the timbre of the reference audio while producing emotionally expressive output tailored to the context of the target sample. Here are some demos generated by UniTTS. 

| Ref Audio | Inference Text | Inference Audio |
|-----------|----------------|-----------------|
| [system_audio.wav](./demos/voice0/system_audio.wav) | 求求你…不要离开我，我真的好害怕… | [infer.wav](./demos/voice0/infer_0.wav) |
| [system_audio.wav](./demos/voice1/system_audio.wav) | 天啊！这竟然是真的？我简直不敢相信！ | [infer.wav](./demos/voice1/infer_1_1.wav) |
| [system_audio.wav](./demos/voice2/system_audio.wav) | 立刻停止你的行为！这是最后的警告！ | [infer.wav](./demos/voice2/infer_2_1.wav) |
| [voice3/system_audio.wav](./demos/voice3/system_audio.wav) | 天啊！这绝对是我见过最不可思议的画面！ | [infer.wav](./demos/voice3/infer_3_1.wav) |
| [system_audio.wav](./demos/voice4/system_audio.wav) | 你怎么能这样对我？我简直无法忍受！ | [infer.wav](./demos/voice4/infer_4_1.wav) |
| [system_audio.wav](./demos/voice5/system_audio.wav) | 今天的阳光真温暖，公园里的花开得特别灿烂！！ | [infer.wav](./demos/voice5/infer_5_1.wav) |
| [system_audio.wav](./demos/voice6/system_audio.wav) | 可是，她有一个不太好看的孩子，这个孩子被送到了挖沟工人的老婆家里抚养。而安妮·莉斯贝自己呢，住进了伯爵的公馆。 | [infer.wav](./demos/voice6/infer_6_1.wav) |
| [system_audio.wav](./demos/voice7/system_audio.wav) | 求求你…不要离开我，我真的好害怕… | [infer.wav](./demos/voice7/infer_7_1.wav) |
| [system_audio.wav](./demos/voice8/system_audio.wav) | 当我看到那双眼睛时，仿佛整个宇宙都安静了下来。 | [infer.wav](./demos/voice8/infer_8_1.wav) |
| [system_audio.wav](./demos/voice9/system_audio.wav) | 听到这个消息，我的心一下子沉到了谷底。 | [infer.wav](./demos/voice9/infer_9_1.wav) |
| [system_audio.wav](./demos/voice10/system_audio.wav) | 当我看到那双眼睛时，仿佛整个宇宙都安静了下来。 | [infer.wav](./demos/voice10/infer_10_1.wav) |



## Citation
```
@article{wang2025unitts,
  title={UniTTS: An end-to-end TTS system without decoupling of acoustic and semantic information},
  author={Rui Wang,Qianguo Sun,Tianrong Chen,Zhiyun Zeng,Junlong Wu,Jiaxing Zhang},
  journal={arXiv preprint arXiv:2408.16532},
  year={2025}
}
```

## References
The UniTTS model underwent a three-phase training paradigm consisting of pretraining, SFT, and DPO. Our training framework was developed through extensive customization of the open-source PAI-Megatron-Patch infrastructure. The training data underwent rigorous preprocessing utilizing open-source speech processing tools including FunASR and Whisper, which implemented advanced audio cleansing techniques such as voice activity detection and silence removal algorithms to ensure data quality.

[1] [pai-megagtron-patch](https://github.com/alibaba/Pai-Megatron-Patch/tree/main)

[2][FunASR](https://github.com/modelscope/FunASR)

[3][whisper](https://github.com/openai/whisper)


## Disclaimer

Our model provides zero-shot voice cloning services only for academic research purposes. We encourage the community to uphold safety and ethical principles in AI research and applications.

Important Notes:

- Compliance with the model's open-source license is mandatory.

- Unauthorized voice replication applications are strictly prohibited.

- Developers bear no responsibility for any misuse of this model.
