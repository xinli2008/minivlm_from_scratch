# miniVLM

A lightweight vision-language model implementation built from scratch

## Installation

### Environment Setup
```bash
git clone https://github.com/xinli2008/minivlm_from_scratch.git
cd minivlm_from_scratch

conda create -n minivlm python=3.10 -y && conda activate minivlm
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Data Preparation

Pretrain_dataset：
```bash
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/pretrain_data.jsonl
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/pretrain_images.zip
unzip pretrain_images.zip && rm pretrain_images.zip
```

SFT_dataset：
```bash
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/sft_data.jsonl
wget https://hf-mirror.com/datasets/jingyaogong/minimind-v_dataset/resolve/main/sft_images.zip
unzip sft_images.zip && rm sft_images.zip
```

### Pretrained Weights

```bash
cd pretrained_model && wget https://huggingface.co/jingyaogong/MiniMind2-V-PyTorch/blob/main/llm_512.pth && cd ..
cd model/vision_model && git clone https://huggingface.co/openai/clip-vit-base-patch16 && cd ../../
```

## Architecture

![VLM-architecture](./assets/VLM-structure.png)
![VLM-architecture](./assets/VLM-structure-moe.png)

## Training

### Pretraining
```bash
python3 train_pretrain_vlm.py --epochs 10 --batch_size 64 --hidden_states 512 --from_weight llm --freeze_llm 1 
```

## Acknowledgements
1. our code is heavily inspired by [minimind-v](https://github.com/jingyaogong/minimind-v). Please refer to their repository for more details.