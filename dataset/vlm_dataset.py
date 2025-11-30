import json
from PIL import Image
from torch.utils.data import Dataset
import os
import torch
from model.model_vlm import VLMModel

class VLMDataset(Dataset):
    def __init__(self, json_path, images_path, tokenizer, preprocess=None, max_length=512, image_special_token="@"*196):
        super().__init__()
        self.samples = self.load_json(json_path)
        self.images_path = images_path

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_token = image_special_token

        # NOTE: bos_id的作用是将<|im_start|>assistant这个特殊标记通过tokenizer转化为对应的token_id序列, 即整数序列。同理eos_id也是类似的作用。
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids
    
    def __len__(self):
        return len(self.samples)

    def load_json(self, json_path):
        samples = []
        with open(json_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    sample = json.loads(line.strip())
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line {line_num}: {e}")
        return samples
    
    def _create_chat_prompt(self, conversations):
        """
        将对话内容转换为chat格式的prompt, conversations示例:
        [
            {'role': 'user', 'content': '<image>\n简要, 清晰地说明所显示的图片.'},
            {'role': 'assistant', 'content': '鱼在游泳池里无监督地游泳'}
        ]
        先将<image>替换为image_token, 得到以下格式。其中@的数量为196个:
        [
            {'role': 'user', 'content': '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n简要, 清晰地说明所显示的图片.'},
            {'role': 'assistant', 'content': '鱼在游泳池里无监督地游泳'}
        ],
        然后调用tokenizer的apply_chat_template方法生成最终的prompt,得到以下格式。其中@的数量为196个:
        <|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n简要, 清晰地说明所显示的图片.<|im_end|>\n<|im_start|>assistant\n鱼在游泳池里无监督地游泳<|im_end|>\n
        """
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content'].replace('<image>', self.image_token)})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        """
        生成损失mask, 只对assistant的回答部分计算loss。
        例如, 对于以下input_ids, 假设它的序列长度是512。
        <|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n'@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n简要, 清晰地说明所显示的图片.<|im_end|>\n<|im_start|>assistant\n鱼在游泳池里无监督地游泳<|im_end|>\n
        最终得到的loss_mask的长度也是512, 其中只有鱼在游泳池里无监督地游泳<|im_end|>\n对应的位置是1, 其他位置都是0。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image_paths = sample['image']
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]

        # NOTE: padding到统一长度, 方便后续batch处理
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        
        # NOTE: 生成loss_mask, 只对assistant的回答部分计算loss
        loss_mask = self._generate_loss_mask(input_ids)

        # NOTE: 构造模型输入和标签, 这是一种典型的语言模型训练方式。
        # NOTE: X是输入序列, 包含input_ids的前n-1个token, Y是模型的目标序列，包含input_ids的后n-1个token。
        # 假设 input_ids是以下序列，[101, 102, 103, 104, 105], 则X将是[101, 102, 103, 104]，Y将是[102, 103, 104, 105]。
        # 训练时，模型会学习：基于X来预测Y中的下一个token, 即输入 101，预测 102； 输入 102，预测 103；以此类推。
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        image_tensors = []
        for image_name in image_paths.split(','):
            image_name = image_name.strip()
            image = Image.open(f'{self.images_path}/{image_name}')
            image_tensor = VLMModel.image2tensor(image, self.preprocess)
            image_tensors.append(image_tensor)
        image_tensors = torch.stack(image_tensors, dim=0)

        return X, Y, loss_mask, image_tensors