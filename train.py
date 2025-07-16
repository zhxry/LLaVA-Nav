# 基于 LLaVA 的航线规划微调代码框架
# 输入：遥感图像 + 自然语言 prompt
# 输出：路径坐标序列文本

import torch
from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict
from PIL import Image
import os
import json

# === Step 1: 加载模型和处理器 ===
model_name = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()

# === Step 2: 构造自定义 Dataset ===
def load_custom_dataset(image_dir, annotation_file):
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    data = []
    for item in annotations:
        img_path = os.path.join(image_dir, item['image'])
        prompt = item['prompt']
        output = item['path']
        data.append({
            "image": img_path,
            "prompt": prompt,
            "text_output": output
        })
    return DatasetDict({"train": Dataset.from_list(data)})

# 示例格式（JSON）：
# [
#   {"image": "img001.jpg", "prompt": "图中红点为起点...", "path": "[(200,300), (210,295), (220,290)]"},
#   ...
# ]

# === Step 3: Tokenize 输入数据 ===
def preprocess(example):
    image = Image.open(example["image"]).convert("RGB")
    inputs = processor(prompt=example["prompt"], images=image, return_tensors="pt")
    inputs = {k: v.squeeze(0) for k, v in inputs.items()}
    inputs['labels'] = processor.tokenizer(example["text_output"], return_tensors="pt")["input_ids"].squeeze(0)
    return inputs

# === Step 4: 设置训练器 ===
training_args = TrainingArguments(
    output_dir="llava_path_planner",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    fp16=True,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    learning_rate=5e-5,
    report_to="none",
)

class LLaVADataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = preprocess(self.dataset[idx])
        return item

# === Step 5: 加载数据并训练 ===
dataset = load_custom_dataset("./images", "annotations.json")
train_dataset = LLaVADataset(dataset["train"])

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()

# === 使用方式 ===
# 1. 下载 llava-1.5 模型：https://huggingface.co/llava-hf/llava-1.5-7b-hf
# 2. 构造 JSON 注释文件（带图路径、prompt 和路径输出）
# 3. 调整 batch size 和显存设置（7B 模型建议使用 A100）
