import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
from PIL import Image

import torch
from peft import LoraConfig, get_peft_model
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq, LlavaForConditionalGeneration

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    TrlParser,
)

def get_dataset(data_dir="./data") -> DatasetDict:
    """加载数据集，样例见 README.md"""
    prompt = (
        "This is a 5128*5128 remote sensing image. You are required to plan a feasible flight path for a drone. "
        + "The red circle marks the starting point, the yellow circle marks the destination, "
        + "blue dots or rectangles indicate mandatory waypoints or regions, "
        + "and green dots or rectangles represent no-fly points or restricted zones. "
        + "Please provide a valid path (including the start and end points) in the format: "
        + "[(x1, y1), (x2, y2), ..., (xn, yn)]."
    )
    datasets = {}
    num_images = 9
    num_trajs = 7
    for split in ["train", "validation"]:
        datasets[split] = []
        if split == "validation":
            continue
        for img_id in range(1, num_images + 1):
            for traj_id in range(1, num_trajs + 1):
                img_path = os.path.join(data_dir, f"Train/{img_id}/{img_id}-{traj_id}.jpg")
                images = [img_path]

                json_path = os.path.join(data_dir, f"Label/{img_id}/{img_id}-{traj_id}.json")
                with open(json_path, "r") as f:
                    json_data = json.load(f)
                traj = json_data["route"]
                traj = [tuple(point) for point in traj]
                messages = [
                    {
                        "content": [
                            {"index": None, "type": "text", "text": prompt},
                            {"index": 0, "type": "image", "text": None}
                        ],
                        "role": "user"
                    },
                    {
                        "content": [
                            {"index": None, "type": "text", "text": str(traj)}
                        ],
                        "role": "assistant"
                    }
                ]

                datasets[split].append({"images": images, "messages": messages})
        # 保存为 JSON 文件
        with open(f"{split}_data.json", "w", encoding="utf-8") as f:
            json.dump(datasets[split], f, ensure_ascii=False, indent=4)
        datasets[split] = Dataset.from_list(datasets[split])
    return DatasetDict(datasets)


if __name__ == "__main__":
    # 验证 GPU 设置
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    model_name = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 加载数据集
    dataset = get_dataset("./data")

    # data collator
    def collate_fn(examples):
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        image_paths = [example["images"][0] for example in examples]
        images = [Image.open(img_path).convert("RGB") for img_path in image_paths]

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  # Padding

        # Ignore the image token index in the loss computation (model specific)
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch

    # 配置训练参数
    training_args = SFTConfig(
        output_dir="model/sft",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=8,
        learning_rate=2e-5,
        bf16=True,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    # 配置模型
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=dataset["train"],
        processing_class=processor,
        peft_config=lora_config,
    )

    # 开始训练
    print("Starting training...")
    trainer.train()
    trainer.save_model(training_args.output_dir)
