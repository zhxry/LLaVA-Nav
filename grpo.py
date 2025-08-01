import os
import json
from PIL import Image

import torch
from peft import LoraConfig
from datasets import Dataset, DatasetDict

from transformers import AutoProcessor, AutoModelForVision2Seq
from trl import GRPOTrainer, GRPOConfig

from reward import get_reward_funcs


def get_dataset(data_dir="./data") -> DatasetDict:
    """加载数据集，样例见 README.md"""
    with open("./prompt.md", 'r', encoding='utf-8') as f:
        prompt = f.read().strip()
    num_images = 24
    num_trajs = 7
    datasets = {}
    for split in ["train", "validation"]:
        datasets[split] = []
        if split == "validation":
            continue
        for img_id in range(1, num_images + 1):
            for traj_id in range(1, num_trajs + 1):
                img_path = os.path.join(data_dir, f"Train/{img_id}/{img_id}-{traj_id}.jpg")
                img = Image.open(img_path).convert("RGB").resize((512, 512), Image.BICUBIC)

                messages = [
                    {"role": "user", "content": prompt}
                ]

                json_path = os.path.join(data_dir, f"Label/{img_id}/{img_id}-{traj_id}.json")
                with open(json_path, "r") as f:
                    json_data = json.load(f)
                traj = json_data["route"]
                traj = [(x // 10, y // 10) for x, y in traj]
                solution = str(traj)

                datasets[split].append({"image": img, "prompt": messages, "solution": solution})
        # 保存为 JSON 文件
        with open(f"{split}_data_grpo.json", "w", encoding="utf-8") as f:
            json.dump(datasets[split], f, ensure_ascii=False, indent=4)
        datasets[split] = Dataset.from_list(datasets[split])
    return DatasetDict(datasets)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 验证 GPU 设置
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    dataset = get_dataset("./data")

    training_args = GRPOConfig(
        output_dir="model/grpo-qwen7b",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=8,
        learning_rate=2e-5,  # 25 组图片可以改成 1e-5
        bf16=True,
        max_prompt_length=2048,  # 不能太小
        max_completion_length=512,  # 如果使用 CoT 可以改成 1024
        remove_unused_columns=False,
        gradient_checkpointing=True,
        reward_weights=[0.5, 0.5]
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=get_reward_funcs(),
        args=training_args,
        train_dataset=dataset["train"],
        processing_class=processor,
    )

    # 开始训练
    print("Starting training...")
    trainer.train()
    trainer.save_model(training_args.output_dir)
