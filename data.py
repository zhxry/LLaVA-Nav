import os
import json
from datasets import DatasetDict
from transformers import AutoProcessor
from PIL import Image


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
    return DatasetDict(datasets)


if __name__ == "__main__":
    data_dir = "./data"
    dataset = get_dataset(data_dir)
    # print(dataset)
    print(f"Loaded dataset with {len(dataset['train'])} training samples and {len(dataset['validation'])} validation samples.")

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model_name = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    def collate_fn(examples):
        texts = [processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        image_paths = [example["images"][0] for example in examples]
        images = [Image.open(img_path).convert("RGB") for img_path in image_paths]
        print(texts, image_paths)

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

    examples = [dataset["train"][0], dataset["train"][1]]
    collated_data = collate_fn(examples)
    print(collated_data.keys())
    exit()