# LLaVA-Nav

基于 LLaVA 的遥感图像航线规划，使用 RLHF 训练

## SFT

### 数据

数据集结构如下：
```
.
├── data/
│   ├── Train/
│   │   ├── 1/
│   │   |   ├── 1-1.jpg
│   │   |   ├── 1-2.jpg
│   │   |   ├── 1-3.jpg
│   │   |   ├── 1-4.jpg
│   │   |   ├── 1-5.jpg
│   │   |   ├── 1-6.jpg
│   │   |   └── 1-7.jpg
│   │   ├── 2/...
│   │   ├── 3/...
│   │   │   ...
│   │   └── 9/...
│   ├── Label/
│   │   ├── 1/
│   │   │   ├── 1-1.json
│   │   │   ├── 1-2.json
│   │   │   ├── 1-3.json
│   │   │   ├── 1-4.json
│   │   │   ├── 1-5.json
│   │   │   ├── 1-6.json
│   │   │   ├── 1-7.json
│   │   ├── 2/...
│   │   ├── 3/...
│   │   │   ...
│   │   └── 9/...
└── train.py
```

- 数据在 `data` 下，包含 `Train` 和 `Label` 两个目录，下有 `1`~`9` 的子目录，每个子目录下包含 7 张遥感图像 (`Train`) 和对应的 JSON 标签文件 (`Label`)。
- 文件命名格式为 `IDX-TYPE.jpg` 和 `IDX-TYPE.json`，同一 `IDX` 为同一张遥感图像，`TYPE` 是同一张图像的不同任务。

数据格式如下
```json
images = ["./data/Train/1/1-1.jpg"]
messages = [
    {
        "content": [
            {
                "index": null,
                "type": "text",
                "text": "Plan a feasible flight path for the drone."
            },
            {
                "index": 0,
                "type": "image",
                "text": null
            }
        ],
        "role": "user"
    },
    {
        "content": [
            {
                "index": null,
                "type": "text",
                "text": "[(930, 690), (800, 1360), (710, 1910), (880, 2380)]"
            }
        ],
        "role": "assistant"
    }
]
```

## DPO

### Preference Dataset

DPO 训练需要使用偏好数据集，格式如下：
```python
# Standard format
## Explicit prompt (recommended)
preference_example = {"prompt": "The sky is", "chosen": " blue.", "rejected": " green."}
# Implicit prompt
preference_example = {"chosen": "The sky is blue.", "rejected": "The sky is green."}

# Conversational format
## Explicit prompt (recommended)
preference_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                      "chosen": [{"role": "assistant", "content": "It is blue."}],
                      "rejected": [{"role": "assistant", "content": "It is green."}]}
## Implicit prompt
preference_example = {"chosen": [{"role": "user", "content": "What color is the sky?"},
                                 {"role": "assistant", "content": "It is blue."}],
                      "rejected": [{"role": "user", "content": "What color is the sky?"},
                                   {"role": "assistant", "content": "It is green."}]}
```

特殊地，对于 VLM，需要做如下替换：

```python
# Textual dataset:
"content": "What color is the sky?"

# Vision dataset:
"content": [
    {"type": "image"}, 
    {"type": "text", "text": "What color is the sky in the image?"}
]
```