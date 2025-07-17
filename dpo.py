import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import json

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoProcessor, AutoModelForVision2Seq, LlavaForConditionalGeneration

from trl import (
    ModelConfig,
    DPOConfig,
    DPOTrainer,
    TrlParser,
)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"