import os 
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer
from dotenv import load_dotenv

load_dotenv()

llama_access_token  = os.getenv("llama_access_token ")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B",  cache_dir = "./models")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", cache_dir = "./models")