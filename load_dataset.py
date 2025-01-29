from datasets import load_dataset

ds = load_dataset("mlabonne/guanaco-llama2-1k", cache_dir = "/home/binit/finetune_llama")
dataset = load_dataset(ds, split="train")