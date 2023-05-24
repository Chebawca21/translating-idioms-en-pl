from transformers import MarianMTModel, MarianTokenizer
from transformers import pipeline
import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


MODEL_DIR = "models/mt-tc-en-pl"
TRAIN_NEW = False

EXAMPLE = ["Mother had a cat", "Szczepan is truly special"]


def main():
    
    if TRAIN_NEW:
        print("Work in progress")
    else:
        model = MarianMTModel.from_pretrained(MODEL_DIR)
        tokenizer = MarianTokenizer.from_pretrained(MODEL_DIR)
    
    src_text = EXAMPLE
    print(tokenizer.prepare_seq2seq_batch(src_text, return_tensors="pt"))
    translated = model.generate(**tokenizer.prepare_seq2seq_batch(src_text, return_tensors="pt"))
    tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    
    print(translated)
    print(tgt_text)




if __name__ == "__main__":
    main()