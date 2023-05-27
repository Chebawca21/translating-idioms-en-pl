from liter import LiTER
from transformer import Transformer
import torch

with open('data/processed_data.txt', 'r', encoding='utf8') as f:
    sources = f.readlines()

with open('data/translated_data.txt', 'r', encoding='utf8') as f:
    references = f.readlines()

device = "cuda:0" if torch.cuda.is_available() else "cpu"

hypothesises = []

translator = Transformer(device)
liter = LiTER()

for i in range(len(sources)):
    hypothesis = translator.translate(sources[i])
    hypothesises.append(hypothesis)

n_literal = liter.evaluate_liter(sources, references, hypothesises, 'en', 'pl')

print(n_literal)
