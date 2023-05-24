from liter import LiTER
from tokenizer import Tokenizer
from transformer import Transformer

with open('data_after/processed_data.txt', 'r', encoding='utf8') as f:
    sources = f.readlines()

with open('data_after/translated_data.txt', 'r', encoding='utf8') as f:
    references = f.readlines()

hypothesises = []

translator = Transformer()
tokenizer = Tokenizer()
liter = LiTER()

n_literal = 0
for i in range(len(sources)):
    hypothesis = translator.translate(sources[i])
    hypothesises.append(hypothesis)

    tk_source = tokenizer.tokenize(sources[i], 'en')
    tk_reference = tokenizer.tokenize(references[i], 'pl')
    tk_hypothesis = tokenizer.tokenize(hypothesis, 'pl')

    is_literal = liter.has_liter(tk_source, tk_reference, tk_hypothesis, 'en')
    if is_literal:
        print(i + 1, hypothesis)
        n_literal = n_literal + 1

print(n_literal)
