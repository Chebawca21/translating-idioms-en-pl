from liter import LiTER
from tokenizer import Tokenizer
from transformer import Transformer, DataCollatorForSeq2Seq
from datasets import Dataset, load_dataset
import torch

FIRST_CORPUS = "data/ted2020/idiom_sentences.en"
SECOND_CORPUS = "data_/ted2020/idiom_sentences.pl"

def prepare_dataset(first_corpus, second_corpus):
    with open(first_corpus, 'r') as f:
        sentences = f.read().splitlines()

    with open(second_corpus, 'r') as f:
        translated_sentences = f.read().splitlines()

    first_lang = first_corpus[-2:]
    second_lang = second_corpus[-2:]

    dataset = {}
    dataset['id'] = []
    dataset['translation'] = []
    i = 0
    for first, second in zip(sentences, translated_sentences):
        dataset['id'].append(i)
        dataset['translation'].append({first_lang: first, second_lang: second})
    return Dataset.from_dict(dataset)


source_lang = "en"
target_lang = "pl"
prefix = "translate English to Polish: "

device = "cuda:0" if torch.cuda.is_available() else "cpu"

translator = Transformer(device)

dataset = prepare_dataset(FIRST_CORPUS, SECOND_CORPUS)
dataset = dataset.train_test_split(test_size=0.2)

tokenized_dataset = dataset.map(translator.preprocess_function, batched=True)

translator.train(tokenized_dataset)
