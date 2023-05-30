from liter import LiTER
from transformer import Transformer
from datasets import Dataset
import torch

FIRST_CORPUS_TRAIN = "data/europarl/idiom_sentences_train.en"
SECOND_CORPUS_TRAIN = "data/europarl/idiom_sentences_train.pl"
FIRST_CORPUS_TEST = "data/europarl/idiom_sentences_test.en"
SECOND_CORPUS_TEST = "data/europarl/idiom_sentences_test.pl"

PREFIX = "translate English to Polish: "
SOURCE_LANG = 'en'
TARGET_LANG = 'pl'


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

if __name__ == '__main__':
    with open(FIRST_CORPUS_TEST, 'r', encoding='utf8') as f:
        sources = f.readlines()

    with open(SECOND_CORPUS_TEST, 'r', encoding='utf8') as f:
        references = f.readlines()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    translator = Transformer(device)
    liter = LiTER()

    hypothesises = translator.translate(sources)
    n_literal_before = liter.evaluate_liter(sources, references, hypothesises, SOURCE_LANG, TARGET_LANG)
    bleu_before = translator.evaluate_bleu(sources, references)
    print(bleu_before)

    train_dataset = prepare_dataset(FIRST_CORPUS_TRAIN, SECOND_CORPUS_TRAIN)
    test_dataset = prepare_dataset(FIRST_CORPUS_TEST, SECOND_CORPUS_TEST)
    tokenized_train_dataset = train_dataset.map(translator.preprocess_function, batched=True)
    tokenized_test_dataset = test_dataset.map(translator.preprocess_function, batched=True)
    translator.train_idioms(tokenized_train_dataset, tokenized_test_dataset, sources, references)

    hypothesises = translator.translate(sources)
    n_literal_after = liter.evaluate_liter(sources, references, hypothesises, SOURCE_LANG, TARGET_LANG)
    bleu_after = translator.evaluate_bleu(sources, references)

    print(n_literal_before, n_literal_after)
