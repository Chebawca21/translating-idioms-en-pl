import argparse
import pandas as pd
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations',
                        help='tsv file created by the phrase extractor.',
                        required=True)
    parser.add_argument('--corpus',
                        help='Path to the corpus files (without the language).',
                        required=True)
    parser.add_argument('--output',
                        help='Output **directory** containing the sentences in both languages.',
                        required=True)
    parser.add_argument('--lang',
                        help='Languages from which to wchich language translation is being made in format: en-pl.',
                        required=True)
    parser.add_argument('--test_split',
                        help='How much of the annotations should be for the test instances',
                        required=True)

    args = parser.parse_args()

    columns = ['Index', 'Idiom', 'Spans', 'Sentence']
    annotations = pd.read_csv(args.annotations, sep='\t', header=None, names=columns, engine='python', quoting=3)
    indexes = np.array(annotations['Index'])

    # Make dictionary of indexes and sentences grouped by idioms
    idioms = {}
    for _, row in annotations.iterrows():
        index = row['Index']
        idiom = row['Idiom']
        sentence = row['Sentence'] + '\n'
        if idiom not in idioms:
            idioms[idiom] = []

        idioms[idiom].append((index, sentence))

    # Distribute idioms and sentences to train and test
    indexes_train = []
    sentences_train = []
    indexes_test = []
    sentences_test = []
    for key, values in idioms.items():
        if len(values) > 1:  # Ignore idioms which have one example
            for i in range(len(values)):
                if i < len(values) * float(args.test_split):
                    indexes_test.append(values[i][0])
                    sentences_test.append(values[i][1])
                else:
                    indexes_train.append(values[i][0])
                    sentences_train.append(values[i][1])

    source_lang = args.lang[:2]
    target_lang = args.lang[-2:]
    source_corpus = args.corpus + '.' + source_lang
    target_corpus = args.corpus + '.' + target_lang

    # Read corpus
    with open(source_corpus, 'r') as f:
        sentences = f.readlines()

    with open(target_corpus, 'r') as f:
        target_sentences = f.readlines()

    target_sentences = np.array(target_sentences)
    target_sentences_train = target_sentences[indexes_train]
    target_sentences_test = target_sentences[indexes_test]

    sent_train_path = args.output + '/idiom_sentences_train.' + source_lang
    target_sent_train_path = args.output + '/idiom_sentences_train.' + target_lang
    sent_test_path = args.output + '/idiom_sentences_test.' + source_lang
    target_sent_test_path = args.output + '/idiom_sentences_test.' + target_lang

    # Write idiom data
    with open(sent_train_path, 'w') as f:
        f.writelines(sentences_train)

    with open(target_sent_train_path, 'w') as f:
        f.writelines(target_sentences_train)

    with open(sent_test_path, 'w') as f:
        f.writelines(sentences_test)

    with open(target_sent_test_path, 'w') as f:
        f.writelines(target_sentences_test)

    # Write corpus data with no idioms
    source_corpus = args.corpus + '_no-idioms' + '.' + source_lang
    target_corpus = args.corpus + '_no-idioms' + '.' + target_lang
    with open(source_corpus, 'w') as f:
        f.writelines(np.delete(sentences, indexes))

    with open(target_corpus, 'w') as f:
        f.writelines(np.delete(target_sentences, indexes))
