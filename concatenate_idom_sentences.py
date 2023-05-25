import argparse
import pandas as pd
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations',
                        help='tsv file created by the phrase extractor.',
                        required=True)
    parser.add_argument('--corpus',
                        help='Text file containg translated sentences.',
                        required=True)
    parser.add_argument('--output',
                        help='Output **directory** containing the sentences in both languages.',
                        required=True)
    parser.add_argument('--lang',
                        help='Languages from which to wchich language translation is being made in format: en-pl.',
                        required=True)

    args = parser.parse_args()

    columns = ['Index', 'Idiom', 'Spans', 'Sentence']
    annotations = pd.read_csv(args.annotations, sep='\t', header=None, names=columns, engine='python', quoting=3)
    indexes = np.array(annotations['Index'])
    sentences = np.array(annotations['Sentence'])
    sentences = sentences + '\n'

    with open(args.corpus, 'r') as f:
        translated_sentences = f.readlines()
    translated_sentences = np.array(translated_sentences)
    translated_sentences = translated_sentences[indexes]

    first_languge = args.lang[:2]
    second_language = args.lang[-2:]
    sent_path = args.output + '/idiom_sentences.' + first_languge
    trans_sent_path = args.output + '/idiom_sentences.' + second_language

    with open(sent_path, 'w') as f:
        f.writelines(sentences)

    with open(trans_sent_path, 'w') as f:
        f.writelines(translated_sentences)
