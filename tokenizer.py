import spacy


class Tokenizer:
    def __init__(self):
        self.pl = spacy.load("pl_core_news_sm")
        self.en = spacy.load('en_core_web_sm')

    def tokens_to_string(self, tokens):
        reference = ""
        for token in tokens:
            reference = reference + token
            reference = reference + " "
        return reference

    def tokenize(self, source, lang):
        if lang == 'en':
            tokens = self.en(source)
        elif lang == 'pl':
            tokens = self.pl(source)
        else:
            return []

        tokens = map(lambda token: token.text.lower(), tokens)
        tokens = list(tokens)
        return self.tokens_to_string(tokens)
