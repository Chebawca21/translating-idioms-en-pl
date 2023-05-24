EN_PL = "data_after/en-pl.txt"
PL_EN = "data_after/pl-en.txt"


class LiTER:
    def __init__(self):
        self.en_pl = {}
        self.pl_en = {}

        with open(EN_PL, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                words = line.split()
                if not words[0] in self.en_pl:
                    self.en_pl[words[0]] = []
                self.en_pl[words[0]].append(words[1])

        with open(PL_EN, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                words = line.split()
                if not words[0] in self.pl_en:
                    self.pl_en[words[0]] = []
                self.pl_en[words[0]].append(words[1])

    def get_translations(self, word, lang):
        if lang == 'en':
            if word in self.en_pl:
                return self.en_pl[word]
            else:
                return []
        elif lang == 'pl':
            if word in self.pl_en:
                return self.pl_en[word]
            else:
                return []
        else:
            return []

    def has_liter(self, source, reference, hypothesis, lang):
        blocklists = []
        source_words = source.split()
        for word in source_words:
            translations = self.get_translations(word, lang)
            if translations:
                blocklists.append(translations)

        reference_words = reference.split()
        delete_blocklist = [False] * len(blocklists)
        for word in reference_words:
            i = -1
            for blocklist in blocklists:
                i = i + 1
                is_in = False
                for blocked in blocklist:
                    if word == blocked:
                        is_in = True
                        break
                if is_in:
                    delete_blocklist[i] = True
                    break
        i = 0
        for delete in delete_blocklist:
            if delete:
                blocklists[i] = []
            i = i + 1

        hypothesis_words = hypothesis.split()
        has_blocked = False
        for word in hypothesis_words:
            for blocklist in blocklists:
                for blocked in blocklist:
                    if word == blocked:
                        has_blocked = True
                        break
                if has_blocked:
                    break
            if has_blocked:
                break

        return has_blocked
