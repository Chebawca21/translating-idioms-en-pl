from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Transformer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gsarti/opus-mt-tc-en-pl")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("gsarti/opus-mt-tc-en-pl")

    def translate(self, sentence):
        batch = self.tokenizer([sentence], return_tensors="pt")
        generated_ids = self.model.generate(**batch)
        out = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return out
