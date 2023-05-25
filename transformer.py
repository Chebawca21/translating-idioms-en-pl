from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class Transformer:
    def __init__(self, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("gsarti/opus-mt-tc-en-pl")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("gsarti/opus-mt-tc-en-pl").to(self.device)

    def translate(self, sentence):
        batch = self.tokenizer(sentence, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**batch)
        out = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return out
