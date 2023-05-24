from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer, AutoModelForSeq2SeqLM
src_text = [
    'Please let me go',
]

dir = "models/mt-tc-en-pl"
tokenizer = MarianTokenizer.from_pretrained("models/spm")

model = MarianMTModel.from_pretrained(dir)
translated = model.generate(**tokenizer.prepare_seq2seq_batch(src_text, return_tensors="pt"))
tgt_text = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

print(tgt_text)
#model.save_pretrained(dir)
#tokenizer.save_pretrained(dir)