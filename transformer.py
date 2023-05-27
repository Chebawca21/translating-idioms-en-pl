from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
import evaluate
import numpy as np


MODEL_NAME = "gsarti/opus-mt-tc-en-pl"
PREFIX = "translate English to Polish: "
SOURCE_LANG = "en"
TARGET_LANG = "pl"


class Transformer:
    def __init__(self, device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(self.device)
        self.prefix = PREFIX
        self.source_lang = SOURCE_LANG
        self.target_lang = TARGET_LANG

    def translate(self, sentence):
        batch = self.tokenizer(sentence, return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(**batch)
        out = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return out

    def preprocess_function(self, examples):
        inputs = [self.prefix + example[self.source_lang] for example in examples["translation"]]
        targets = [example[self.target_lang] for example in examples["translation"]]
        model_inputs = self.tokenizer(inputs, text_target=targets, max_length=128, truncation=True)
        return model_inputs

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def train(self, tokenized_dataset):
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=MODEL_NAME)

        metric = evaluate.load("sacrebleu")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

            result = metric.compute(predictions=decoded_preds, references=decoded_labels)
            result = {"bleu": result["score"]}

            prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in preds]
            result["gen_len"] = np.mean(prediction_lens)
            result = {k: round(v, 4) for k, v in result.items()}
            return result

        training_args = Seq2SeqTrainingArguments(
            output_dir="pretrained_marian",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=2,
            predict_with_generate=True,
            fp16=True,
            push_to_hub=False,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
