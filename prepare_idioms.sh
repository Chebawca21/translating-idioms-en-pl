python3 phrase_extractor/phrase_extractor.py --phrases data/idioms-en.txt --corpus data/europarl/Europarl.en-pl.en --output data/europarl/out_euro --lang en --model_size sm &&
python3 concatenate_idom_sentences.py --annotations data/europarl/out_euro/annotations.tsv --corpus data/europarl/Europarl.en-pl --output data/europarl --lang en-pl --test_split 0.3