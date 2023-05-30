## Setup


1. Create virtual environment

```bash
python3 -m venv venv
```

2. Activate virtual environment

```bash
source venv/bin/activate
```

3. Install requirements

```bash
pip install -r requirements.txt
```

4. Download models for tokenizing data

 ```bash
chmod +x download_data.sh
 ./download_data.sh
 ```

## Preparing idiom data
Even though idiom data is already in the repository to create new ones (for another corpus) you can use this script

 ```bash
chmod +x prepare_idioms.sh
 ./prepare_idioms.sh
 ```

## Training
To train and evaluate marian model using idiom data run `train_marian.py`

```bash
python3 train_marian.py
```