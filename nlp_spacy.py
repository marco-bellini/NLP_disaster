import numpy as np
import pandas as pd
import pylab as plt

import spacy
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from spacy.util import minibatch, compounding

import plac
import re
import urllib.request
from pathlib import Path


@plac.annotations(
    download_files=("Download data files True / False", "flag", "d" ),
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_texts=("Number of texts to train from", "option", "t", int),
    n_iter=("Number of training iterations", "option", "n", int),
)


def download_data_files():
    # download the dataset 
    url = 'https://datasets.figure-eight.com/figure_eight_datasets/disaster_response_data/disaster_response_messages_training.csv'  
    urllib.request.urlretrieve(url, 'disaster_response_messages_training.csv')
    url = 'https://datasets.figure-eight.com/figure_eight_datasets/disaster_response_data/disaster_response_messages_test.csv'  
    urllib.request.urlretrieve(url, 'disaster_response_messages_test.csv')
    url = 'https://datasets.figure-eight.com/figure_eight_datasets/disaster_response_data/disaster_response_messages_validation.csv'  
    urllib.request.urlretrieve(url, 'disaster_response_messages_validation.csv')
    
def spacy_tokenizer(message):
    # simple tokenizer
    
    message=re.sub("[#@$'!*+%\".()!,]?;",'',message).replace('','').replace('-','')
    message=' '.join(message.split())
    doc=nlp(message)
    words=[]

    remove_ent=[]
    for ent in doc.ents:
        if ent.label_ in ['GPE','LOC','NORP','FAC','ORG','LANGUAGE']:
            remove_ent.append(ent.text)

    # remove punctuation etc
    for token in doc:
        if ( (~token.is_stop)   & (token.pos_!='NUM') & (token.pos_!='PUNCT') & (token.pos_!='SYM') &
           ~(token.text in (remove_ent)) & (len(token.text)>1) ):
            words.append(token.text)
    return(words)

def embedding_tokenizer(message):
    message=re.sub("[#@$'!*+%\".()!,]?;",'',message).replace('','').replace('-','')
    message=' '.join(message.split())
    doc=nlp(message)
    words=[]

    remove_ent=[]
    for ent in doc.ents:
        if ent.label_ in ['GPE','LOC','NORP','FAC','ORG','LANGUAGE']:
            remove_ent.append(ent.text)

    # remove punctuation etc
    for token in doc:
        if ( (~token.is_stop)   & (token.pos_!='NUM') & (token.pos_!='PUNCT') & (token.pos_!='SYM') &
           ~(token.text in (remove_ent)) & (len(token.text)>1) & (token.has_vector) ):
            words.append(token.text)
    return(words)

def load_data(train_data,limit=0, split=0.8):
    """Load data from the IMDB dataset."""
    # Partition off part of the train data for evaluation

    random.shuffle(train_data)
    train_data = train_data[-limit:]
    texts, labels = zip(*train_data)
    cats = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)} for y in labels]
    split = int(len(train_data) * split)
    return (texts[:split], cats[:split]), (texts[split:], cats[split:])

def evaluate(tokenizer, textcat, texts, cats):
    
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if label == "NEGATIVE":
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}


def main( download_files , model=None, output_dir=None, n_iter=20, n_texts=2000):
    
    print(download_files, model, output_dir, n_iter, n_texts)
    
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()

    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.load("en_core_web_md")
        print("Loaded model '%s'" % "en_core_web_md")

    if download_files:
        print("downloading files...")
        download_data_files()
    
    df=pd.read_csv('disaster_response_messages_training.csv')
    df=df.drop(columns=['original','split'])
    train =pd.concat((df['message'], df[column]),axis=1).values

    # initialized cnn
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.create_pipe(
            "textcat",
            config={
                "exclusive_classes": True,
                "architecture": "simple_cnn",
            }
        )
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe("textcat")
    
    textcat.add_label('POSITIVE')
    textcat.add_label("NEGATIVE")

    train_texts = train_texts[:n_texts]
    train_cats = train_cats[:n_texts]

    print(
            "Using {} examples ({} training, {} evaluation)".format(
                n_texts, len(train_texts), len(dev_texts)
            )
        )
    
    train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "textcat"]
    
    init_tok2vec=None

    with nlp.disable_pipes(*other_pipes):  # only train textcat
        optimizer = nlp.begin_training()

        print("Training the model...")
        print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
        batch_sizes = compounding(4.0, 32.0, 1.001)
        for i in range(n_iter):
            losses = {}
            # batch up the examples using spaCy's minibatch
            random.shuffle(train_data)
            batches = minibatch(train_data, size=batch_sizes)
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
            print(
                "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print a simple table
                    losses["textcat"],
                    scores["textcat_p"],
                    scores["textcat_r"],
                    scores["textcat_f"],
                )
            )

    with nlp.use_params(optimizer.averages):
        nlp.to_disk(output_dir)
    print("Saved model to", output_dir)
    

if __name__ == "__main__":
    plac.call(main)