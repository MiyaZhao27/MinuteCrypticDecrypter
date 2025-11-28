import sys
import numpy as np
from gensim.models import KeyedVectors

model = None

def get_model():
    global model
    if model is None:
        print("Loading Word2Vec model....")
        model = KeyedVectors.load(
            "/Users/amandahuang/Desktop/MinuteCrypticSolver/word2vec-google-news-300.model",
            mmap="r"
        )
    return model