import gensim.downloader as api

model = None


def get_model():
    global model
    if model is None:
        print("Loading FastText 300d model from Gensim...")
        model = api.load("fasttext-wiki-news-subwords-300")
        print("Model loaded.")
    return model
