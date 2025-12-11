import gensim.downloader as api


def get_model():
    print("Loading GloVe 50d embeddings from Gensim API…")
    model = api.load("glove-wiki-gigaword-50")
    print("Done!")
    return model
