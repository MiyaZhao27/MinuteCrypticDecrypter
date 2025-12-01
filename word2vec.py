from gensim.models import KeyedVectors

MODEL_PATH = "glove.6B.50d.txt"


def get_model():
    print("Loading GloVe embeddings (50d)...")
    model = KeyedVectors.load_word2vec_format(
        MODEL_PATH,
        binary=False,
        no_header=True
    )
    print("Done!")
    return model
