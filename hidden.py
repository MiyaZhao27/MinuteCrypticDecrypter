import re
from wordfreq import top_n_list

english_words = set(top_n_list("en", 100000))


def clean_fodder(text):
    """
    Clean the fodder by removing apostrophes, hyphens, dashes, punctuations,
    forces lowercase, and collapse spaces to treat them like a continous string
    """
    text = re.sub(r"[’'`]", "", text)
    text = re.sub(r"[-–—]", "", text)
    text = re.sub(r"[^A-Za-z ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()


def ngrams_of(n, word):
    """
    Return all n-grams of the fodders
    """
    word = clean_fodder(word)
    word = word.replace(" ", "")  # treat as continuous string
    ngrams = set()

    # normal hiddens
    for i in range(len(word) - n + 1):
        ngrams.add(word[i: i + n])

    # the reverse case
    rev = word[::-1]
    for i in range(len(rev) - n + 1):
        ngrams.add(rev[i:i+n])

    return ngrams


def filter_real_words(ngrams):
    """
    input: n-grams
    output: the n-grams that are valid english words
    """
    real_words = set()

    for ng in ngrams:
        cleaned = ng.lower()

        if cleaned in english_words:
            real_words.add(cleaned)

    return real_words
