from nltk.corpus import words
import nltk
nltk.download('words')

english_words = set(w.lower() for w in words.words())


def ngrams_of(n, word):
    """
    Return all n-grams of the fodders
    """
    word = word.replace(
        " ", "")  # removes the sapces to treat them like one string
    ngrams = set()

    for i in range(len(word) - n + 1):
        ngrams.add(word[i: i + n])

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
