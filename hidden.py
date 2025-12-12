import re
from wordfreq import top_n_list

# decided against using NLTK library as it screened some words
# like origami and chess out
english_words = set(top_n_list("en", 100000))

# I cleaned the fodders by doing stuff like removing apostrophes, hyphens, dashes,
# forcing lowercase, and removing anything not a letter or a space


def clean_fodder(text):
    text = re.sub(r"[’'`]", "", text)
    text = re.sub(r"[-–—]", "", text)
    text = re.sub(r"[^A-Za-z ]", "", text)
    return text.lower().strip()

# Then I returned all the n-grams of the fodders

def ngrams_of(n, word):
    word = clean_fodder(word)
    word = word.replace(" ", "")  # treat as continuous string
    ngrams = set()

    # normal hiddens
    for i in range(len(word) - n + 1):
        ngrams.add(word[i: i + n])

    # the reverse case (happens occasionally, if the reverse isn't a
    # real word it would be filtered out)
    rev = word[::-1]
    for i in range(len(rev) - n + 1):
        ngrams.add(rev[i:i+n])

    return ngrams

# since not all ngrams would make sense as words we filtered them through
# the 100K most frequent words in english.

def filter_real_words(ngrams):
    real_words = set()

    for ng in ngrams:
        cleaned = ng.lower()

        if cleaned in english_words:
            real_words.add(cleaned)

    return real_words
