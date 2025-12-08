from wordfreq import top_n_list
import string

english_words = set(top_n_list("en", 100000))

def generate_permutations(word):
    word = word.lower()

    if len(word) <= 1:
        return {word}

    perms = set()

    for i, char in enumerate(word):
        remaining = word[:i] + word[i+1:]
        for p in generate_permutations(remaining):
            perms.add(char + p)

    return perms


def anagrams_of(word):
    cleaned = word.replace(" ", "")
    return generate_permutations(cleaned)


def filter_real_words(candidates):
    real_words = set()

    for cand in candidates:
        if cand.lower() in english_words:
            real_words.add(cand.lower())

    return real_words


def extend_with_added_letters(words):
    extended = set()

    for w in words:
        extended.add(w)

        for letter in string.ascii_lowercase:
            extended.add(letter + w)
            extended.add(w + letter)

    return extended


def do_anagram(word):
    # all permutations
    perms = anagrams_of(word)

    # real anagrams
    real_anas = filter_real_words(perms)

    # +1 letter variants
    extended_candidates = extend_with_added_letters(perms)

    # filter extended ones
    plus_one_real = filter_real_words(extended_candidates)

    # combine into one result
    combined = real_anas.union(plus_one_real)

    # remove the original word
    combined.discard(word.lower())

    return combined

