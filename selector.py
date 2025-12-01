from wordfreq import top_n_list

english_words = set(top_n_list("en", 100000))


def letters_from_words(words):
    """Flatten list of words into a list of characters."""
    chars = []
    for w in words:
        chars.extend(list(w))
    return chars

# basic word wise selectors like first letters, last letters, etc


def first_letters(words):
    return "".join(w[0] for w in words if len(w) > 0)


def last_letters(words):
    return "".join(w[-1] for w in words if len(w) > 0)


def middle_letters(words):
    return "".join(w[len(w)//2] for w in words if len(w) >= 3)


# nth letter selectors

def nth_letters(chars, n):
    return "".join(chars[0::n])


def odd_letters(chars):
    return "".join(chars[0::2])


def even_letters(chars):
    return "".join(chars[1::2])

# half words


def first_half(w):
    return w[: len(w)//2]


def second_half(w):
    return w[len(w)//2:]


# substrings

def word_substrings(w):
    subs = []
    for i in range(len(w)):
        for j in range(i+1, len(w)+1):
            subs.append(w[i:j])
    return subs


def string_substrings(s):
    subs = []
    for i in range(len(s)):
        for j in range(i+1, len(s)+1):
            subs.append(s[i:j])
    return subs

# sometimes you have to take the halves of two words and frakenstein them


def cross_half_combinations(words):
    """Combine first/second halves across all words correctly."""
    halves = []

    halves_list = []
    for w in words:
        if len(w) >= 2:
            halves_list.append((first_half(w), second_half(w)))

    # combine every half-A with every half-B
    for (A_fh, A_sh) in halves_list:
        for (B_fh, B_sh) in halves_list:

            halves.extend([
                A_fh + B_fh,
                A_fh + B_sh,
                A_sh + B_fh,
                A_sh + B_sh,
            ])

    return halves


# combine them all!


def generate_all_selectors(fodder, length=None):
    """
    Generate ALL selector possibilities:
    - word-wise (first/last/middle letters)
    - string-wise (odd/even/every-nth)
    - half-word (first half / second half)
    - cross-half combinations
    - all substrings of each word
    - all substrings of the full string
    """

    words = fodder.split()
    chars = letters_from_words(words)
    combined = "".join(chars)

    candidates = []

    # word-level selectors
    candidates.append(first_letters(words))
    candidates.append(last_letters(words))
    candidates.append(middle_letters(words))

    # string-level selectors
    candidates.append(odd_letters(chars))
    candidates.append(even_letters(chars))

    for n in range(2, max(3, len(chars) + 1)):
        candidates.append(nth_letters(chars, n))

    # half-word selectors
    for w in words:
        if len(w) >= 2:
            candidates.append(first_half(w))
            candidates.append(second_half(w))

    # cross-half combinations
    candidates.extend(cross_half_combinations(words))

    # substring selectors for each word
    for w in words:
        candidates.extend(word_substrings(w))

    # substring selectors for entire string
    candidates.extend(string_substrings(combined))

    # accomodate for the reverse case
    rev_candidates = [c[::-1] for c in candidates]
    candidates.extend(rev_candidates)

# remove dups
    candidates = [c for c in candidates if c]

    if length is not None:
        candidates = [c for c in candidates if len(c) == length]

    # remove duplicates
    candidates = list(dict.fromkeys(candidates))

    return candidates


def filter_real_words(candidates):
    """
    input: candidates
    output: the n-candidates that are valid english words
    """
    real_words = set()

    for c in candidates:
        cleaned = c.lower()

        if cleaned in english_words:
            real_words.add(cleaned)

    return real_words
