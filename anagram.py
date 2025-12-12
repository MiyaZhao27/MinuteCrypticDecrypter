from wordfreq import top_n_list
import string

english_words = set(top_n_list("en", 100000))

#  Function to create every permutation of a word
def generate_permutations(word):
    
    # Lowercasing word for consistency
    word = word.lower()
    
    # Edge case in case word has no permutations
    if len(word) <= 1:
        return {word}

    perms = set()

    # For each character in the word, create a substring recursively
    for i, char in enumerate(word):
        remaining = word[:i] + word[i+1:]
        for p in generate_permutations(remaining):
            perms.add(char + p)

    return perms


def anagrams_of(word):
    
    # Removing all spaces from the input word
    cleaned = word.replace(" ", "")
    return generate_permutations(cleaned)

# Filtering our permutations to see if it exists in our 100k most common English words
def filter_real_words(candidates):
    real_words = set()

    for cand in candidates:
        if cand.lower() in english_words:
            real_words.add(cand.lower())

    return real_words

# This function is an additional cae for anagram puzzles that have additional plus one letters (usually due to other clues)
def extend_with_added_letters(words):
    extended = set()

    for w in words:
        extended.add(w)

        for letter in string.ascii_lowercase:
            extended.add(letter + w)
            extended.add(w + letter)

    return extended


def do_anagram(word):
    # Generate all permutations of the input word's letters.
    perms = anagrams_of(word)

    # Filter the permutations to find the actual anagrams (real words).
    real_anas = filter_real_words(perms)

    # +1 letter variants
    extended_candidates = extend_with_added_letters(perms)

    # Filter the extended candidates
    plus_one_real = filter_real_words(extended_candidates)

    # Combine the two result sets (regular anagrams + plus-one anagrams)
    combined = real_anas.union(plus_one_real)

    # Remove the original word itself from the results
    combined.discard(word.lower())

    # Return list of candidates
    return combined

