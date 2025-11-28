from nltk.corpus import words
import nltk
nltk.download('words')

english_words = set(w.lower() for w in words.words())

def generate_permutations(word):
    """
    Return all unique permutations of a word
    """
    word = word.lower()
    
    # Base case
    if len(word) <= 1:
        return {word}
    
    perms = set()
    
    # Recursive case: fix each letter in turn
    for i, char in enumerate(word):
        # Remove char at position i
        remaining = word[:i] + word[i+1:]
        
        # Permute the rest and prepend char
        for p in generate_permutations(remaining):
            perms.add(char + p)
    
    return perms


def anagrams_of(word):
    """
    Return all unique anagrams of the fodder word (no itertools).
    """
    cleaned = word.replace(" ", "")
    return generate_permutations(cleaned)


def filter_real_words(candidates):
    """
    Input: set of anagram strings
    Output: subset that are valid English words
    """
    real_words = set()

    for cand in candidates:
        if cand.lower() in english_words:
            real_words.add(cand.lower())

    return real_words
