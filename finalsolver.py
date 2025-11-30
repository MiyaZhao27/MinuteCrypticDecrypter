from indicatorcracker import predict_category_from_length
from hidden import ngrams_of, filter_real_words
from anagram import anagrams_of, filter_real_words


def run_anagram_algorithm(clue, indicator, fodder, length):
    all_anagrams = anagrams_of(fodder)
    real_words = filter_real_words(all_anagrams)

    def filter_by_length(words, length):
        return {w for w in words if len(w) == length}
    real_words = filter_by_length(real_words, length)

    if len(real_words) == 0:
        return f"No anagram English words of length {length} found."
    
    return f"Anagram English words of length {length}: {real_words}"



def run_reversal_algorithm(clue, indicator, fodder, length):
    return "I'm running the REVERSAL algorithm"


def run_hidden_algorithm(clue, indicator, fodder, length):
    all_ngrams = ngrams_of(length, fodder)
    real_words = filter_real_words(all_ngrams)

    if len(real_words) == 0:
        return f"No hidden English words of length {length} found."

    return f"Hidden English words of length {length}: {real_words}"


def run_selector_algorithm(clue, indicator, fodder, length):
    return "I'm running the SELECTOR algorithm"


def solve_clue():
    print("=== Minute Cryptic Decrypter ===\n")
    # prompt the user
    clue = input("Enter the clue: ")
    indicator = input("Enter the indicator word: ")
    fodder = input("Enter the fodder words: ")
    while True:
        try:
            length = int(input("Enter the solution length: "))
            break
        except:
            print("Length must be an integer. Try again.")

    # predict categroy based on length
    category, probs = predict_category_from_length(length)
    print(f"\nPredicted category: {category}")

    category_lower = category.lower()

    if "anagram" in category_lower:
        print(run_anagram_algorithm(clue, indicator, fodder, length))

    elif "reversal" in category_lower:
        print(run_reversal_algorithm(clue, indicator, fodder, length))

    elif "hidden" in category_lower:
        print(run_hidden_algorithm(clue, indicator, fodder, length))

    elif "selector" in category_lower:
        print(run_selector_algorithm(clue, indicator, fodder, length))

    else:
        print("Unknown category — sorry.")


if __name__ == "__main__":
    solve_clue()
