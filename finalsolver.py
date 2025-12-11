from hidden import ngrams_of, filter_real_words as filter_hidden_words
from anagram import do_anagram
from selector import generate_all_selectors, filter_real_words as filter_selector_words
import pandas as pd

import numpy as np
from glove import get_model

model = get_model()


def cosine_sim(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def get_vec(word):
    word = word.lower().strip()
    if word in model.key_to_index:
        return model[word]
    return None


def avg_vec(text):
    tokens = text.lower().strip().split()
    vecs = [get_vec(t) for t in tokens if get_vec(t) is not None]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def best_definition_match(definition, candidates):
    dvec = avg_vec(definition)
    if dvec is None:
        return None, {}

    scores = {w: cosine_sim(dvec, get_vec(w)) for w in candidates}
    best_word = max(scores, key=scores.get)
    return best_word, scores


def run_anagram_algorithm(fodder, length):
    words = do_anagram(fodder)
    return {w for w in words if len(w) == length}


def run_hidden_algorithm(fodder, length):
    return set(filter_hidden_words(ngrams_of(length, fodder)))


def run_selector_algorithm(fodder, length):
    return set(filter_selector_words(generate_all_selectors(fodder, length)))


def solve_clue():
    print("=== Minute Cryptic Decrypter + Glove Meaning Matcher ===\n")

    fodder = input("Enter the fodder: ").strip()

    while True:
        try:
            length = int(input("Enter the answer length: "))
            break
        except:
            print("Length must be an integer.")

    category = input("Enter category (anagram / hidden / selector): ").lower()

    # Generate candidates
    if "anagram" in category:
        candidates = run_anagram_algorithm(fodder, length)
    elif "hidden" in category:
        candidates = run_hidden_algorithm(fodder, length)
    elif "selector" in category:
        candidates = run_selector_algorithm(fodder, length)
    else:
        print("Unknown category.")
        return

    # NONE FOUND
    if not candidates:
        print("No English words found.")
        return

    # EXACTLY ONE → DONE
    if len(candidates) == 1:
        print("Solution:", list(candidates)[0])
        return

    # MULTIPLE → ask for definition
    print(f"\nMultiple candidate words: {candidates}")
    definition = input("Enter the DEFINITION part of the clue: ").strip()

    best, scores = best_definition_match(definition, candidates)

    print("\n=== Glove Scoring ===")
    print(f"Definition: {definition}")
    print(f"Best Match: {best}\n")

    for w, s in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"{w:<15} {s:.4f}")

    print("\nFinal Answer:", best)


if __name__ == "__main__":
    # --- interactive mode (unchanged) ---
    solve_clue()

    # --- batch mode: run over testsolver.csv and save candidates + similarities ---
    print("\n=== Running batch solver on testsolver.csv ===")

    ts_path = "testsolver.csv"
    out_ts_path = "testsolver_results.csv"

    try:
        ts_df = pd.read_csv(ts_path)
    except FileNotFoundError:
        print(f"Could not find {ts_path}. Skipping batch solve.\n")
    else:
        rows = []

        # Expecting: Clue,Category,Fodder,Length,definition
        for idx, row in ts_df.iterrows():
            clue = row["Clue"]
            category = row["Category"]
            fodder = row["Fodder"]
            length = int(row["Length"])
            definition = row["Definition"]

            # choose algorithm
            cat = category.lower()
            if "anagram" in cat:
                candidates = run_anagram_algorithm(fodder, length)
            elif "hidden" in cat:
                candidates = run_hidden_algorithm(fodder, length)
            elif "selector" in cat:
                candidates = run_selector_algorithm(fodder, length)
            else:
                candidates = set()

            if not candidates:
                # still record something if no candidates found
                rows.append({
                    "clue": clue,
                    "category": category,
                    "fodder": fodder,
                    "length": length,
                    "definition": definition,
                    "candidate": "",
                    "similarity_to_definition": 0.0
                })
                continue

            # score each candidate against the definition
            _, scores = best_definition_match(definition, candidates)

            for cand in candidates:
                rows.append({
                    "clue": clue,
                    "category": category,
                    "fodder": fodder,
                    "length": length,
                    "definition": definition,
                    "candidate": cand,
                    "similarity_to_definition": scores.get(cand, 0.0)
                })

        out_df = pd.DataFrame(rows)
        out_df.to_csv(out_ts_path, index=False)
        print(f"Saved batch results to {out_ts_path}")
