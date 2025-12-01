import numpy as np
from word2vec import get_model

model = get_model()

# Cosine similarity


def cosine_sim(v1, v2):
    if v1 is None or v2 is None:
        return 0.0
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def get_vec(word):
    word = word.lower()
    if word in model.key_to_index:
        return model[word]
    return None

# Category indicator lists


ANAGRAM_WORDS = ["mix", "throwing", "destroy", "strange",
                 "dancing", "sort", "tampering", "exploded"]
HIDDEN_WORDS = ["hides", "displays", "reveals", "within", "held",
                "capturing", "absorbed", "sample", "selection", "bit", "taken"]
SELECTOR_WORDS = ["head", "tail", "heart", "borders", "coat",
                  "contents", "guts", "odd", "even", "alternate", "regularly"]


CATEGORY_MAP = {
    "anagram": ANAGRAM_WORDS,
    "selectors": SELECTOR_WORDS,
    "hidden": HIDDEN_WORDS,
}

# Score computation


def w2v_category_scores(indicator):
    indicator = indicator.lower().strip()
    ivec = get_vec(indicator)

    scores = {}
    for cat, wordlist in CATEGORY_MAP.items():
        sims = [cosine_sim(ivec, get_vec(w)) for w in wordlist]
        scores[cat] = float(np.mean(sims))
    return scores


def w2v_best_category(indicator):
    scores = w2v_category_scores(indicator)
    best = max(scores, key=scores.get)
    return best, scores

# Interactive


def interactive_w2v():
    print("\n=== Word2Vec Category Predictor ===\n")

    while True:
        word = input("Enter an INDICATOR word (or 'quit'): ").strip()
        if word.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        best, scores = w2v_best_category(word)

        print(f"\nIndicator: {word}")
        print(f"Predicted Category: **{best.upper()}**\n")

        print("Scores:")
        for cat, score in sorted(scores.items(), key=lambda x: -x[1]):
            print(f"  {cat:<10} {score:.4f}")

        print("\n-----------------------------------\n")


if __name__ == "__main__":
    interactive_w2v()
