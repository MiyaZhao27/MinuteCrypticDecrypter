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
    """Compute the average vector for a sentence/definition."""
    tokens = text.lower().strip().split()
    vecs = [get_vec(t) for t in tokens if get_vec(t) is not None]

    if not vecs:
        return None

    return np.mean(vecs, axis=0)


def best_definition_match(definition, candidates):
    """
    Given a definition string and a list of candidate words,
    return the candidate whose vector is closest in meaning.
    """
    dvec = avg_vec(definition)
    if dvec is None:
        return None, {}

    scores = {}
    for w in candidates:
        wvec = get_vec(w)
        scores[w] = cosine_sim(dvec, wvec)

    best_word = max(scores, key=scores.get)
    return best_word, scores


def interactive_definition_matcher():
    print("\n=== Word Embedding Definition Matcher ===\n")

    while True:
        definition = input("Enter the DEFINITION (or 'quit'): ").strip()
        if definition.lower() in ["quit", "exit"]:
            print("\nGoodbye!")
            break

        cand_str = input("Enter candidate words (comma separated): ").strip()
        candidates = [c.strip() for c in cand_str.split(",")]

        best, score_map = best_definition_match(definition, candidates)

        print(f"\nDefinition: {definition}")
        print(f"Best Match: **{best}**\n")

        print("Scores:")
        for w, s in sorted(score_map.items(), key=lambda x: -x[1]):
            print(f"  {w:<15} {s:.4f}")

        print("\n-------------------------------------\n")


if __name__ == "__main__":
    interactive_definition_matcher()
