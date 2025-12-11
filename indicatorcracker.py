from glove import get_model
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# load data

data_path = "logistic_data.csv"

try:
    df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Could not find {data_path}. Make sure it is in the same folder.")
    sys.exit(1)

# Expecting columns: category, clue, indicator, fodder, definition, length


# Load GloVe Model

print("Loading GloVe model...")
model = get_model()
print("GloVe model loaded.")


# compare cosine similarities
def cosine_sim(v1, v2):
    """Cosine similarity between two vectors, safely handling None/zero-norm."""
    if v1 is None or v2 is None:
        return 0.0
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)

# return teh GloVe vectorization


def get_vec(word):
    """Return GloVe vector for a word, or None if OOV."""
    if word is None:
        return None
    word = str(word).lower()
    if word in model.key_to_index:
        return model[word]
    return None


# wordbanks for each clue type
ANAGRAM_WORDS = [
    "mix", "throwing", "destroy", "strange",
    "dancing", "sort", "tampering", "exploded"
]

HIDDEN_WORDS = [
    "hides", "displays", "reveals", "within", "held",
    "capturing", "absorbed", "sample", "selection", "bit", "taken"
]

SELECTOR_WORDS = [
    "head", "tail", "heart", "borders", "coat",
    "contents", "guts", "odd", "even", "alternate", "regularly"
]


def avg_similarity(indicator, word_list):
    """Average cosine similarity between indicator and a list of reference words."""
    ivec = get_vec(indicator)
    sims = [cosine_sim(ivec, get_vec(w)) for w in word_list]
    return float(np.mean(sims)) if sims else 0.0


# features

# Cleaned fodder length: count only alphabetic characters
df["fodder_length"] = (
    df["fodder"]
    .astype(str)
    .str.replace(r"[^A-Za-z]", "", regex=True)
    .str.len()
)

# Number of words in fodder
df["fodder_word_count"] = df["fodder"].astype(str).str.split().apply(len)

# GloVe similarity features based on the indicator
df["glove_anagram"] = df["indicator"].apply(
    lambda x: avg_similarity(str(x), ANAGRAM_WORDS)
)
df["glove_hidden"] = df["indicator"].apply(
    lambda x: avg_similarity(str(x), HIDDEN_WORDS)
)
df["glove_selector"] = df["indicator"].apply(
    lambda x: avg_similarity(str(x), SELECTOR_WORDS)
)

# Feature matrix
FEATURE_COLS = [
    "length",
    "fodder_length",
    "fodder_word_count",
    "glove_anagram",
    "glove_hidden",
    "glove_selector",
]

X = df[FEATURE_COLS].values.astype(float)
y_raw = df["category"].values

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)


# Cross-Validated Evaluation

print("\n=== Cross-Validated Evaluation (5-fold Stratified) ===")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_y_true = []
all_y_pred = []

fold_idx = 1
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Scale within fold (avoid data leakage)
    scaler_cv = StandardScaler()
    X_train_scaled = scaler_cv.fit_transform(X_train)
    X_test_scaled = scaler_cv.transform(X_test)

    # Logistic Regression (multinomial)
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=500
    )
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

    fold_idx += 1

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

print("\n=== Classification Report (aggregated over folds) ===")
print(classification_report(
    all_y_true,
    all_y_pred,
    target_names=label_encoder.classes_,
    zero_division=0
))

print("\n=== Confusion Matrix (aggregated over folds) ===")
cm = confusion_matrix(all_y_true, all_y_pred)
print(cm)

print("\nLabels (row = true, col = pred):", list(label_encoder.classes_))


# Final Model

# Train scaler and model on FULL dataset for interactive use
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

logreg = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=500
)
logreg.fit(X_scaled, y)

print("\nFinal model trained on all data. Ready for interactive prediction.")


# Predict new categories

def extract_features(clue, indicator, length, fodder, definition):
    """
    Given a new clue, build the feature vector in the same way
    as for the training data.
    """
    # Clean fodder: count only letters
    fodder_clean_len = len("".join([c for c in str(fodder) if c.isalpha()]))

    return np.array([[
        float(length),
        float(fodder_clean_len),
        len(str(fodder).split()),
        avg_similarity(indicator, ANAGRAM_WORDS),
        avg_similarity(indicator, HIDDEN_WORDS),
        avg_similarity(indicator, SELECTOR_WORDS)
    ]])


def predict_category(clue, indicator, length, fodder, definition):
    """
    Predict category and probability distribution for a new clue.
    """
    x = extract_features(clue, indicator, length, fodder, definition)
    x_scaled = scaler.transform(x)

    pred_id = logreg.predict(x_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_id])[0]
    pred_probs = logreg.predict_proba(x_scaled)[0]

    return pred_label, pred_probs


# Interactive part

print("\n=== Test a New Unseen Clue ===")

try:
    clue = input("Enter the clue: ")
    indicator = input("Enter the indicator: ")
    fodder = input("Enter the fodder words: ")
    definition = input("Enter the definition: ")

    while True:
        try:
            length = int(input("Enter the solution length: "))
            break
        except ValueError:
            print("Length must be an integer, try again.")

    pred_label, pred_probs = predict_category(
        clue, indicator, length, fodder, definition
    )

    print("\n=== Prediction Result ===")
    print("Predicted Category:", pred_label)

    print("\nProbabilities:")
    for cat, p in zip(label_encoder.classes_, pred_probs):
        print(f"  {cat}: {p:.4f}")

except KeyboardInterrupt:
    print("\nExiting interactive mode.")


# Per-clue probabilities for entire dataset

results = []

for idx, row in df.iterrows():
    clue = row["clue"]
    indicator = row["indicator"]
    fodder = row["fodder"]
    definition = row["definition"]
    length = row["length"]
    true_cat = row["category"]

    # Build features exactly like training
    x = extract_features(clue, indicator, length, fodder, definition)
    x_scaled = scaler.transform(x)

    # Get probabilities and predicted label
    prob_vec = logreg.predict_proba(x_scaled)[0]
    pred_id = np.argmax(prob_vec)
    pred_cat = label_encoder.inverse_transform([pred_id])[0]

    # Map probabilities to category names
    prob_dict = {
        f"prob_{cat}": prob_vec[i]
        for i, cat in enumerate(label_encoder.classes_)
    }

    # Store a row of results
    results.append({
        "clue": clue,
        "true_category": true_cat,
        "predicted_category": pred_cat,
        **prob_dict
    })

# Convert to DataFrame
probs_df = pd.DataFrame(results)

print("\n=== Per-clue Probabilities ===")
print(probs_df.head())


# Save to CSV
output_path = "clue_probabilities.csv"
probs_df.to_csv(output_path, index=False)
print(f"\nSaved per-clue probabilities to {output_path}")
