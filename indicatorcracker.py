from glove import get_model
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


# Load Data

df = pd.read_csv("logistic_data.csv")


# GLOVE STUFF

model = get_model()


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
    ivec = get_vec(indicator.lower())
    sims = []
    for w in word_list:
        sims.append(cosine_sim(ivec, get_vec(w)))
    return float(np.mean(sims)) if sims else 0.0


# Feature Engineering

df["fodder_length"] = (
    df["fodder"]
    .astype(str)
    .str.replace(r"[^A-Za-z]", "", regex=True)
    .str.len()
)

# Number of words in fodder
df["fodder_word_count"] = df["fodder"].astype(str).str.split().apply(len)

# glove features
df["glove_anagram"] = df["indicator"].apply(
    lambda x: avg_similarity(str(x), ANAGRAM_WORDS))
df["glove_hidden"] = df["indicator"].apply(
    lambda x: avg_similarity(str(x), HIDDEN_WORDS))
df["glove_selector"] = df["indicator"].apply(
    lambda x: avg_similarity(str(x), SELECTOR_WORDS))


# feature matrix
X = df[[
    "length",
    "fodder_length",
    "fodder_word_count",
    "glove_anagram",
    "glove_hidden",
    "glove_selector"
]].values.astype(float)

y_raw = df["category"].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)


# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Train Logistic Regression

logreg = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=500
)

logreg.fit(X_train_scaled, y_train)


# Evaluation

y_pred = logreg.predict(X_test_scaled)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# Feature Extraction for New Inputs


def extract_features(clue, indicator, length, fodder, definition):

    fodder_clean_len = len("".join([c for c in str(fodder) if c.isalpha()]))

    return np.array([[
        float(length),
        float(fodder_clean_len),
        len(str(fodder).split()),
        avg_similarity(indicator, ANAGRAM_WORDS),
        avg_similarity(indicator, HIDDEN_WORDS),
        avg_similarity(indicator, SELECTOR_WORDS)
    ]])


# Prediction Function

def predict_category(clue, indicator, length, fodder, definition):
    x = extract_features(clue, indicator, length, fodder, definition)
    x_scaled = scaler.transform(x)

    pred_id = logreg.predict(x_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_id])[0]
    pred_probs = logreg.predict_proba(x_scaled)[0]

    return pred_label, pred_probs


# Interactive Mode

print("\n=== Test a New Unseen Clue ===")

clue = input("Enter the clue: ")
indicator = input("Enter the indicator: ")
fodder = input("Enter the fodder words: ")
definition = input("Enter the definition: ")

while True:
    try:
        length = int(input("Enter the solution length: "))
        break
    except:
        print("Length must be an integer, try again.")

pred_label, pred_probs = predict_category(
    clue, indicator, length, fodder, definition
)

print("\n=== Prediction Result ===")
print("Predicted Category:", pred_label)

print("\nProbabilities:")
for cat, p in zip(label_encoder.classes_, pred_probs):
    print(f"  {cat}: {p:.4f}")
