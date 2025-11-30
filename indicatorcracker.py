import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from word2vec import get_model

# import data
df = pd.read_csv("logistic_data.csv")





# FEATURES HERE!!!!

## ====== WORD2VEC INDICATOR FEATURES ==========================

model = get_model()

# baseline: can add those lists of words later too

ANAGRAM_WORDS = ["erupt", "shake", "mix", "confuse", "stir"]
REVERSAL_WORDS = ["back", "reverse", "turned", "over", "around"]
HIDDEN_WORDS = ["inside", "within", "amid", "concealed"]
SELECTOR_WORDS = ["take", "odd", "even", "end", "middle", "first", "second", "third", "fourth"]

def avg_similarity(word, word_list):
    sims = []
    for w in word_list:
        if word in model and w in model:
            sims.append(model.similarity(word, w))
    if sims:
        return np.mean(sims) 
    else:
        return 0

df["sim_anagram"] = df["indicator"].apply(lambda x: avg_similarity(str(x).lower(), ANAGRAM_WORDS))
df["sim_reversal"] = df["indicator"].apply(lambda x: avg_similarity(str(x).lower(), REVERSAL_WORDS))
df["sim_hidden"] = df["indicator"].apply(lambda x: avg_similarity(str(x).lower(), HIDDEN_WORDS))
df["sim_selector"] = df["indicator"].apply(lambda x: avg_similarity(str(x).lower(), SELECTOR_WORDS))


# Fodder Length
## NEW

df['fodder_length'] = df['fodder'].apply(lambda s: len(s) if isinstance(s, str) else 0)

df["fodder_length"] = (df["fodder"]
    .astype(str)
    .str.replace(r"[^A-Za-z]", "", regex=True) 
    .str.len()
)


# Update all features here!
X = df[[
    "length",
    "fodder_length",
    "sim_anagram",
    "sim_reversal",
    "sim_hidden",
    "sim_selector"
]].values.astype(float)

# label categories
y_raw = df["category"].values

# encode label for model
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.4,
    stratify=y,
    random_state=42
)
# scale it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# train the model classifier
logreg = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=500
)

logreg.fit(X_train_scaled, y_train)

# evaluate the performance

y_pred = logreg.predict(X_test_scaled)

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))

# have it spit out the categrory for solver


def predict_category_from_length(L: float, F: float):
    """
    Predict the cryptic clue category using the length and Fodder Length feature.
    L = answer length (float or int)
    F = Fodder length (Float)
    """
    x = np.array([[float(L), float(F)]])
    x_scaled = scaler.transform(x)

    pred_id = logreg.predict(x_scaled)[0]
    pred_label = label_encoder.inverse_transform([pred_id])[0]
    pred_probs = logreg.predict_proba(x_scaled)[0]

    return pred_label, pred_probs
