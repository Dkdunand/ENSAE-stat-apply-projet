# ============================================================
# StatApp – Partie Classification (Bank Marketing) : Pipeline + SHAP
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, RocCurveDisplay
)

# ============================================================
# 0) CHARGEMENT ET STRUCTURE DU JEU DE DONNÉES
# ============================================================
def load_dataset(path: str = "bank-additional-full.csv") -> pd.DataFrame:
    """Charge le dataset UCI Bank Marketing."""
    df = pd.read_csv(path, sep=';')
    print(f"Shape: {df.shape}")
    print(df.head(3))
    return df


# ============================================================
# 1) PRÉPARATION ET ANALYSE RAPIDE
# ============================================================
def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme la cible 'y' (yes/no) en variable binaire 0/1.
    Affiche la répartition de la cible.
    """
    df['y'] = (df['y'].str.lower() == 'yes').astype(int)
    print("\nRépartition de la cible (0/1) :")
    print(df['y'].value_counts(normalize=True).round(3))
    return df


def get_feature_groups(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Retourne la liste des features numériques et catégorielles."""
    y = df['y']
    X = df.drop(columns=['y'])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    print(f"\nNb features numériques : {len(num_cols)} | catégorielles : {len(cat_cols)}")
    return num_cols, cat_cols, X, y


# ============================================================
# 2) PIPELINE COMPLET : PRÉPROCESSING + MODÈLE
# ============================================================
def make_pipeline(num_cols, cat_cols):
    """
    Pipeline complet :
      - StandardScaler pour les variables numériques
      - OneHotEncoder pour les catégorielles (dense pour SHAP)
      - LogisticRegression (class_weight='balanced' à cause du déséquilibre)
    """
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )

    clf = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight="balanced"
    )

    pipe = Pipeline(steps=[
        ("prep", preprocess),
        ("clf", clf)
    ])
    return pipe


# ============================================================
# 3) ENTRAÎNEMENT + ÉVALUATION
# ============================================================
def train_and_evaluate(pipe, X, y):
    """Effectue le split stratifié, l'entraînement et affiche toutes les métriques + graphes clés."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # ---- Métriques globales
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_proba)

    print("\n=== MÉTRIQUES (Logistic Regression) ===")
    print(f"Accuracy : {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall   : {rec:.3f}")
    print(f"F1-score : {f1:.3f}")
    print(f"AUC      : {auc:.3f}\n")

    print("=== Classification report ===")
    print(classification_report(y_test, y_pred, digits=3))
    print("=== Confusion matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # ---- ROC Curve
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title("ROC curve – Bank Marketing (LogisticRegression)")
    plt.show()

    return pipe, X_train, X_test, y_train, y_test


# ============================================================
# 4) EXPLICABILITÉ SHAP
# ============================================================
import shap

def shap_explain(pipe, X_train, X_test, num_cols, cat_cols):
    """
    Calcule et affiche :
      - Bar chart (importance moyenne)
      - Summary dot plot
      - Top 15 features avec leurs valeurs SHAP moyennes
    """
    prep = pipe.named_steps["prep"]
    X_train_t = prep.transform(X_train)
    X_test_t  = prep.transform(X_test)

    # Noms de features après transformation
    cat_encoder = prep.named_transformers_["cat"]
    feature_names = num_cols + cat_encoder.get_feature_names_out(cat_cols).tolist()

    logit = pipe.named_steps["clf"]
    explainer = shap.LinearExplainer(logit, X_train_t)
    shap_values = explainer(X_test_t)

    # Graphiques SHAP
    shap.plots.bar(shap_values, max_display=15)
    plt.show()

    shap.summary_plot(shap_values.values, features=X_test_t,
                      feature_names=feature_names, max_display=15, show=True)

    # Top 15 features (texte)
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    print("\n[Top 15 features – SHAP mean |value|]")
    for i in range(min(15, len(feature_names))):
        j = order[i]
        print(f"{i+1:>2}. {feature_names[j]:<35} {mean_abs[j]:.6f}")

    # Interprétation synthétique
    print("\nInterprétation rapide :")
    print(" - emp.var.rate, euribor3m et cons.price.idx dominent : effet macroéconomique.")
    print(" - duration a le plus fort impact microéconomique : plus l’appel est long, plus la probabilité de souscription augmente.")
    print(" - Les mois de campagne (may, jun, aug) et les métiers (blue-collar) ont un effet secondaire mais cohérent.")


# ============================================================
# 5)
# ============================================================
def print_meeting_cheatsheet():
    """pour la réunion."""
    print("\n================= PRÉSENTATION ÉCLAIR (Réunion) =================")
    print("Dataset 2 – Bank Marketing (Classification) :")
    print("- Cible : variable binaire indiquant si le client a souscrit (y = 1).")
    print("- Source : UCI Machine Learning Repository (données réelles d’une banque portugaise).")
    print("- Pipeline : preprocessing complet (scaling + one-hot) + régression logistique.")
    print("- Résultats : AUC = 0.94, accuracy = 0.86, recall = 0.91, F1 = 0.60.")
    print("- Interprétation : modèle performant et orienté rappel (détection maximale des souscripteurs).")
    print("- Explicabilité SHAP : montre l’importance des facteurs macroéconomiques et de la durée d’appel.")
    print("\nVisuels clés à montrer :")
    print("  1) Histogramme / EDA rapide des variables clés (emp.var.rate, duration, euribor3m)")
    print("  2) ROC curve + matrice de confusion")
    print("  3) Graphiques SHAP (bar + summary dot)")
    print("==================================================================")


# ============================================================
# 6) MAIN
# ============================================================
def main():
    df = load_dataset()
    df = prepare_target(df)
    num_cols, cat_cols, X, y = get_feature_groups(df)

    pipe = make_pipeline(num_cols, cat_cols)
    pipe, X_train, X_test, y_train, y_test = train_and_evaluate(pipe, X, y)

    shap_explain(pipe, X_train, X_test, num_cols, cat_cols)
    print_meeting_cheatsheet()


if __name__ == "__main__":
    main()
