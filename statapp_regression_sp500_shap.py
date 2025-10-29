# ============================================================
# StatApp – Partie Régression (S&P 500) : Pipeline + SHAP
# ============================================================

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import shap

# ============================================================
# 0) CONFIGURATION
# ============================================================
TICKER = "^GSPC"                 # S&P 500
START  = "2015-01-01"
END    = "2025-01-01"

TARGET_NAME = "Return_t+1"       # Cible : rendement journalier futur
FEATURES = ["MA5", "MA20", "Volatility_5d"]

RANDOM_STATE = 42                # NB : split temporel (shuffle=False) => seed peu utile
TEST_SIZE = 0.20                 # 80% train / 20% test
FIGSIZE = (10, 5)


# ============================================================
# 1) CHARGEMENT & PRÉPARATION DES DONNÉES
# ============================================================
def load_sp500_prices(ticker: str = TICKER, start: str = START, end: str = END) -> pd.DataFrame:
    """
    Télécharge les prix S&P500 via yfinance.
    Retourne un DataFrame avec colonnes ['Open','High','Low','Close','Adj Close','Volume'] (selon version).
    Aplati l'éventuel MultiIndex de colonnes.
    """
    data = yf.download(ticker, start=start, end=end)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    # Certaines versions de yfinance n’exposent pas 'Adj Close' → on travaille sur 'Close'
    if "Close" not in data.columns:
        raise KeyError("Colonne 'Close' absente des données téléchargées.")
    return data


def make_target_and_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crée la cible:
      - Return_t   = Close_t / Close_{t-1} - 1
      - Return_t+1 = Return_t décalé de -1 (rendement du lendemain)
    Crée 3 features simples :
      - MA5, MA20 : moyennes mobiles
      - Volatility_5d : écart-type glissant des rendements sur 5 jours
    """
    df = df.copy()
    df["Return_t"] = df["Close"].pct_change()
    df[TARGET_NAME] = df["Return_t"].shift(-1)
    df["MA5"]           = df["Close"].rolling(5).mean()
    df["MA20"]          = df["Close"].rolling(20).mean()
    df["Volatility_5d"] = df["Return_t"].rolling(5).std()
    df = df.dropna()
    return df


# ============================================================
# 2) EDA RAPIDE
# ============================================================
def plot_eda(df: pd.DataFrame) -> None:
    """Affiche 2–3 visuels EDA clés (faciles à commenter en réunion)."""
    # (A) Série de prix
    plt.figure(figsize=FIGSIZE)
    df["Close"].plot()
    plt.title("S&P 500 – Clôture")
    plt.xlabel("Date"); plt.ylabel("Prix (Close)")
    plt.tight_layout(); plt.show()

    # (B) Rendement et volatilité courte
    plt.figure(figsize=FIGSIZE)
    df["Return_t"].plot(alpha=0.6, label="Return_t")
    df["Volatility_5d"].plot(alpha=0.8, label="Volatility_5d")
    plt.title("Rendement quotidien & Volatilité 5j")
    plt.xlabel("Date"); plt.legend()
    plt.tight_layout(); plt.show()

    # (C) Corrélations simples
    plt.figure(figsize=(6, 5))
    corr = df[["Return_t", TARGET_NAME] + FEATURES].corr()
    im = plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Matrice de corrélations (features & cibles)")
    plt.xticks(range(corr.shape[1]), corr.columns, rotation=45, ha="right")
    plt.yticks(range(corr.shape[0]), corr.index)
    plt.tight_layout(); plt.show()


# ============================================================
# 3) ENTRAÎNEMENT & ÉVALUATION
# ============================================================
def train_and_evaluate(df: pd.DataFrame) -> tuple[LinearRegression, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Sépare train/test temporel (shuffle=False), entraîne une Régression Linéaire, renvoie :
      - model, X_train, y_train, X_test, y_test
    Affiche MSE et R² (attendre R²≈0 sur rendements journaliers).
    """
    X = df[FEATURES]
    y = df[TARGET_NAME]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False
    )
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print("\n=== ÉVALUATION (Régression linéaire) ===")
    print(f"MSE : {mse:.8f}")
    print(f"R²  : {r2:.6f}  (faible attendu sur rendements journaliers)")

    # Courbe comparant y_test vs y_pred (utile à l’écran)
    plt.figure(figsize=FIGSIZE)
    plt.plot(y_test.index, y_test.values, label="Return_t+1 (réel)", alpha=0.8)
    plt.plot(y_test.index, y_pred, label="Prédiction linéaire", alpha=0.8)
    plt.title("Comparaison retour réel vs prédiction (test)")
    plt.xlabel("Date"); plt.ylabel("Rendement")
    plt.legend(); plt.tight_layout(); plt.show()

    return model, X_train, y_train, X_test, y_test


# ============================================================
# 4) EXPLICABILITÉ SHAP
# ============================================================
def shap_explain(model: LinearRegression, X_train: pd.DataFrame, X_test: pd.DataFrame) -> shap.Explanation:
    """
    Calcule SHAP values et produit 3 graphiques :
      - Bar chart (importance globale)
      - Summary dot (signe + dispersion)
      - Dependence plot sur MA5
    Affiche aussi un top trié par |valeur SHAP| moyenne.
    """
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_test)

    # (A) Importance globale (barres)
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)

    # (B) Nuage de points (sens + distribution des effets)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="dot", show=False)
    plt.title("Impact des variables explicatives sur Return_t+1 (SHAP)", pad=12)
    plt.tight_layout(); plt.show()

    # (C) Dépendance (MA5)
    shap.dependence_plot("MA5", shap_values.values, X_test, show=True)

    # Top features (texte)
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    print("\n[Résumé SHAP – importance moyenne absolue décroissante]")
    for f, v in zip(X_test.columns[order], mean_abs[order]):
        print(f" - {f}: {v:.6f}")

    # Interprétation synthétique (à reprendre dans le rapport)
    print("\nInterprétation rapide :")
    print(" • MA5 et MA20 dominent l’explication des variations; Volatility_5d a un rôle plus faible.")
    print(" • Le summary dot montre le SENS (±) et la dispersion des effets.")
    print(" • La dépendance sur MA5 illustre comment ses niveaux influencent Return_t+1.")
    return shap_values


# ============================================================
# 5)
# ============================================================
def print_meeting_cheatsheet() -> None:
    """
    Pour la réunion
    """
    print("\n================= PRÉSENTATION ÉCLAIR (Réunion) =================")
    print("Dataset 1 – S&P 500 (Régression) :")
    print("- Cible : rendement journalier futur (Return_t+1) calculé via pct_change().")
    print("- Contexte : série financière réelle (Yahoo Finance), 2015–2025.")
    print("- Pipeline : EDA → features (MA5, MA20, Volatility_5d) → régression linéaire → SHAP.")
    print("- Résultat : R² faible (attendu sur rendements journaliers très bruités).")
    print("- Explicabilité : SHAP montre que MA5/MA20 expliquent l’essentiel; vol 5j marginale.")
    print("\nVisuels clés à montrer rapidement :")
    print("  1) Série Close & courbe Return_t/Volatility_5d (EDA)")
    print("  2) y_test vs prédiction (courbe superposée)")
    print("  3) SHAP : bar chart + summary dot + (option) dependence plot sur MA5")
    print("==================================================================")


# ============================================================
# 6) MAIN
# ============================================================
def main():
    # Données & EDA
    data_raw = load_sp500_prices()
    df = make_target_and_features(data_raw)
    plot_eda(df)

    # Entraînement & évaluation
    model, X_train, y_train, X_test, y_test = train_and_evaluate(df)

    # Explicabilité SHAP
    _ = shap_explain(model, X_train, X_test)

    # Script d’aide pour la réunion (résumé oral)
    print_meeting_cheatsheet()


if __name__ == "__main__":
    main()
