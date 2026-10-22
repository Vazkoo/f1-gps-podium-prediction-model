"""
PASO 2 - Modelos SVM
------------------------------------------------------
Se comparan dos nucleos (lineal y RBF) x tres tratamientos del desbalance:
    - sin ajuste
    - class_weight='balanced'            (visto en clase)
    - SMOTE (sobremuestreo sintetico)    (NO visto en clase -> la "herramienta nueva")
y como extra:
    - Lineal + embedding polinomial explicito (grado 2): el "embedding de expansion"
      de clase, para contrastarlo con el truco del kernel.

Reglas de clase que se respetan:
  * Estandarizar SIEMPRE y dentro del Pipeline (sin filtrar la media de prueba).
  * C y gamma se buscan JUNTOS en grilla logaritmica con validacion cruzada.
  * La metrica que decide se fija ANTES: average_precision (PR-AUC).
  * El umbral y el modelo se eligen SIN mirar la prueba. La prueba se mide UNA vez.
  * Particion temporal ("validacion hacia adelante"), no aleatoria.

Division: entrenamiento <= 2012 | validacion 2013-2017 | prueba 2018-2024
"""
import time
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC, LinearSVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# SMOTE vive en imbalanced-learn (pip install imbalanced-learn). Si no esta, se omite.
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    HAY_SMOTE = True
except ImportError:
    from sklearn.pipeline import Pipeline
    HAY_SMOTE = False

FEATURES = [
    "grid_ajustado", "grid_rel", "n_autos", "edad", "piloto_podio_5",
    "piloto_pos_5", "piloto_podio_carrera", "piloto_n_carreras", "equipo_podio_10",
]
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)
SEMILLA = 42

GRID_LINEAL = {"svm__C": [0.01, 0.1, 1, 10, 100]}
GRID_RBF = {"svm__C": [0.1, 1, 10, 100], "svm__gamma": [1e-3, 1e-2, 1e-1, "scale"]}


# ---------------------------------------------------------------- utilidades
def construir(kernel, trat):
    """Pipeline: escalado -> (SMOTE) -> SVM. SMOTE solo actua al entrenar."""
    cw = "balanced" if trat == "class_weight" else None
    if kernel == "lineal":
        svm = LinearSVC(class_weight=cw, dual="auto", max_iter=20000)
    else:
        svm = SVC(kernel="rbf", class_weight=cw)
    pasos = [("sc", StandardScaler())]
    if trat == "smote":
        pasos.append(("smote", SMOTE(sampling_strategy=0.5, random_state=SEMILLA)))
    pasos.append(("svm", svm))
    return Pipeline(pasos)


def construir_poly():
    return Pipeline([
        ("sc", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("sc2", StandardScaler()),
        ("svm", LinearSVC(class_weight="balanced", dual="auto", max_iter=20000)),
    ])


def mejor_umbral(y, score):
    """Umbral que maximiza F1 sobre la curva precision-recall (se elige en VALIDACION)."""
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(t[np.argmax(f1[:-1])])


def en_umbral(y, score, u):
    pred = (score >= u).astype(int)
    return {
        "F1": f1_score(y, pred, zero_division=0),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "MCC": matthews_corrcoef(y, pred),
        "BalAcc": balanced_accuracy_score(y, pred),
        "Accuracy": accuracy_score(y, pred),
    }


# -------------------------------------------------------------------- main
def main():
    df = pd.read_csv("data/processed/dataset_modelado.csv")
    train = df[df["year"] <= 2012]
    val = df[(df["year"] >= 2013) & (df["year"] <= 2017)]
    test = df[df["year"] >= 2018]
    Xtr, ytr = train[FEATURES].values, train["podio"].values
    Xva, yva = val[FEATURES].values, val["podio"].values
    Xte, yte = test[FEATURES].values, test["podio"].values

    print(f"Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    print(f"Tasa de podio -> train {ytr.mean():.1%} | val {yva.mean():.1%} | test {yte.mean():.1%}")
    print(f"Un modelo que SIEMPRE dice 'no podio' tendria accuracy {1 - yte.mean():.1%} en prueba "
          f"(y F1 = 0). PR-AUC de un azar = {yte.mean():.3f}\n")

    # Submuestra (ordenada en el tiempo) para buscar C y gamma del RBF sin esperar horas
    sub = train.sample(n=min(6000, len(train)), random_state=SEMILLA).sort_index()
    Xsub, ysub = sub[FEATURES].values, sub["podio"].values
    cv = TimeSeriesSplit(n_splits=3)  # validacion cruzada hacia adelante

    configs = [
        ("Lineal | sin ajuste",    "lineal", "ninguno"),
        ("Lineal | class_weight",  "lineal", "class_weight"),
        ("Lineal | SMOTE",         "lineal", "smote"),
        ("RBF | sin ajuste",       "rbf",    "ninguno"),
        ("RBF | class_weight",     "rbf",    "class_weight"),
        ("RBF | SMOTE",            "rbf",    "smote"),
        ("Lineal + poly2 (embedding explicito) | class_weight", "poly", "class_weight"),
    ]

    if not HAY_SMOTE:
        print("AVISO: imbalanced-learn no esta instalado -> se omiten los modelos SMOTE.\n"
              "       Instala con: pip install imbalanced-learn\n")
        configs = [c for c in configs if c[2] != "smote"]

    filas, scores_test, entrenados = [], {}, {}
    for nombre, kernel, trat in configs:
        t0 = time.time()
        print(f"[{nombre}] buscando hiperparametros (CV temporal, scoring=average_precision)...")
        pipe = construir_poly() if kernel == "poly" else construir(kernel, trat)
        grid = GRID_RBF if kernel == "rbf" else GRID_LINEAL
        Xb, yb = (Xsub, ysub) if kernel == "rbf" else (Xtr, ytr)
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="average_precision", n_jobs=-1, refit=False).fit(Xb, yb)
        modelo = clone(pipe).set_params(**gs.best_params_).fit(Xtr, ytr)  # ajuste final en TODO el train

        s_val, s_te = modelo.decision_function(Xva), modelo.decision_function(Xte)
        u = mejor_umbral(yva, s_val)
        m0, mu = en_umbral(yte, s_te, 0.0), en_umbral(yte, s_te, u)
        svm = modelo.named_steps["svm"]
        n_sv = int(svm.n_support_.sum()) if hasattr(svm, "n_support_") else None
        filas.append({
            "modelo": nombre,
            "params": str({k.replace("svm__", ""): v for k, v in gs.best_params_.items()}),
            "PR_AUC_val": average_precision_score(yva, s_val),
            "PR_AUC_test": average_precision_score(yte, s_te),
            "F1_umbral0": m0["F1"], "Recall_umbral0": m0["Recall"],
            "umbral": u, **{k: mu[k] for k in ["F1", "Precision", "Recall", "MCC", "BalAcc", "Accuracy"]},
            "n_vectores_soporte": n_sv,
        })
        scores_test[nombre], entrenados[nombre] = s_te, (modelo, u, kernel)
        print(f"    mejor: {gs.best_params_}  | PR-AUC val {filas[-1]['PR_AUC_val']:.3f} | {time.time() - t0:.0f} s")

    tabla = pd.DataFrame(filas)
    tabla.round(4).to_csv(OUT / "resultados.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n" + "=" * 100)
    print("RESULTADOS EN PRUEBA 2018-2024 (el umbral se eligio en validacion, nunca en prueba)")
    print("=" * 100)
    print(tabla.drop(columns=["params"]).round(3).to_string(index=False))

    # ---- Modelo final: se elige por PR-AUC en VALIDACION (no en prueba) -----
    candidatos = tabla[tabla["modelo"].str.startswith("RBF")]
    final = candidatos.loc[candidatos["PR_AUC_val"].idxmax(), "modelo"]
    modelo_f, u_f, _ = entrenados[final]
    print(f"\nModelo final para la demo (mejor RBF por PR-AUC en validacion): {final}")
    joblib.dump({"modelo": modelo_f, "umbral": u_f, "features": FEATURES, "nombre": final}, OUT / "modelo_final.joblib")

    # ---- Grafica 1: curvas precision-recall ---------------------------------
    plt.figure(figsize=(8, 5.5))
    for nombre, sc in scores_test.items():
        p, r, _ = precision_recall_curve(yte, sc)
        plt.plot(r, p, lw=1.5, label=f"{nombre} (AP={average_precision_score(yte, sc):.3f})")
    plt.axhline(yte.mean(), ls="--", color="gray", label=f"Azar ({yte.mean():.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Curvas Precision-Recall (prueba 2018-2024)")
    plt.legend(fontsize=7); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(OUT / "curvas_pr.png", dpi=150); plt.close()

    # ---- Grafica 2: matrices de confusion (como y por que falla) -------------
    s_base = scores_test["Lineal | sin ajuste"]
    s_fin = scores_test[final]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ConfusionMatrixDisplay.from_predictions(yte, (s_base >= 0).astype(int), ax=ax[0], colorbar=False,
                                            display_labels=["No podio", "Podio"])
    ax[0].set_title("Lineal sin ajuste (umbral 0)")
    ConfusionMatrixDisplay.from_predictions(yte, (s_fin >= u_f).astype(int), ax=ax[1], colorbar=False,
                                            display_labels=["No podio", "Podio"])
    ax[1].set_title(f"{final}\n(umbral ajustado)", fontsize=9)
    plt.tight_layout(); plt.savefig(OUT / "matrices_confusion.png", dpi=150); plt.close()

    # ---- Grafica 3: PCA 2D (Unidad 4) ----------------------------------------
    sc = StandardScaler().fit(Xtr)
    pca_full = PCA().fit(sc.transform(Xtr))
    pca = PCA(n_components=2).fit(sc.transform(Xtr))
    muestra = train.sample(n=min(6000, len(train)), random_state=SEMILLA)
    Z = pca.transform(sc.transform(muestra[FEATURES].values))
    plt.figure(figsize=(7, 5))
    for clase, color, et in [(0, "#b0b0b0", "No podio"), (1, "#e10600", "Podio")]:
        m = muestra["podio"].values == clase
        plt.scatter(Z[m, 0], Z[m, 1], s=6, alpha=.4, c=color, label=et)
    plt.title(f"PCA 2D ({pca.explained_variance_ratio_.sum():.0%} de la varianza)")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})"); plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    plt.legend(markerscale=3); plt.tight_layout(); plt.savefig(OUT / "pca_2d.png", dpi=150); plt.close()
    print("\nPCA - varianza explicada por componente (autovalor_i / suma de autovalores):")
    print("  " + "  ".join(f"PC{i + 1}: {v:.1%}" for i, v in enumerate(pca_full.explained_variance_ratio_)))
    print("\nGuardado en outputs/: resultados.csv, modelo_final.joblib y 3 graficas (.png)")


if __name__ == "__main__":
    main()
