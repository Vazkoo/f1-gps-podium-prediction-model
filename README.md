# Miniproyecto F1 - Prediccion de podio con SVM

## Preparacion (una sola vez)
1. Descarga el dataset de Kaggle y descomprime los CSV dentro de `data/`
   (necesitas: results.csv, races.csv, drivers.csv, constructors.csv).
2. En la terminal de VS Code, dentro de esta carpeta:

```
python -m venv .venv
.venv\Scripts\activate          # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Ejecucion (en orden)
```
python 1_preparar_datos.py   # arma el dataset, crea el objetivo, EDA del desbalance
python 2_modelos.py          # entrena lineal vs RBF, metricas, graficas, PCA
python 3_demo.py             # demo interactiva por consola
```

Las graficas y tablas quedan en `outputs/` (listas para pegar en Canva).

## Diseño experimental (segun las notas de clase)
- Split temporal ("validacion hacia adelante"): train 1950-2012 | val 2013-2017 | prueba 2018-2024.
- Escalado dentro del Pipeline; C y gamma en grilla logaritmica con CV temporal (TimeSeriesSplit).
- Metrica que decide, fijada de antemano: PR-AUC (`average_precision`).
- Umbral y modelo final se eligen en validacion; la prueba se mide una sola vez.
- Modelos: {Lineal, RBF} x {sin ajuste, class_weight, SMOTE} + Lineal con embedding polinomial explicito.
- SMOTE requiere `imbalanced-learn` (esta en requirements.txt). Si no esta instalado, se omite con un aviso.
