"""
PASO 1 - Preparar los datos
---------------------------
Lee los CSV de Kaggle (carpeta data/), arma UNA tabla con una fila por
piloto y carrera, crea la variable objetivo (podio) y las caracteristicas
que se conocen ANTES de la carrera (sin fuga de informacion).

Salida: data/processed/dataset_modelado.csv  +  graficas de EDA en outputs/
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = Path("data")
PROC = DATA / "processed"
OUT = Path("outputs")
PROC.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "grid_ajustado",         # posicion de largada (0 = pit lane -> ultimo)
    "grid_rel",              # largada / numero de autos
    "n_autos",               # tamano de la parrilla
    "edad",                  # edad del piloto
    "piloto_podio_5",        # % podios del piloto en sus ultimas 5 carreras
    "piloto_pos_5",          # posicion media en sus ultimas 5 carreras
    "piloto_podio_carrera",  # % podios historico del piloto
    "piloto_n_carreras",     # experiencia (carreras previas)
    "equipo_podio_10",       # % podios del equipo en sus ultimas 10 carreras
]


def leer(nombre):
    # En estos CSV los nulos vienen como \N
    return pd.read_csv(DATA / nombre, na_values=["\\N"])


def main():
    results = leer("results.csv")
    races = leer("races.csv")
    drivers = leer("drivers.csv")
    constructors = leer("constructors.csv")

    # ---- 1. Unir tablas -------------------------------------------------
    res = (
        results[["raceId", "driverId", "constructorId", "grid", "positionOrder"]]
        .merge(
            races[["raceId", "year", "round", "name", "date"]].rename(columns={"name": "gp"}),
            on="raceId",
        )
        .merge(drivers[["driverId", "forename", "surname", "dob"]], on="driverId")
        .merge(
            constructors[["constructorId", "name"]].rename(columns={"name": "equipo"}),
            on="constructorId",
        )
    )
    res["date"] = pd.to_datetime(res["date"])
    res["dob"] = pd.to_datetime(res["dob"])
    res["piloto"] = res["forename"] + " " + res["surname"]
    res = res.sort_values(["date", "raceId", "positionOrder"]).reset_index(drop=True)

    # ---- 2. Variable objetivo ------------------------------------------
    # positionOrder siempre existe (incluso para quien abandona)
    res["podio"] = (res["positionOrder"] <= 3).astype(int)

    # ---- 3. Caracteristicas pre-carrera --------------------------------
    # Parrilla
    res["n_autos"] = res.groupby("raceId")["driverId"].transform("count")
    max_grid = res.groupby("raceId")["grid"].transform("max")
    res["grid_ajustado"] = np.where(res["grid"] == 0, max_grid + 1, res["grid"])
    res["grid_rel"] = res["grid_ajustado"] / res["n_autos"]
    res["edad"] = (res["date"] - res["dob"]).dt.days / 365.25

    # Forma del piloto. shift(1) = SOLO carreras anteriores (evita fuga)
    g = res.groupby("driverId")
    res["piloto_podio_5"] = g["podio"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    res["piloto_pos_5"] = g["positionOrder"].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    res["piloto_podio_carrera"] = g["podio"].transform(lambda s: s.shift(1).expanding().mean())
    res["piloto_n_carreras"] = g.cumcount()

    # Forma del equipo (promedio de podios por carrera, ultimas 10 carreras)
    eq = (
        res.groupby(["constructorId", "raceId"])
        .agg(podio_eq=("podio", "mean"), date=("date", "first"))
        .reset_index()
        .sort_values(["date", "raceId"])
    )
    eq["equipo_podio_10"] = eq.groupby("constructorId")["podio_eq"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )
    res = res.merge(eq[["constructorId", "raceId", "equipo_podio_10"]], on=["constructorId", "raceId"], how="left")

    # Debutantes: sin historial -> podio 0, posicion media = mitad de la parrilla
    for c in ["piloto_podio_5", "piloto_podio_carrera", "equipo_podio_10"]:
        res[c] = res[c].fillna(0.0)
    res["piloto_pos_5"] = res["piloto_pos_5"].fillna(res["n_autos"] / 2)

    res = res.sort_values(["date", "raceId", "positionOrder"]).reset_index(drop=True)
    columnas = ["raceId", "year", "round", "gp", "piloto", "equipo", "grid", "positionOrder", "podio"] + FEATURES
    res[columnas].to_csv(PROC / "dataset_modelado.csv", index=False)

    # ---- 4. EDA del desbalance -----------------------------------------
    tasa = res["podio"].mean()
    print("=" * 60)
    print(f"Filas (piloto-carrera): {len(res):,}")
    print(f"Carreras:               {res['raceId'].nunique():,}  ({res['year'].min()}-{res['year'].max()})")
    print(f"Podios (clase 1):       {res['podio'].sum():,}  ->  {tasa:.1%}")
    print(f"No podios (clase 0):    {(1 - res['podio']).sum():,.0f}  ->  {1 - tasa:.1%}")
    print(f"Razon de desbalance:    {(1 - tasa) / tasa:.1f} : 1")
    print(f"Un clasificador que SIEMPRE dice 'no podio' acierta {1 - tasa:.1%} (accuracy enganosa)")
    print("=" * 60)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    por_grid = res[res["grid_ajustado"] <= 20].groupby("grid_ajustado")["podio"].mean()
    ax[0].bar(por_grid.index, por_grid.values, color="#e10600")
    ax[0].set(title="Tasa de podio segun posicion de largada", xlabel="Posicion de largada", ylabel="% de podios")
    ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    por_decada = res.groupby((res["year"] // 10) * 10)["podio"].mean()
    ax[1].bar([f"{d}s" for d in por_decada.index], por_decada.values, color="#15151e")
    ax[1].set(title="Tasa de podio por decada", ylabel="% de podios")
    ax[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    plt.tight_layout()
    plt.savefig(OUT / "eda_desbalance.png", dpi=150)
    plt.close()
    print("Guardado: data/processed/dataset_modelado.csv y outputs/eda_desbalance.png")


if __name__ == "__main__":
    main()
