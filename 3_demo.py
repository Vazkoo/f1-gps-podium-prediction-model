"""
PASO 3 - Demo por consola
-------------------------
Elige una carrera (anio y ronda) y el modelo predice quien llega al podio
usando SOLO informacion previa a la carrera. Compara con el resultado real.

Uso:  python 3_demo.py
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

MODELO = Path("outputs/modelo_final.joblib")
DATOS = Path("data/processed/dataset_modelado.csv")


def mostrar_carrera(df, paquete, anio, ronda):
    c = df[(df["year"] == anio) & (df["round"] == ronda)].copy()
    if c.empty:
        print("  No encontre esa carrera.")
        return
    c["puntaje"] = paquete["modelo"].decision_function(c[paquete["features"]].values)
    c = c.sort_values("puntaje", ascending=False).reset_index(drop=True)
    c["pred_umbral"] = c["puntaje"] >= paquete["umbral"]
    c["top3"] = c.index < 3

    nota = ""
    if anio <= 2012:
        nota = "  (OJO: carrera usada en el ENTRENAMIENTO)"
    elif anio <= 2017:
        nota = "  (carrera de VALIDACION)"
    print(f"\n{c['gp'].iloc[0]} {anio} - ronda {ronda}{nota}")
    print("-" * 78)
    print(f"{'#':>2}  {'Piloto':<22}{'Equipo':<18}{'Largo':>6}{'Puntaje':>9}  {'Predice':<8}{'Real':>5}")
    for i, f in c.iterrows():
        marca = "PODIO" if f["pred_umbral"] else ""
        real = int(f["positionOrder"])
        real_txt = f"{real}" + (" *" if f["podio"] == 1 else "")
        print(f"{i + 1:>2}  {f['piloto'][:21]:<22}{f['equipo'][:17]:<18}{int(f['grid']):>6}"
              f"{f['puntaje']:>9.2f}  {marca:<8}{real_txt:>5}")
    print("-" * 78)
    reales = set(c.loc[c["podio"] == 1, "piloto"])
    aciertos_top3 = len(reales & set(c.loc[c["top3"], "piloto"]))
    aciertos_umbral = len(reales & set(c.loc[c["pred_umbral"], "piloto"]))
    print(f"Podio real (*): {', '.join(sorted(reales))}")
    print(f"Con umbral ({paquete['umbral']:.2f}): marco {int(c['pred_umbral'].sum())} pilotos, acerto {aciertos_umbral}/3")
    print(f"Top-3 por puntaje: acerto {aciertos_top3}/3")


def main():
    if not MODELO.exists():
        sys.exit("Falta outputs/modelo_final.joblib. Corre primero 1_preparar_datos.py y 2_modelos.py")
    df = pd.read_csv(DATOS)
    paquete = joblib.load(MODELO)
    print(f"=== PREDICTOR DE PODIO F1 | {paquete['nombre']} ===")
    print("Escribe 'q' para salir. Carreras de prueba: 2018-2024.")
    while True:
        a = input("\nAnio (ej. 2023): ").strip()
        if a.lower() == "q":
            break
        if not a.isdigit() or int(a) not in set(df["year"]):
            print("  Anio no valido.")
            continue
        anio = int(a)
        rondas = df[df["year"] == anio].drop_duplicates("round").sort_values("round")
        print("  Rondas: " + " | ".join(f"{r.round}:{r.gp.replace(' Grand Prix', '')}" for r in rondas.itertuples()))
        r = input("Ronda: ").strip()
        if r.lower() == "q":
            break
        if r.isdigit():
            mostrar_carrera(df, paquete, anio, int(r))


if __name__ == "__main__":
    main()
