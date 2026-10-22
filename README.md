# Formula 1 Podium Predictor — Mini-Proyecto 3 con SVM

**Métodos y Modelos Matemáticos para IA**  
**Pontificia Universidad Javeriana Cali**

**Integrantes:** Nicolas Carrillo · Juan Vasco · Sebastian Silva

---

## 1. ¿Qué hace este proyecto?

Este mini-proyecto construye un clasificador binario para responder una pregunta concreta antes de cada carrera de Fórmula 1:

> **¿Este piloto terminará o no terminará en el podio?**

La variable objetivo se define a partir de la posición final oficial:

- `podio = 1` si `positionOrder <= 3`.
- `podio = 0` en cualquier otro caso.

La solución utiliza **Máquinas de Soporte Vectorial (SVM)** y compara una frontera lineal con una frontera no lineal mediante kernel **RBF**. Como el podio es una clase minoritaria, el proyecto no se guía por accuracy como métrica principal: el criterio central es **PR-AUC / Average Precision**, complementado con **F1, precision, recall, MCC, balanced accuracy y matriz de confusión**.

El proyecto también compara distintas estrategias para manejar el desbalance, ajusta el umbral de decisión usando únicamente validación y ofrece una demo interactiva por consola para probar carreras reales.

---

## 2. Dataset

Se utiliza el dataset **Formula 1 World Championship**, disponible en Kaggle:

https://www.kaggle.com/datasets/rohanrao/formula-1-world-championship-1950-2020/code

El conjunto reúne información histórica del campeonato y se actualiza con temporadas posteriores a las que aparecen en el nombre original del enlace. En esta versión del proyecto se trabajan registros entre **1950 y 2024**.

Para el pipeline principal se utilizan cuatro archivos:

- `results.csv`
- `races.csv`
- `drivers.csv`
- `constructors.csv`

El ZIP del dataset contiene otros CSV de Fórmula 1, pero estos cuatro son suficientes para construir las variables que usa el modelo actual.

### Resumen del dataset modelado

Después de integrar las tablas y construir las características se obtienen:

| Elemento | Valor |
|---|---:|
| Registros piloto-carrera | **26.759** |
| Carreras | **1.125** |
| Periodo | **1950–2024** |
| Podios | **3.397 (12,7%)** |
| No podios | **23.362 (87,3%)** |
| Razón aproximada de desbalance | **6,9 : 1** |

Este desbalance es importante: un clasificador que siempre predijera **“no podio”** alcanzaría aproximadamente **87,3% de accuracy**, pero tendría **F1 = 0 para la clase podio**. Por esa razón la accuracy no se utiliza como criterio principal de selección.

---

## 3. Unidad de observación y momento de la predicción

Cada fila representa una **participación de un piloto en una carrera**.

El objetivo es simular una predicción hecha **antes de que comience la carrera**, por lo que las variables de entrada se limitan a información conocida previamente: posición de salida, tamaño de parrilla, edad e historial anterior del piloto y del equipo.

No se utilizan como características datos que solo se conocen después de la carrera, como:

- posición final como predictor;
- puntos obtenidos;
- vueltas completadas;
- tiempos finales;
- resultado de la carrera actual para construir la forma histórica.

`positionOrder` se conserva únicamente para construir la etiqueta real y para mostrar el resultado durante la demo.

---

## 4. Variables utilizadas

El modelo trabaja con **9 características**:

| Variable | Significado |
|---|---|
| `grid_ajustado` | Posición de largada. Si `grid = 0`, se interpreta como salida desde pit lane y se ubica después de la última posición de parrilla registrada. |
| `grid_rel` | Posición de largada relativa al tamaño de la parrilla. |
| `n_autos` | Número de participantes registrados en la carrera. |
| `edad` | Edad del piloto en la fecha de la carrera. |
| `piloto_podio_5` | Proporción de podios del piloto en sus últimas 5 participaciones anteriores. |
| `piloto_pos_5` | Posición final media del piloto en sus últimas 5 participaciones anteriores. |
| `piloto_podio_carrera` | Proporción histórica de podios del piloto antes de la carrera actual. |
| `piloto_n_carreras` | Número de participaciones anteriores del piloto. |
| `equipo_podio_10` | Proporción de podios del constructor en sus últimas 10 carreras anteriores. |

### Pilotos debutantes

Cuando un piloto todavía no tiene historial disponible:

- sus tasas históricas de podio se inicializan en `0`;
- `piloto_pos_5` se inicializa en la mitad de la parrilla.

Esto permite conservar a los debutantes sin usar información futura.

---

## 5. Cómo evitamos fuga de información

La prevención de **data leakage** es una parte central del diseño.

### Historial solo con carreras anteriores

Las variables de forma utilizan `shift(1)` antes de calcular ventanas móviles o promedios acumulados. Por ejemplo:

```python
res["piloto_podio_5"] = g["podio"].transform(
    lambda s: s.shift(1).rolling(5, min_periods=1).mean()
)
```

El `shift(1)` hace que el resultado de la carrera que se quiere predecir no entre en sus propias variables históricas.

### Escalado dentro del Pipeline

`StandardScaler` se encuentra dentro del `Pipeline` de cada modelo. Así, las medias y desviaciones utilizadas para estandarizar se aprenden únicamente con los datos que corresponden al entrenamiento de cada ajuste.

### El test no decide hiperparámetros ni umbral

Los hiperparámetros se buscan con datos de entrenamiento y el umbral se selecciona en validación. El conjunto de prueba se reserva para la evaluación final.

---

## 6. División temporal de los datos

No se utiliza una partición aleatoria porque el problema es histórico y deportivo. Queremos aproximarnos a la situación real de **aprender del pasado y predecir temporadas futuras**.

| Conjunto | Años | Registros | Podios |
|---|---:|---:|---:|
| Entrenamiento | 1950–2012 | **21.715** | **12,2%** |
| Validación | 2013–2017 | **2.065** | **14,2%** |
| Prueba | 2018–2024 | **2.979** | **15,0%** |

Dentro del entrenamiento se utiliza `TimeSeriesSplit(n_splits=3)` para la búsqueda de hiperparámetros.

Esto evita mezclar artificialmente temporadas futuras con temporadas pasadas durante la evaluación.

---

## 7. Fundamento del modelo SVM

Una SVM lineal busca una frontera de decisión de la forma:

$$
f(x) = w^T x + b
$$

La clase se determina a partir del signo de `f(x)`.

La idea geométrica es elegir el hiperplano que deja el **mayor margen posible** entre las dos clases. En la normalización canónica, el ancho total del margen es:

$$
\frac{2}{\lVert w \rVert}
$$

Por eso maximizar el margen equivale a controlar la norma de `w`.

En datos reales las clases no son perfectamente separables, así que se utiliza **margen blando**. El hiperparámetro `C` regula el compromiso entre:

- permitir ciertas violaciones para obtener una frontera más regular;
- penalizar con mayor fuerza los errores de clasificación.

### Kernel RBF

Además del modelo lineal se prueba una SVM con kernel RBF:

$$
K(x,z)=\exp(-\gamma\lVert x-z\rVert^2)
$$

El kernel permite modelar fronteras no lineales sin construir explícitamente todas las dimensiones del espacio transformado.

`C` y `gamma` se buscan conjuntamente mediante validación cruzada, porque ambos modifican de forma importante la complejidad de la frontera.

---

## 8. Modelos comparados

El experimento compara:

1. **SVM lineal sin tratamiento del desbalance**.
2. **SVM lineal con `class_weight="balanced"`**.
3. **SVM lineal con SMOTE**.
4. **SVM RBF sin tratamiento del desbalance**.
5. **SVM RBF con `class_weight="balanced"`**.
6. **SVM RBF con SMOTE**.
7. **SVM lineal con embedding polinomial explícito de grado 2 + `class_weight="balanced"`**.

El modelo polinomial explícito se incluye para contrastar dos ideas:

- transformar explícitamente las características y luego aplicar una frontera lineal;
- utilizar el truco del kernel para trabajar con relaciones no lineales sin materializar todo el espacio transformado.

### Nota sobre SMOTE

SMOTE es una **extensión adicional del proyecto** y no forma parte del núcleo teórico trabajado en la clase de SVM. Se utiliza como comparación frente a `class_weight="balanced"`.

En este proyecto:

```python
SMOTE(sampling_strategy=0.5, random_state=42)
```

se ejecuta dentro del pipeline y actúa únicamente durante el entrenamiento.

---

## 9. Búsqueda de hiperparámetros

La métrica utilizada en `GridSearchCV` es:

```python
scoring="average_precision"
```

Las grillas son:

### SVM lineal

```python
C = [0.01, 0.1, 1, 10, 100]
```

### SVM RBF

```python
C = [0.1, 1, 10, 100]
gamma = [1e-3, 1e-2, 1e-1, "scale"]
```

Para reducir el costo de la búsqueda RBF, la grilla se evalúa sobre una submuestra reproducible de hasta 6.000 registros del conjunto de entrenamiento. Una vez seleccionados `C` y `gamma`, el modelo se vuelve a ajustar utilizando **todo el conjunto de entrenamiento**.

---

## 10. ¿Por qué PR-AUC y F1?

La clase positiva representa solo alrededor del 13% de los registros. En este contexto, accuracy puede dar una impresión engañosa de buen desempeño.

### PR-AUC / Average Precision

Es la métrica principal para elegir el modelo y los hiperparámetros porque resume el compromiso entre:

- **precision**: de los pilotos señalados como candidatos a podio, cuántos realmente terminaron en el podio;
- **recall**: de los pilotos que realmente subieron al podio, cuántos detectó el modelo.

Para una clasificación aleatoria, el nivel de referencia de PR-AUC está aproximadamente en la prevalencia de la clase positiva. En el conjunto total esa prevalencia es cerca de **0,127**.

### F1

Después de entrenar el modelo, el umbral de decisión se ajusta en el conjunto de validación buscando maximizar:

$$
F1 = 2\frac{Precision\cdot Recall}{Precision+Recall}
$$

El umbral final **no se selecciona mirando el test**.

---

## 11. Resultados obtenidos

Los siguientes resultados corresponden a la ejecución actual del proyecto. La selección del modelo se hace utilizando **PR-AUC en validación**; las métricas de prueba se reportan después.

| Modelo | PR-AUC val. | PR-AUC test | F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Lineal · sin ajuste | 0,7182 | 0,6959 | 0,6850 | 0,5934 | 0,8098 |
| Lineal · class weight | 0,7253 | 0,7039 | 0,6950 | 0,6234 | 0,7852 |
| **Lineal · SMOTE** | 0,7224 | 0,7031 | **0,6972** | 0,6172 | 0,8009 |
| RBF · sin ajuste | 0,7045 | 0,6785 | 0,6762 | 0,5875 | 0,7964 |
| **RBF · class weight** | **0,7297** | 0,7049 | 0,6876 | **0,6273** | 0,7606 |
| RBF · SMOTE | 0,7287 | **0,7063** | 0,6949 | 0,6204 | 0,7897 |
| Lineal + poly2 · class weight | 0,7212 | 0,7013 | 0,6898 | 0,6341 | 0,7562 |

### Modelo utilizado en la demo

La solución guarda como modelo final:

> **RBF + `class_weight="balanced"`**

porque fue el **RBF con mayor PR-AUC en validación** y, en esta ejecución, también obtuvo el mayor PR-AUC de validación entre todos los modelos comparados:

$$
PR\text{-}AUC_{val}=0.7297
$$

Sus hiperparámetros seleccionados fueron:

```text
C = 1
gamma = 0.001
```

El umbral F1 elegido exclusivamente en validación fue:

```text
0.9236
```

En prueba 2018–2024 obtuvo:

| Métrica | Resultado |
|---|---:|
| PR-AUC / AP | **0,7049** |
| F1 | **0,6876** |
| Precision | **0,6273** |
| Recall | **0,7606** |
| MCC | **0,6302** |
| Balanced Accuracy | **0,8404** |
| Accuracy | **0,8963** |
| Vectores de soporte | **9.482** |

### Lectura importante de los resultados

El kernel RBF no supera por una distancia enorme a las alternativas lineales. Eso es un resultado relevante: las características históricas construidas ya contienen una señal bastante separable y el modelo lineal resulta competitivo.

También se observa que **el modelo con mayor F1 de test no es necesariamente el mismo que se selecciona por PR-AUC de validación**. Esto es intencional: no usamos el conjunto de prueba para cambiar de modelo después de conocer sus resultados.

---

## 12. Umbral de decisión

Una SVM produce un puntaje mediante:

```python
decision_function(X)
```

El umbral estándar sería `0`, pero el proyecto permite mover esa frontera de decisión para equilibrar precision y recall.

El procedimiento es:

1. entrenar usando únicamente `train`;
2. obtener los puntajes en `validation`;
3. recorrer la curva Precision-Recall;
4. elegir el umbral que maximiza F1 en validación;
5. congelar ese umbral;
6. medir una sola vez sobre `test`.

Esto modifica la **política de decisión**, no vuelve a entrenar la SVM.

---

## 13. PCA: para visualizar, no para entrenar el modelo final

El script también genera una proyección con PCA a dos dimensiones.

Su propósito es **exploratorio y visual**: observar cuánto se superponen las clases cuando las nueve características estandarizadas se proyectan sobre dos componentes principales.

PCA **no forma parte del pipeline del modelo final** y no se utiliza para seleccionar el ganador.

La gráfica queda guardada en:

```text
outputs/pca_2d.png
```

---

## 14. Demo interactiva

`3_demo.py` permite elegir un año y una ronda y observar la predicción de cada piloto.

Para cada carrera se muestra:

- piloto;
- equipo;
- posición de largada;
- puntaje de `decision_function`;
- clasificación binaria usando el umbral aprendido;
- posición final real;
- podio real;
- número de integrantes del podio detectados.

Además se presenta un **Top-3 por puntaje**.

Es importante distinguir las dos salidas:

- **Predicción por umbral:** cada piloto se clasifica de manera independiente como podio/no podio, por lo que el modelo puede marcar una cantidad diferente de tres pilotos.
- **Top-3 por puntaje:** ordena a todos los pilotos por `decision_function` y toma los tres primeros como lectura deportiva adicional.

El Top-3 no cambia el entrenamiento de la SVM; es una forma de interpretar su ranking dentro de una carrera, donde físicamente solo existen tres lugares de podio.

---

## 15. Estructura del proyecto

```text
f1-miniproyecto3/
│
├── 1_preparar_datos.py
├── 2_modelos.py
├── 3_demo.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── results.csv
│   ├── races.csv
│   ├── drivers.csv
│   ├── constructors.csv
│   ├── ...otros CSV originales de Kaggle
│   └── processed/
│       └── dataset_modelado.csv          # se genera en el paso 1
│
└── outputs/
    ├── eda_desbalance.png                # se genera en el paso 1
    ├── resultados.csv                    # se genera en el paso 2
    ├── modelo_final.joblib               # se genera en el paso 2
    ├── curvas_pr.png                     # se genera en el paso 2
    ├── matrices_confusion.png            # se genera en el paso 2
    └── pca_2d.png                        # se genera en el paso 2
```

### Qué hace cada script

#### `1_preparar_datos.py`

- lee los CSV;
- integra resultados, carreras, pilotos y constructores;
- crea `podio`;
- construye las 9 características pre-carrera;
- evita usar el resultado actual en el historial mediante `shift(1)`;
- guarda `dataset_modelado.csv`;
- genera la gráfica inicial de EDA.

#### `2_modelos.py`

- realiza el split temporal;
- entrena las configuraciones lineales, RBF, SMOTE y polinomiales;
- busca `C` y `gamma` con `GridSearchCV` + `TimeSeriesSplit`;
- selecciona hiperparámetros mediante `average_precision`;
- elige el umbral F1 en validación;
- evalúa los modelos en prueba;
- guarda resultados, modelo y gráficas.

#### `3_demo.py`

- carga el modelo ya entrenado;
- permite seleccionar una carrera histórica;
- calcula el puntaje para cada piloto;
- aplica el umbral aprendido;
- compara la predicción con el podio real.

---

## 16. Instalación

### Requisitos

- Python 3.
- `pip` disponible.

Las dependencias del proyecto están en `requirements.txt`:

```text
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
joblib>=1.3
imbalanced-learn>=0.11
```

### Crear un entorno virtual

Desde la carpeta raíz del proyecto:

```bash
python -m venv .venv
```

#### Windows

```bat
.venv\Scripts\activate
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

Luego instalar:

```bash
python -m pip install -r requirements.txt
```

Si en Windows el comando disponible es `py` en lugar de `python`, se puede utilizar el mismo flujo reemplazando `python` por `py`.

---

## 17. Preparación del dataset

Si los CSV no están incluidos en el repositorio:

1. descargar el dataset de Kaggle;
2. descomprimirlo;
3. copiar como mínimo estos archivos dentro de `data/`:

```text
results.csv
races.csv
drivers.csv
constructors.csv
```

La estructura debe quedar, como mínimo:

```text
data/
├── results.csv
├── races.csv
├── drivers.csv
└── constructors.csv
```

---

## 18. Ejecución completa

Los scripts deben ejecutarse **en este orden** desde la raíz del proyecto.

### Paso 1 — preparar datos

```bash
python 1_preparar_datos.py
```

Genera:

```text
data/processed/dataset_modelado.csv
outputs/eda_desbalance.png
```

### Paso 2 — entrenar y evaluar

```bash
python 2_modelos.py
```

Este es el paso más costoso porque realiza varias búsquedas de hiperparámetros y validación cruzada.

Genera:

```text
outputs/resultados.csv
outputs/modelo_final.joblib
outputs/curvas_pr.png
outputs/matrices_confusion.png
outputs/pca_2d.png
```

### Paso 3 — ejecutar la demo

```bash
python 3_demo.py
```

El programa solicitará un año y una ronda. Las temporadas **2018–2024** corresponden al conjunto de prueba y son las más adecuadas para demostrar el comportamiento sobre datos que no participaron en entrenamiento ni validación.

Ejemplo de interacción:

```text
Anio (ej. 2023): 2023
Rondas: 1:Bahrain | 2:Saudi Arabian | ...
Ronda: 1
```

Para salir:

```text
q
```

---

## 19. Archivos generados y qué significa cada uno

| Archivo | Uso |
|---|---|
| `eda_desbalance.png` | Muestra patrones iniciales de podio según parrilla y década. |
| `resultados.csv` | Tabla completa con hiperparámetros y métricas de todos los modelos. |
| `modelo_final.joblib` | Pipeline final + umbral + lista de características para la demo. |
| `curvas_pr.png` | Compara las curvas Precision-Recall de los modelos sobre test. |
| `matrices_confusion.png` | Contrasta errores del lineal base con el modelo final. |
| `pca_2d.png` | Proyección exploratoria del espacio de características a dos componentes. |

Estas salidas son las que se utilizan para documentar y sustentar los resultados del proyecto.

---

## 20. Reproducibilidad

Se utiliza:

```python
SEMILLA = 42
```

para los procesos donde interviene aleatoriedad, entre ellos:

- submuestra utilizada en la búsqueda RBF;
- SMOTE;
- muestras utilizadas para la visualización PCA.

El split principal no depende de una semilla porque se define por años.

Para reproducir el experimento desde cero basta con disponer de los CSV originales, instalar `requirements.txt` y ejecutar los tres scripts en el orden descrito.

---

## 21. Consideraciones y limitaciones

### No es una probabilidad calibrada de podio

`decision_function` produce un **puntaje de distancia/separación**, no una probabilidad. Un puntaje mayor indica que el modelo coloca al piloto más claramente del lado positivo de la frontera.

### El modelo no impone exactamente tres podios

La clasificación se realiza de manera independiente para cada piloto. Por eso pueden aparecer más o menos de tres predicciones positivas en una carrera. El Top-3 por puntaje se presenta por separado para adaptar la lectura al contexto deportivo.

### Fórmula 1 ha cambiado mucho entre 1950 y 2024

Reglamentos, tamaño de parrilla, equipos, número de carreras y competitividad no son constantes. La división temporal ayuda a evaluar generalización, pero no elimina estos cambios de distribución.

### Registros históricos antiguos

Las primeras décadas contienen particularidades históricas, como participaciones compartidas o estructuras de parrilla que no siguen exactamente las convenciones modernas. Por eso las predicciones sobre esas temporadas deben interpretarse con cautela. El objetivo principal de generalización se evalúa en 2018–2024.

### SMOTE es una comparación, no una obligación del modelo final

El proyecto lo utiliza para estudiar una estrategia adicional frente al desbalance. El modelo final actual utiliza `class_weight="balanced"`.

---

## 22.. Idea central del proyecto

El resultado principal no es simplemente obtener una accuracy alta.

La dificultad real consiste en detectar una clase minoritaria —los podios— sin dejar que la mayoría de “no podios” esconda los errores del modelo. Por eso todo el diseño experimental se construye alrededor de:

1. información disponible antes de la carrera;
2. separación temporal;
3. escalado correcto;
4. SVM lineal y no lineal;
5. tratamiento explícito del desbalance;
6. selección mediante PR-AUC;
7. elección del umbral en validación;
8. evaluación final sobre temporadas futuras no utilizadas en el ajuste.

El proyecto no intenta memorizar quién suele ganar. Intenta comprobar hasta qué punto la posición de salida, la experiencia y la forma reciente de piloto y equipo contienen suficiente señal para separar candidatos a podio de no podio mediante una SVM.
