# Telco Customer Churn — Proyecto MLOps

Sistema de Machine Learning para la predicción de cancelación de clientes (*Customer Churn*), desarrollado mediante un flujo de trabajo MLOps que integra preparación de datos, entrenamiento, seguimiento de experimentos, versionado, orquestación, monitoreo y despliegue de modelos.

El objetivo es identificar clientes con probabilidad de abandonar el servicio de telecomunicaciones, utilizando información relacionada con su antigüedad, cargos y servicios contratados.

## 1. Arquitectura del proyecto

El proyecto implementa un flujo de trabajo automatizado:

```text
                 DATASET ORIGINAL
                        |
                        v
                 PREPARACIÓN DE DATOS
                    (Pandas)
                        |
                        v
                 DATASET PROCESADO
                        |
                        v
                  ENTRENAMIENTO
                  (Scikit-learn)
                        |
             +----------+----------+
             |          |          |
             v          v          v
          Logistic   Random      SVM
         Regression  Forest
             |          |          |
             +----------+----------+
                        |
                        v
                 EVALUACIÓN
              (Métricas de ML)
                        |
                        v
                  MLFLOW
            Experimentos y modelos
                        |
                        v
                MODELO CHAMPION
                        |
              +---------+---------+
              |                   |
              v                   v
          FASTAPI              AIRFLOW
          Predicción         Orquestación
              |                   |
              v                   v
          FRONTEND             EVIDENTLY
                              Data Drift
```

### Tecnologías utilizadas

| Tecnología | Función |
|---|---|
| Python | Lenguaje principal |
| Pandas | Manipulación y preparación de datos |
| Scikit-learn | Entrenamiento y evaluación de modelos |
| DVC | Versionado de datasets |
| MLflow | Seguimiento de experimentos y registro de modelos |
| Apache Airflow | Orquestación del pipeline |
| Evidently AI | Monitoreo de cambios en los datos |
| FastAPI | API para realizar predicciones |
| Pydantic | Validación de datos de entrada |
| Docker Compose | Contenerización y administración de servicios |
| Pytest | Pruebas automatizadas |
| GitHub Actions | Integración continua (CI) |

## 2. Modelos de Machine Learning

Se entrenaron y evaluaron tres algoritmos de clasificación:

- Regresión Logística.
- Random Forest.
- Support Vector Machine (SVM).

El conjunto de datos se divide utilizando una proporción de 80% para entrenamiento y 20% para evaluación, con estratificación de la variable objetivo y `random_state=42`.

Se utiliza `StandardScaler` para las variables numéricas y `OneHotEncoder` para las variables categóricas.

También se aplica balanceo de clases mediante `class_weight="balanced"`.

### Resultados

Los modelos se evaluaron mediante las siguientes métricas:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- ROC-AUC.

Resultados obtenidos:

| Modelo | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7239 | 0.4878 | **0.8021** | 0.6067 | **0.8346** |
| Random Forest | 0.7452 | 0.5166 | 0.6257 | 0.5659 | 0.7914 |
| SVM | **0.7700** | **0.5517** | 0.7139 | **0.6224** | 0.8106 |

El modelo SVM fue seleccionado como modelo principal debido a que obtuvo el mayor F1-score.

El modelo seleccionado se registró en MLflow bajo el nombre `TelcoChurnClassifier`, utilizando el alias `Champion`.

**Consideración:** La selección del modelo depende del objetivo del negocio. Si el costo de no identificar a un cliente que abandonará el servicio es especialmente elevado, el Recall puede tener mayor prioridad que el F1-score.

## 3. Versionado de datos con DVC

DVC permite mantener el control de versiones de los datasets sin almacenar directamente los archivos CSV dentro del repositorio Git.

Los datasets versionados son:

```text
data/
├── raw/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── processed/
    └── telco.csv
```

Los archivos `.dvc` contienen los metadatos necesarios para identificar las versiones de los datasets.

**Nota sobre reproducibilidad:** El repositorio contiene los archivos de seguimiento de DVC, pero no tiene configurado un repositorio remoto de almacenamiento DVC. Por lo tanto, los datos originales deben obtenerse por separado y el almacenamiento local de DVC no se transfiere automáticamente a GitHub.

## 4. Seguimiento de experimentos con MLflow

MLflow se utiliza para registrar los experimentos de entrenamiento y administrar los modelos resultantes.

Se utiliza:

- SQLite como backend de seguimiento.
- Almacenamiento local para los artefactos.
- Model Registry para administrar las versiones.
- Alias `Champion` para identificar el modelo seleccionado.

El modelo registrado puede cargarse mediante:

```python
model_uri = "models:/TelcoChurnClassifier@Champion"
model = mlflow.pyfunc.load_model(model_uri)
```

## 5. Orquestación con Apache Airflow

El pipeline se encuentra definido en:

```text
airflow/dags/telco_churn_pipeline.py
```

Está compuesto por tres tareas:

| Tarea | Descripción |
|---|---|
| `prepare_data` | Ejecuta la preparación y validación de los datos |
| `train_models` | Entrena, evalúa y registra los modelos |
| `monitor_data` | Genera el reporte de monitoreo con Evidently |

Las tareas se ejecutan secuencialmente:

```text
prepare_data
      |
      v
train_models
      |
      v
monitor_data
```

La ejecución del DAG fue validada satisfactoriamente, con las tres tareas completadas en estado `success`.

## 6. Monitoreo con Evidently AI

Se utiliza Evidently para identificar posibles cambios en la distribución de los datos (*Data Drift*).

Las variables monitoreadas son:

- tenure
- MonthlyCharges
- TotalCharges
- Contract
- InternetService

El reporte generado se encuentra en:

```text
reports/data_drift.html
```

El monitoreo utiliza una partición de los datos históricos como referencia y otra como conjunto actual.

**Limitación:** Esta configuración permite demostrar el funcionamiento de la herramienta, pero no representa un escenario real de producción. Para evaluar drift operativo sería necesario comparar los datos históricos con nuevos datos recibidos posteriormente.

## 7. API de predicción

Se desarrolló una API REST utilizando FastAPI.

La API carga el modelo registrado en MLflow mediante el alias `Champion`.

### Endpoints

| Método | Endpoint | Descripción |
|---|---|---|
| GET | `/health` | Verifica el estado del servicio |
| POST | `/predict` | Genera una predicción de cancelación |

Ejemplo de solicitud:

```json
{
    "tenure": 12,
    "MonthlyCharges": 75.5,
    "TotalCharges": 906.0,
    "Contract": "Month-to-month",
    "InternetService": "Fiber optic"
}
```

Ejemplo de respuesta:

```json
{
    "prediction": 1,
    "churn": "Yes",
    "model": "TelcoChurnClassifier@Champion"
}
```

Los datos de entrada son validados mediante Pydantic.

La documentación interactiva de la API está disponible en:

```text
http://localhost:8000/docs
```

## 8. Interfaz de usuario

Se implementó una interfaz web sencilla para interactuar con la API y realizar predicciones.

La interfaz permite introducir información del cliente y visualizar el resultado de la clasificación.

Se utiliza HTML, CSS y JavaScript para consumir el endpoint `/predict`.

## 9. Instalación y ejecución

### Requisitos

- Git.
- Python 3.12 o superior para la ejecución local.
- Docker Engine.
- Docker Compose.
- DVC.

### Clonar el repositorio

```bash
git clone git@github.com:AdorianSAK/telco-mlops.git
cd telco-mlops
```

### Crear entorno virtual

```bash
python -m venv .venv
source .venv/bin/activate
```

### Instalar dependencias

```bash
pip install -r requirements.txt
```

### Preparar los datos

Colocar el dataset original en:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Ejecutar:

```bash
python -m src.prepare_data
```

### Entrenar los modelos

```bash
python -m src.train
```

### Generar el reporte de monitoreo

```bash
python -m src.monitor
```

### Ejecutar las pruebas

```bash
python -m pytest -v
```

### Ejecutar los servicios con Docker Compose

Inicializar Airflow:

```bash
docker compose -f airflow/docker-compose.yaml up airflow-init
```

Levantar los servicios:

```bash
docker compose -f airflow/docker-compose.yaml up -d
```

Interfaces disponibles:

| Servicio | Dirección |
|---|---|
| Airflow | http://localhost:8080 |
| API | http://localhost:8000 |
| Documentación API | http://localhost:8000/docs |

Para iniciar el frontend por separado:

```bash
python -m http.server 5500 --directory frontend
```

Acceder a:

```text
http://localhost:5500
```

## 10. Integración continua

Se configuró GitHub Actions mediante:

```text
.github/workflows/ci.yml
```

El workflow se ejecuta automáticamente ante:

- Push a la rama `master`.
- Pull requests dirigidos a `master`.

El proceso realiza:

1. Descarga del repositorio.
2. Configuración de Python.
3. Instalación de dependencias.
4. Ejecución de las pruebas automatizadas con Pytest.

La ejecución inicial del workflow en GitHub Actions finalizó satisfactoriamente.

## 11. Pruebas automatizadas

Actualmente se cuenta con cuatro pruebas automatizadas:

| Prueba | Objetivo |
|---|---|
| `test_missing_total_charges_with_zero_tenure` | Verifica la corrección de cargos faltantes para clientes con antigüedad cero |
| `test_missing_total_charges_with_nonzero_tenure` | Verifica el rechazo de registros inconsistentes |
| `test_prediction_returns_churn_yes` | Verifica la respuesta de predicción de la API utilizando un modelo simulado |
| `test_customer_data_rejects_negative_tenure` | Verifica la validación de datos de entrada |

Las cuatro pruebas fueron ejecutadas satisfactoriamente tanto en el entorno local como mediante GitHub Actions.

## 12. Estructura del repositorio

```text
telco-mlops/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── airflow/
│   ├── dags/
│   │   └── telco_churn_pipeline.py
│   ├── Dockerfile
│   └── docker-compose.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── frontend/
│   └── index.html
│
├── reports/
│   ├── data_drift.html
│   └── model_comparison.csv
│
├── src/
│   ├── api.py
│   ├── mlflow_config.py
│   ├── monitor.py
│   ├── prepare_data.py
│   └── train.py
│
├── tests/
│   ├── test_api.py
│   └── test_prepare_data.py
│
├── .dvc/
├── .gitignore
├── .dvcignore
└── requirements.txt
```

## 13. Conclusiones

El proyecto demuestra la implementación de un flujo de trabajo MLOps que integra distintas herramientas para administrar el ciclo de vida de un modelo de Machine Learning.

Se desarrolló un proceso que abarca desde la preparación de los datos hasta el despliegue de un modelo mediante una API REST, incorporando seguimiento de experimentos, versionado de datos, orquestación, monitoreo y pruebas automatizadas.

La integración de estas herramientas permite mejorar la organización, trazabilidad y mantenibilidad de los procesos de Machine Learning.

Como trabajo futuro se propone implementar almacenamiento remoto para DVC, monitoreo continuo con nuevos datos de producción, evaluación periódica del rendimiento del modelo y despliegue en infraestructura externa.

---

**Autor:** AdorianSAK  
**Proyecto:** Telco Customer Churn — MLOps