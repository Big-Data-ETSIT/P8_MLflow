<img  align="left" width="150" style="float: left;" src="https://www.upm.es/sfs/Rectorado/Gabinete%20del%20Rector/Logos/UPM/CEI/LOGOTIPO%20leyenda%20color%20JPG%20p.png">
<img  align="right" width="60" style="float: right;" src="https://www.dit.upm.es/images/dit08.gif">


<br/><br/>


# Práctica Orquestación MLOps - MLflow

## 1. Objetivo

- Afianzar los conceptos sobre Orquestación y MLOps.
- Registro y versionado de modelos de machine learning.
- Desplieguede modelo entrenado.

## 2. Dependencias
Para realizar la práctica el alumno deberá tener instalado en su ordenador:
- Entorno de ejecución de Python 3 [Python](https://www.python.org/downloads/)

## 3. Descripción de la práctica

En esta práctica se presenta la herramienta MLflow que permite gestionar distintas tareas de MLOps (entrenamiento, registro de modelos, versionado de modelos, despliegue, etc.) Para ello se proporciona al alumno el archivo `grades.csv` con las notas de diferentes pruebas realizadas por alumnos a lo largo del curso. El objetivo del sistema es predecir si un alumno va a aprobar el examen final de la asignatura. Para ello se proponen varios experimentos que darán como resultado distintas versiones de modelos de machine learning, se evaluarán, se compararán, y por último se desplegarán. 

## 4. Descargar e inicializar el entorno

Abra un terminal en su ordenador y siga los siguientes pasos.

Descarguese y descomprima el código con git clone o pinchando en el botón code y eligiendo opción "Download ZIP".

Navegue a través de un terminal a la carpeta P8_MLflow.
```
$ cd P8_MLflow
```

Una vez dentro de la carpeta, se instalan las dependencias. Para ello debe crear un virtual environment de la siguiente manera:

```
[LINUX/MAC] $ python3 -m venv venv
[WINDOWS] > py -m venv env
```

Si no tiene instalado venv, Lo puede instalar de la siguiente manera:

```
[LINUX/MAC] $ python3 -m pip install --user virtualenv
[WINDOWS] > py -m pip install --user virtualenv
```

Una vez creado el virtual environment lo activamos para poder instalar las dependencias:

```
[LINUX/MAC] $ source venv/bin/activate
[WINDOWS] > .\env\Scripts\activate
```

Instalamos las dependencias con pip:

```
$ pip3 install -r requirements.txt 
```

## 5. Tareas a realizar

### Bloque 1: Arrancar la interfaz de usuario de mlflow.

```
mlflow ui
```
Acceder a http://localhost:5000/

**Tarea 1: Explique qué muestra la interfaz gráfica de MLflow. Incluya en su explicación alguna captura de pantalla.**

### Bloque 2: Entrenar y registrar distintas versiones del modelo de machine learning.

El algoritmo utilizado para el entrenamiento es un arbol de decisión (DecisionTree). Cada nodo del árbol evalúa la entrada en función de alguna de sus características y como consecuencia lo deriva a un nivel inferior hasta que se alcance alguna hoja del árbol que contenga la decisión final. Por ejemplo en el caso de un predictor de supervivencia a un accidente:

![DecisionTree](https://upload.wikimedia.org/wikipedia/commons/f/f3/CART_tree_titanic_survivors.png)

1. Ejecutar el script de entrenamiento configurando la máxima profundidad del algoritmo DecisionTree a 2:

```
python train.py 2
```

El anterior script además de entrenar el modelo, lo evalúa, y lo registra con MLFlow. 

**Tarea 2: Indique qué realiza la siguiente sentencia: *mlflow.log_param("max_depth", max_depth)***

**Tarea 3: Indique qué muestra la interfaz gráfica de MLflow ahora. Incluya su explicación con alguna captura de pantalla**

2. Vuelva a entrenar el modelo utilizando ahora 10 niveles de profundidad como máximo: 

```
python train.py 10
```

**Tarea 4: Acceda al apartado modelos de interfaz gráfica de MLflow. Compare los resultados de evaluación de ambos modelos. Incluya una captura de pantalla con la comparación.**

### Bloque 3: Ejecución de los modelos.

En el anterior bloque se guardaron los modelos tras el entrenamiento. MLflow permite desplegar las distintas versiones del modelo en distintos entornos (Azure, AWS, etc.). En esta práctica se proporciona el script `predict.py` al que se le debe pasar los siguientes parámetros:

- version: versión del modelo que realizará la predicción
- laboratory_test: nota de laboratorio (número de 0 a 10)
- first_test: nota del primer test (número 0 a 1)
- second_test: nota del segundo test (número de 0 a 1)
- days_missing: días que no asistió a clase (entero)
- hours_studied: horas de estudio (entero)
- first_exam: (número de 0 a 10)

1. Ejecute la predicción del siguiente alumno para la primera versión del modelo:

```
python predict.py 1 5 0.5 0.3 10 50 3
```

**Tarea 5: Ejecute la misma predicción pero utilizando la segunda versión del modelo. Incluya el resultado de la predicción y la comando que ha utilizado para ejecutarla.**

