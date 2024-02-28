# CALIFORNIA HOUSING DATASET (REGRESSION)

Se documenta un modelo de regresión lineal en Github y WandB para el dataset "California Housing" de scikit-learn. Se prepocesan los datos y posteriormnete son evaluados mediante las métricas R^2 y MSE. 

Para la ejecución del proyecto se realizaron los siguientes pasos: 

1. Carga de datos: Se carga el conjunto de datos "California Housing dataset (regression)" y se almacena en el artefacto "Housing-Raw". 

2. Prepocesamiento de datos: Se realiza limpieza de la data, se normalizan los datos y se fragmenta en test, validation y train. Además se crea el artefacto "Housing.preprocess" en WandB que almacena los datos preprocesados. 

3. Entrenamiento y evalución: Se implementa el entrenamiento y evaluación del modelo de regresión lineal, se almacena en el artefacto "trained-model" y se calculan las métricas R^2 y el error cuadrático medio (MSE). 

4. Visualización: Se genera diagrama de dispersión con los residuos del modelo y resultados de R^2 y MSE. 





## Regresión lineal

LinearRegression ajusta un modelo lineal con coeficientes w = (w1,…, wp) para minimizar la suma residual de cuadrados entre los objetivos observados en el conjunto de datos y los objetivos predichos por la aproximación lineal.

Más información en [Scikit-Learn](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)
## Documentation

Weights and Biases: 

Weights & Biases es la plataforma de aprendizaje automático para que los desarrolladores creen mejores modelos más rápido. Utilice las herramientas interoperables y livianas de W&B para realizar un seguimiento rápido de experimentos, versionar e iterar conjuntos de datos, evaluar el rendimiento del modelo, reproducir modelos, visualizar resultados y detectar regresiones, y compartir hallazgos con colegas.

Más información en [WandB](https://docs.wandb.ai/)

GitHub Actions:

GitHub Actions automatiza, personaliza y ejecuta flujos de trabajo de desarrollo de software directamente en un repositorio con GitHub Actions. También proporciona una amplia gama de acciones prediseñadas que se pueden utilizar para automatizar tareas comunes, como ejecutar pruebas, implementar en plataformas en la nube y enviar notificaciones.

Más información en [GitHub Actions](https://docs.github.com/es/actions)

Scikit-Learn: 

Es una biblioteca de Python que proporciona acceso a versiones eficaces de muchos algoritmos comunes. También proporciona una API propia y estandarizada. Por tanto, una de las grandes ventajas de Scikit-Learn es que una vez que se entiende el uso básico y su sintaxis para un tipo de modelo, cambiar a un nuevo modelo o algoritmo es muy sencillo. La biblioteca no solo permite hacer el modelado, sino que también puede garantizar los pasos de preprocesamiento que veremos en el siguiente artículo.

Más información en [Scikit-Learn](https://scikit-learn.org/dev/)





## Authors

- Andrés Puerta Gonzáles
- Matías Camelo Valera
- Stevens Restrepo Vallejo
- Karolina Arrieta Salgado
- Viviana Idárraga Ojeda.
