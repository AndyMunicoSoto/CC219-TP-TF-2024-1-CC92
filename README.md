# CC219-TP-TF-2024-1-CC92

# OBJETIVO DEL TRABAJO

# PARTICIPANTES

* Andy Johan Muñico Soto – u201523017
* Jorge Omar Tarapa Peña – u202021508
* Lennin Jhair Vargas Soto – u20201e766
* Angel Ruben Zuñiga Lovera – u202111299


# DESCRIPCION DEL DATASET

En base al caso de uso se encontró el conjunto de datos de sarcasmo en la página Kaggle. Este consiste en frases donde se busca determinar si hay sarcasmo o ironía en la oración. Las etiquetas asignadas son 0 para indicar que no hay sarcasmo presente en la frase y 1 para indicar que sí hay sarcasmo o ironía. Por otro lado, se divide en dos archivos .csv donde hay un conjunto de entrenamiento llamado “train.csv” con 19964 datos y otro de prueba llamado “test.csv” con 8576 datos.

# CONCLUSIONES

En este proyecto, se exploraron y compararon varios enfoques para la clasificación de sarcasmo en textos. Utilizando modelos como SVM, KNN y CNN, y combinando sus predicciones a través de un ensamble, se obtuvo un modelo robusto con una precisión aceptable.

##  **Técnicas Utilizadas**

Preprocesamiento exhaustivo de texto, vectorización TF-IDF, tokenización y padding.
Modelos de clasificación como SVM, KNN y CNN.
Ensamble por votación para combinar predicciones.

## **Resultados:**

La CNN demostró una alta precisión en el entrenamiento pero mostró signos de sobreajuste en la validación.
El modelo de ensamble alcanzó una precisión del 77.25% en el conjunto de prueba, mostrando una buena generalización.


# LICENCIA
