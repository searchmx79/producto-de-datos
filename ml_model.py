# =======================================================================================
#                       IEXE Tec - Maestría en Ciencia de Datos 
#                       Productos de Datos. Proyecto Integrador
#
# Debes de adaptar este script para integrar tu modelo predictivo.
# =======================================================================================
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle


# =======================================================================================
def create_simple_model():
    """ Función para entrenar un modelo simple usando los datos de prueba de tipos de flores
        Consulta la documentación de este famoso conjunto de datos:
        https://en.wikipedia.org/wiki/Iris_flower_data_set

        Modifica esta función para integrar el modelo predictivo que quieres integrar a
        tu API REST.

        Esta función crea un modelo muy simple usando los datos de Tipos de Flores. Es un
        modelo muy simple usando el algoritmo de regresión logística. Para este ejemplo 
        no hay ningún proceso de análisis de variables, entrenamiento o comparación de 
        múltiples modelos, etc. Estas actividades son muy importantes y se realizan en una
        etapa posterior a la creación del modelo, en este punto tu modelo ya está entrenado
        y probado, listo para integrarse a un modelo predicitivo.

        Una vez que el modelo se encuentra listo usamos la biblioteca pickle 
        (https://docs.python.org/3/library/pickle.html) para "guardar" el modelo en un
        archivo en disco duro. Este archivo contiene el estado del entrenamiento y 
        pruebas del modelo, de esta forma se "desconecta" del entorno de laboratorio del
        modelo para que se pueda usar en una aplicación de cómputo.

        El archivo en disco duro que contiene el modelo se "recupera" en el API REST, 
        quien es responsable de transformar la entrada del API en un formato que sea
        entendible por el modelo.
    """

    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    dataset = read_csv('iris.csv', names=names)
    array = dataset.values
    X = array[:, 0:4]
    y = array[:, 4]
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X, y, test_size=0.20, random_state=1, shuffle=True
    )

    # Para este ejemplo creamos un modelo de regresión logística simple
    model = LogisticRegression(solver='liblinear', multi_class='ovr')
    model.fit(X_train, Y_train)

    # Hasta este punto tu modelo debe de estar integrado y funcioando como lo esperas.

    # --------------------------------------------------------------------------------
    # Esta línea crea el archivo "pickle" que contiene el modelo predictivo, este 
    # archivo nos va a servir para reconstruir el modelo en el API REST. 
    # --------------------------------------------------------------------------------
    pickle.dump(model, open('simple_model.pkl', 'wb'))


# =======================================================================================
if __name__ == '__main__':
    """ Esta función ejecuta el programa de Python cuando se invoca desde 
        línea de comando.
    """
    create_simple_model()
