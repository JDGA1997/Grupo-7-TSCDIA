# Librerías necesarias
from Pillow import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Adecuar las imagenes antes de entrenar el algoritmo para que todas tengan el mismo tamaño y color

from tensorflow.keras.preprocessing.image import ImageDataGenerator

path_training = "Database/Glaucoma_Positive-Training"

path_validation = "Database/Glaucoma_Positive-Validation"


# Configurar el generador de datos de imágenes
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalizar los valores de píxeles a 0-1
    validation_split=0.2,  # Separar el 20% de las imágenes para validación
)

# Cargar las imágenes del directorio 'path' y redimensionar a 256x256 píxeles
train_ds = datagen.flow_from_directory(
    path_training,
    target_size=(256, 256),
    batch_size=32,
    subset='training',
)

# Cargar las imágenes del directorio 'path' y redimensionar a 256x256 píxeles
val_ds = datagen.flow_from_directory(
    path_validation,
    target_size=(256, 256),
    batch_size=32,
    subset='validation',
)

# Se carga el conjunto de datos y se divide en un conjunto de entrenamiento y otro de prueba

# import matplotlib.pyplot as plt
# from matplotlib.image import imread

# # Ruta de la imagen
# img_path = "/content/drive/My Drive/Glaucoma_Detection/dataset/Fundus_Train_Val_Data/Fundus_Scanes_Sorted/Train/Glaucoma_Positive/036.jpg"

# # Leer la imagen
# img = imread(img_path)

# # Mostrar la imagen
# plt.imshow(img)
# plt.axis('off')
# plt.show()


train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path_training,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(256, 256),
    batch_size=32)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path_validation,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(256, 256),
    batch_size=32)


# Preprocesamiento de datos: En este ejemplo, normalizamos los píxeles de las imágenes y las escalamos a un rango de 0 a 1:
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(
    1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Modelo de aprendizaje automático elegico CNN (red neuronal convolucional)
model = keras.Sequential([
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

# Entrenar el modelo: Utilizamos la función compile para configurar el modelo con una función de pérdida, un optimizador y una métrica de evaluación:
model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])


# A continuación, entrenamos el modelo utilizando el conjunto de datos de entrenamiento:
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)


# Evaluar el modelo: Finalmente, podemos evaluar la precisión del modelo utilizando el conjunto de datos de prueba:
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "path/to/dataset",
    validation_split=None,
    subset="testing",
    seed=123,
    image_size=(256, 256),
    batch_size=32)
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))
test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy:", test_acc)


# Para utilizar el modelo entrenado en una aplicación donde se cargan imágenes, se pueden seguir los siguientes pasos:

# Cargar el modelo entrenado: Después de entrenar el modelo, podemos guardarlo en un archivo y luego cargarlo en nuestra aplicación. Por ejemplo, en TensorFlow y Keras podemos guardar el modelo utilizando la función model.save('my_model.h5'), y luego cargarlo en nuestra aplicación con la función tf.keras.models.load_model('my_model.h5').

# Preprocesamiento de la imagen: Antes de pasar la imagen al modelo, debemos procesarla de la misma manera que las imágenes en el conjunto de datos de entrenamiento. En el ejemplo anterior, normalizamos y escalamos las imágenes utilizando la capa Rescaling. En nuestra aplicación, podemos utilizar la misma capa o crear una función similar para normalizar y escalar la imagen de entrada.

# Realizar la predicción: Finalmente, podemos pasar la imagen preprocesada al modelo para realizar la predicción. En nuestro ejemplo, utilizamos el método model.predict() para obtener las predicciones para un conjunto de imágenes. En nuestra aplicación, podemos utilizar el mismo método para obtener la predicción para una única imagen.

# Por ejemplo, si estamos desarrollando una aplicación web que permite a los usuarios cargar imágenes de enfermedades de las plantas, podríamos utilizar el modelo entrenado para comparar las imágenes y determinar si representan la misma enfermedad. El proceso sería similar al siguiente:

# Cargar el modelo entrenado:

model = tf.keras.models.load_model('my_model.h5')

# Preprocesamiento de la imagen: Supongamos que los usuarios pueden cargar imágenes en formato JPG. Podríamos utilizar la biblioteca de Python Pillow para cargar la imagen y luego escalarla y normalizarla de la misma manera que en el conjunto de datos de entrenamiento:


# Cargar la imagen
img = Image.open('path/to/image.jpg')

# Escalar y normalizar la imagen
img = img.resize((256, 256))
img = normalization_layer(np.array(img))

# Realizar la predicción: Finalmente, podemos pasar la imagen preprocesada al modelo para realizar la predicción. En nuestro ejemplo, el modelo devuelve un vector de probabilidades para cada clase de enfermedad. Podemos utilizar el índice de la clase con la mayor probabilidad para determinar la enfermedad más probable de la imagen:

# Realizar la predicción
pred = model.predict(np.array([img]))
class_idx = np.argmax(pred)

# Obtener la etiqueta correspondiente a la clase
labels = ['enfermedad_1', 'enfermedad_2', 'enfermedad_3']
label = labels[class_idx]

# En este ejemplo, si la etiqueta es enfermedad_1, entonces podemos decir que la imagen representa la misma enfermedad que las imágenes de enfermedad_1 en el conjunto de datos de entrenamiento.




#Incorporando modulo de reconocimiento de voz para que los usuarios puedan interactuar con el sistema mendiante comandos de voz.
#Esto puede ser útil para accesibilidad y usabilidad, especialmente para usuarios con discapacidades visuales.

#La app completa incluyendo los comandos de voz

#Librerias
# !pip install tensorflow
# !pip install SpeechRecognition
# !pip install pyaudio

import speech_recognition as sr
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Definir funciones para comandos de voz
def reconocer_voz():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Di algo:")
        audio = recognizer.listen(source)

    try:
        comando = recognizer.recognize_google(audio, language="es-ES")
        print(f"Has dicho: {comando}")
        return comando.lower()
    except sr.UnknownValueError:
        print("No se entendió el comando de voz.")
        return ""
    except sr.RequestError as e:
        print(f"Error al solicitar resultados; {e}")
        return ""

def procesar_comando(comando):
    if "cargar imágenes" in comando:
        cargar_imagenes()
    elif "entrenar modelo" in comando:
        entrenar_modelo()
    elif "evaluar modelo" in comando:
        evaluar_modelo()
    else:
        print("Comando no reconocido.")

# Función para cargar imágenes
def cargar_imagenes():
    print("Cargando imágenes...")
    path_training = "http://drive.google.com/drive/folders/1rWPaxRdd0bNlX5-9wI_vGiZb6eCHIE7X?usp=sharing"
    path_validation = "http://drive.google.com/drive/folders/1wEg0RnxZvs3cj3hLHCvBqPtU-onyGYUF?usp=sharing"

    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
    )

    train_ds = datagen.flow_from_directory(
        path_training,
        target_size=(256, 256),
        batch_size=32,
        subset='training',
    )

    val_ds = datagen.flow_from_directory(
        path_validation,
        target_size=(256, 256),
        batch_size=32,
        subset='validation',
    )

    return train_ds, val_ds

# Función para entrenar el modelo
def entrenar_modelo():
    print("Entrenando el modelo...")
    train_ds, val_ds = cargar_imagenes()

    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    num_classes = 2  # Ajustar según tu problema de clasificación

    model = keras.Sequential([
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )
    print("Modelo entrenado.")

# Función para evaluar el modelo
def evaluar_modelo():
    print("Evaluando el modelo...")
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "path/to/dataset",
        validation_split=None,
        subset="testing",
        seed=123,
        image_size=(256, 256),
        batch_size=32
    )
    normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    # Aquí deberías cargar tu modelo previamente entrenado
    model = keras.models.load_model('path/to/your/model.h5')
    test_loss, test_acc = model.evaluate(test_ds)
    print("Test accuracy:", test_acc)

# Ciclo principal
while True:
    comando = reconocer_voz()
    if comando == "salir":
        break
    procesar_comando(comando)
