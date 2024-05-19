import tensorflow as tf
from tensorflow import keras
from keras import layers

def entrenar_modelo():
    print("Entrenando el modelo...")

    path_training = "Database/Glaucoma_Training"
    path_validation = "Database/Glaucoma_Validacion"

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_ds = datagen.flow_from_directory(
        path_training,
        target_size=(256, 256),
        batch_size=32,
        subset='training',
        class_mode='categorical',
        shuffle=True  
    )

    val_ds = datagen.flow_from_directory(
        path_validation,
        target_size=(256, 256),
        batch_size=32,
        subset='validation',
        class_mode='categorical',
        shuffle=False
    )
   # Convertir etiquetas a formato one-hot
    y_train = tf.keras.utils.to_categorical(train_ds.labels, num_classes=2)
    y_val = tf.keras.utils.to_categorical(val_ds.labels, num_classes=2)

    print (train_ds)
    print (val_ds)

    model = tf.keras.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=(256, 256, 3)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(2, activation='softmax')  
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10
    )
    print("Modelo entrenado.")
    model.save('my_model.h5')

def evaluar_modelo(model_path):
    print("Evaluando el modelo...")

    path_validation = "Database/Glaucoma_Validacion"

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    val_ds = datagen.flow_from_directory(
        path_validation,
        target_size=(256, 256),
        batch_size=32,
        subset='validation',
        class_mode='categorical',
        shuffle=False  
    )


    model = tf.keras.models.load_model(model_path)
    test_loss, test_acc = model.evaluate(val_ds)
    print("Test accuracy:", test_acc)
    print("Test loss:", test_loss)
    return test_acc, test_loss


def predecir_con_modelo_entrenado(model_path, path_image):
    model = tf.keras.models.load_model(model_path)
    img = tf.keras.preprocessing.image.load_img(path_image, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) 

    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
    print("Predicted class:", predicted_class)
    return predicted_class

