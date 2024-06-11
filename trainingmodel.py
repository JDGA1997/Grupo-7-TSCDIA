import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D, Flatten, Dense, Input, Dropout, Activation, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
import numpy as np

def entrenar_modelo():
    print("Entrenando el modelo con validación cruzada...")

    path_training = "Database/Glaucoma_Training"

    #Data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.4,1.5]
    )

    data_generator = datagen.flow_from_directory(
        path_training,
        target_size=(224, 224),
        batch_size=16,
        class_mode='binary',
        shuffle=True
        )
    
    # Cargar todas las imágenes en memoria
    X, y = [], []
    num_batches = len(data_generator)
    for i in range(num_batches):
        batch = data_generator[i]
        X.extend(batch[0])
        y.extend(batch[1])

    X = np.array(X)
    y = np.array(y)

    # Crear y compilar el modelo
    def crear_modelo():
        model = tf.keras.Sequential([
            layers.Input(shape=(224, 224, 3)),
            layers.Conv2D(32, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(0.01), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2,2)),
            layers.Dropout(0.25),
            layers.Conv2D(64, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(0.01),  activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2,2)),
            layers.Dropout(0.25),
            layers.Conv2D(128, kernel_size=(3,3), padding='same', kernel_regularizer=regularizers.l2(0.01),  activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2,2)),
            layers.Dropout(0.5),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),  #0.0002
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        return model


 #Definir callback
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )

    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'best_model.model.keras',
        monitor='accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.1,
        patience=3,
        verbose=1,
        min_lr=1e-6
    )

    # Guardar mejor desempeño de modelo obtenido
    model = model_checkpoint

    # Validación cruzada    
    num_folds = 5
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_no = 1
    accuracies = []
    losses = []
    precisions = []
    recalls = []
    f1s = []
    aucs = []

    
    for train, test in kfold.split(X, y):
        model = crear_modelo()
        history = model.fit(X[train], y[train], 
                            epochs=30, batch_size=16, 
                            callbacks=[early_stopping, model_checkpoint, reduce_lr], 
                            verbose=1, 
                            validation_data=(X[test], y[test]))

        scores = model.evaluate(X[test], y[test], verbose=0)
        print(f"Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]}")
        accuracies.append(scores[1])
        losses.append(scores[0])

        y_pred = model.predict(X[test])
        y_pred_labels = (y_pred > 0.5).astype(int)  # Convertir probabilidades a etiquetas binarias
        y_true_labels = y[test].astype(int)

        precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
        recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
        f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
        auc = roc_auc_score(y[test], y_pred)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        aucs.append(auc)

        print(f"Precision for fold {fold_no}: {precision}")
        print(f"Recall for fold {fold_no}: {recall}")
        print(f"F1-Score for fold {fold_no}: {f1}")
        print(f"AUC for fold {fold_no}: {auc}")

        fold_no += 1

    print(f"Average Accuracy: {np.mean(accuracies)}")
    print(f"Average Loss: {np.mean(losses)}")
    print(f"Average Precision: {np.mean(precisions)}")
    print(f"Average Recall: {np.mean(recalls)}")
    print(f"Average F1-Score: {np.mean(f1s)}")
    print(f"Average AUC: {np.mean(aucs)}")

    model.save('my_model.keras')


def evaluar_modelo(model_path):
    print("Evaluando el modelo...")

    path_validation = "Database/Glaucoma_Validacion"
    

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    val_ds = datagen.flow_from_directory(
        path_validation,
        target_size=(224, 224),
        batch_size=16,
        class_mode='binary',
        shuffle=False
    )

    model = tf.keras.models.load_model(model_path)
    test_loss, test_acc = model.evaluate(val_ds)
    print("Test accuracy:", test_acc)
    print("Test loss:", test_loss)
    
    # Predicciones y etiquetas verdaderas
    y_pred = model.predict(val_ds)
    y_pred_labels = (y_pred > 0.5).astype(int)  # Convertir probabilidades a etiquetas binarias
    y_true_labels = val_ds.labels.astype(int)

    # Calcular matriz de confusión
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    print("Matriz de Confusión:")
    print(cm)

    # Calcular precisión, recall, F1-score y ROC AUC
    precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
    recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
    f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
    y_pred_prob = model.predict(val_ds)
    y_true = val_ds.classes
    roc_auc = roc_auc_score(y_true, y_pred_prob)

    print(f"Test accuracy: {test_acc}")
    print(f"Test loss: {test_loss}")
    print(f"Confusion Matrix:\n {cm}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-score: {f1}")
    print(f"ROC AUC: {roc_auc}")

    return test_acc, test_loss, cm, precision, recall, f1, roc_auc 

def predecir_con_modelo_entrenado(model_path, path_image):
    model = tf.keras.models.load_model(model_path)
    img = tf.keras.preprocessing.image.load_img(path_image, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)  # Expande las dimensiones para que coincida con el formato de entrada del modelo

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=0).item()
    print("Predicted class:", predicted_class)
    return predicted_class


# # Rutas de los directorios de entrenamiento y validación
# path_training = "Database/Glaucoma_Training"
# path_validation = "Database/Glaucoma_Validacion"

# # Cargar y preprocesar los datos de entrenamiento y validación
# train_datagen = keras.preprocessing.image.ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# train_generator = train_datagen.flow_from_directory(
#     path_training,
#     target_size=(256, 256),
#     batch_size=32,
#     class_mode='categorical',
#     shuffle=True
# )

# validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# validation_generator = validation_datagen.flow_from_directory(
#     path_validation,
#     target_size=(256, 256),
#     batch_size=32,
#     class_mode='categorical',
#     shuffle=False
# )

# # Cargar ResNet50 pre-entrenado
# base_model = keras.applications.ResNet50(weights='imagenet', include_top=False)

# # Agregar capas personalizadas
# x = base_model.output
# x = layers.GlobalAveragePooling2D()(x)
# x = layers.Dense(128, activation='relu')(x)
# predictions = layers.Dense(2, activation='softmax')(x)  # Suponiendo dos clases (glaucoma y no glaucoma)

# model = keras.Model(inputs=base_model.input, outputs=predictions)

# # Congelar las capas base de ResNet50
# for layer in base_model.layers:
#     layer.trainable = False

# # Compilar el modelo
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Entrenar el modelo
# model.fit(train_generator, epochs=20, validation_data=validation_generator)

# # Evaluar el modelo
# test_loss, test_accuracy = model.evaluate(validation_generator)
# print("Test accuracy:", test_accuracy)
# print("Test loss:", test_loss)

# # Guardar el modelo entrenado
# model.save('my_model_with_resnet.h5')