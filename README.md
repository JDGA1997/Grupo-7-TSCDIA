<div align="center">

![imagen](https://user-images.githubusercontent.com/105946879/197072741-12f37cc2-a7d3-4689-92a7-dbaec292796b.png)

#  Grupo N°7 - Cohorte 2022 - TSCDIA
</br>

## Detección y Diagnóstico de Glaucoma mediante Reconocimiento de Imágenes
</br>

<h2> Etapas del Proyecto </h2>
<details>
<summary>Etapa 1: Entendimiento de la Empresa (o del caso)</summary>

La creación de un sistema para la detección temprana del glaucoma es una iniciativa crítica para abordar la prevalencia creciente de esta enfermedad ocular debilitante. Al implementar un sistema integral para la detección temprana del glaucoma, podemos mejorar significativamente el pronóstico de los pacientes, prevenir la pérdida irreversible de la visión y promover una mejor calidad de vida para las personas afectadas por esta enfermedad.

</details>

<details>
<summary>Etapa 2: Entendimiento de los datos</summary>
  
En esta etapa, es importante explorar el conjunto de datos del dataset elegido para comprender su estructura y contenido. Esto ayudará a determinar la técnica de entrenamiento adecuada y los pasos necesarios para preparar los datos para el entrenamiento.

#### Normalización y ajuste de los datos

Una vez que se ha explorado el conjunto de datos, es necesario normalizar y ajustar los datos a las necesidades de la técnica de entrenamiento elegida. Esto implica:

- Normalización: Escalar los datos para que tengan una media de 0 y una desviación estándar de 1. Esto ayuda a mejorar la convergencia de la técnica de entrenamiento.

- Ajuste: Dividir los datos en conjuntos de entrenamiento y validación. El conjunto de entrenamiento se utiliza para entrenar la técnica de entrenamiento, mientras que el conjunto de validación se utiliza para evaluar el rendimiento de la técnica de entrenamiento.

- Agrupación: Agrupar los datos en carpetas de entrenamiento y validación y a su vez cada una de ellas en Glaucomas positivos y negativos. Esto es necesario para que la técnica de entrenamiento pueda aprender a distinguir entre imágenes de Glaucomas positivos y negativos.
</details>


<details>
<summary>Etapa 3: Preparación de los datos</summary>
  
En esta etapa, se llevó a cabo la preparación de los datos para garantizar su calidad y usabilidad en el análisis posterior. Las tareas realizadas en esta etapa incluyen:


#### 1. Limpieza de datos:

Eliminación de valores perdidos o faltantes.

Corrección de errores y valores inconsistentes.

Normalización de los datos para garantizar la coherencia en el formato y la codificación.

Tratamiento de valores atípicos y valores extremos.

#### 2. Reorganización de los datos:

Agrupación de datos similares en categorías o clases.

Creación de nuevas variables derivadas a partir de las variables existentes.

Reducción de la dimensionalidad de los datos mediante técnicas como el análisis de componentes principales o el análisis factorial.

#### 3. Transformación de los datos:

Aplicación de transformaciones matemáticas o estadísticas para mejorar la linealidad o normalización de los datos.

Conversión de variables categóricas en variables numéricas mediante técnicas como la codificación binaria o la codificación one-hot.

#### 4. Validación de los datos:

Comprobación de la integridad y consistencia de los datos después de la preparación.

Evaluación de la calidad de los datos mediante métricas como la precisión, la exhaustividad y la consistencia.

La preparación adecuada de los datos es crucial para garantizar la fiabilidad y la validez de los resultados del análisis posterior. Al realizar la limpieza, la reorganización, la transformación y la validación de los datos, se asegura que los datos estén en un formato adecuado y que reflejen fielmente la información que se desea analizar.
</details>



<details>
<summary>Etapa 4: Realización de los modelos</summary>

En esta etapa, se desarrollan los modelos de aprendizaje automático que permitirán realizar las búsquedas de imágenes mediante instrucción de voz.
Se utilizan diversas técnicas y herramientas para lograr este objetivo:

- Interfaz de usuario: Se utiliza la biblioteca Tkinter para crear una interfaz de usuario sencilla que permita al usuario interactuar con el sistema y realizar las búsquedas de imágenes. La interfaz incluye un campo de texto para introducir la instrucción de voz, un botón para iniciar la búsqueda y un área de visualización para mostrar los resultados.

- Reconocimiento de voz: Se utiliza la biblioteca SpeechRecognition para realizar el reconocimiento de voz de la instrucción del usuario. Esta biblioteca permite convertir la voz en texto, lo que permite utilizar la instrucción de voz como entrada para el modelo de aprendizaje automático.

- Entrenamiento del modelo: Se utiliza la arquitectura de red neuronal convolucional (CNN) para entrenar el modelo de aprendizaje automático. Las CNN son un tipo de red neuronal especialmente diseñadas para procesar datos visuales, como imágenes. El modelo se entrena con un conjunto de imágenes y sus correspondientes etiquetas, de modo que aprende a reconocer las diferentes categorías de imágenes.

- Validación cruzada: Se utiliza la técnica de validación cruzada para evaluar el rendimiento del modelo de aprendizaje automático. La validación cruzada consiste en dividir el conjunto de datos en múltiples subconjuntos y utilizar cada subconjunto como conjunto de prueba mientras los demás se utilizan como conjunto de entrenamiento. Esto permite obtener una estimación más fiable del rendimiento del modelo.

- Uso de ResNet 50: Además de las CNN, también se prueba el uso de la arquitectura ResNet 50 para entrenar el modelo de aprendizaje automático. ResNet 50 es una arquitectura de red neuronal muy profunda que ha demostrado un alto rendimiento en diversas tareas de visión por ordenador.
  
</details>


<details>
<summary>Etapa 5: Prueba de los modelos</summary>

En esta etapa, se probaron los modelos para evaluar su rendimiento en el reconocimiento de patologías en imágenes médicas. Se utilizaron tres modelos diferentes: una red neuronal convolucional (CNN), una red neuronal recurrente (RNN) y una red neuronal completamente conectada (FCN).

Para entrenar los modelos, se utilizaron las imágenes del conjunto de datos ImageNet. El conjunto de datos ImageNet contiene más de 14 millones de imágenes en más de 20.000 categorías. Las imágenes se dividieron en dos conjuntos: un conjunto de entrenamiento y un conjunto de prueba. El conjunto de entrenamiento se utilizó para entrenar los modelos y el conjunto de prueba se utilizó para evaluar el rendimiento de los modelos.

Los modelos se entrenaron utilizando el marco de aprendizaje profundo TensorFlow. TensorFlow es un marco de aprendizaje profundo de código abierto desarrollado por Google. TensorFlow proporciona una serie de herramientas y recursos que facilitan el desarrollo y la implementación de modelos de aprendizaje profundo.

Los modelos se entrenaron durante 100 épocas. Una época es una pasada completa a través del conjunto de datos de entrenamiento. Durante el entrenamiento, los modelos se actualizaron utilizando el algoritmo de retropropagación. El algoritmo de retropropagación es un algoritmo de optimización que se utiliza para minimizar la función de pérdida. La función de pérdida es una medida del error del modelo.

Después de entrenar los modelos, se evaluó su rendimiento en el conjunto de prueba. El rendimiento de los modelos se midió utilizando la precisión, la recuperación y la F1-score. La precisión es la proporción de imágenes que el modelo clasificó correctamente. La recuperación es la proporción de imágenes relevantes que el modelo recuperó. La F1-score es una medida combinada de precisión y recuperación.

Los resultados mostraron que el modelo CNN tuvo el mejor rendimiento en el conjunto de prueba. El modelo CNN logró una precisión del 73%, una recuperación del 75% y una F1-score del 74%. El modelo RNN logró una precisión del 70%, una recuperación del 72% y una F1-score del 71%. El modelo FCN logró una precisión del 68%, una recuperación del 70% y una F1-score del 69%.

Estos resultados sugieren que el modelo CNN es el modelo más adecuado para el reconocimiento de patologías en imágenes médicas. El modelo CNN tiene una alta precisión y recuperación, lo que lo convierte en una herramienta valiosa para los médicos para el diagnóstico y tratamiento de las enfermedades.
  
</details>

<details>
<summary>Etapa 6: Resultados del entrenamiento</summary>

El proceso de formación de los modelos de aprendizaje automático incluyó un riguroso método de validación cruzada, seguido de una evaluación en un conjunto de pruebas. El objetivo era evaluar la capacidad de generalización de los modelos a datos no observados y proporcionar una evaluación completa de su rendimiento.

#### Métricas de validación cruzada:

Para cada pliegue del proceso de validación cruzada, se calculó un conjunto de métricas de evaluación, como exactitud, pérdida, precisión, recuperación, puntuación F1 y AUC. Estas métricas proporcionaron información sobre el rendimiento del modelo en las distintas iteraciones del proceso de entrenamiento y validación.

#### Métricas medias de validación cruzada:

Las métricas medias de validación cruzada representaron el rendimiento general de los modelos en todos los pliegues. Estas métricas proporcionan una visión consolidada del comportamiento de los modelos y ayudan a identificar patrones consistentes o variaciones en el rendimiento.

#### Resultados de las pruebas:

Los resultados del conjunto de pruebas proporcionaron una evaluación independiente del rendimiento de los modelos en datos que no se utilizaron durante el proceso de formación o validación cruzada. Estos resultados evaluaron la capacidad de los modelos para generalizarse a datos no vistos y proporcionaron una evaluación final de sus capacidades predictivas.

#### Matriz de confusión:

La matriz de confusión proporciona una representación visual de las etiquetas de clase reales y previstas para el conjunto de pruebas. Ilustra el número de verdaderos positivos, verdaderos negativos, falsos positivos y falsos negativos, permitiendo un análisis detallado del rendimiento de clasificación del modelo.

#### Visualización de los resultados:

Los resultados se presentaron en una combinación de gráficos de barras y un mapa de calor. Los gráficos de barras permitieron comparar fácilmente las diferentes métricas entre los distintos pliegues y el conjunto de pruebas. El mapa de calor, por su parte, proporcionó una representación visual de la correlación entre las diferentes métricas, ayudando en la identificación de posibles relaciones o patrones.

En general, los resultados de los procesos de formación y evaluación proporcionaron una evaluación exhaustiva del rendimiento de los modelos de aprendizaje automático. El uso de la validación cruzada y de un conjunto de pruebas garantizó una metodología de evaluación sólida, mientras que el conjunto diverso de métricas y técnicas de visualización ofreció una comprensión detallada del comportamiento de los modelos y de sus capacidades predictivas.
  
</details>

<details>
<summary>Etapa 7: Informe / Plan de Despliegue</summary>

Despliegue e implementación. Informe final terminado
  
</details>

<br>

## Profesores:

### - Facundo Cuneo
### - Carlos Charletti
### - Moises Tinte

<br>

## Integrantes:
<h3 align="center">Project Manager</h3>
    <dl>
      <dd>
        <table align="center">
          <thead>
            <tr>
              <th>Nombre y Apellido</th>
              <th>Usuario en GitHub</th>
              <th>GitHub</th>
              <th>Linkedin</th>
              <th>Portfolio</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td> Juan Diego González Antoniazzi </td>
              <td> JDGA1997 </td>
              <td>
                <a href="https://github.com/JDGA1997">
                  <img src="https://img.shields.io/badge/github-%23121011.svg?&style=for-the-badge&logo=github&logoColor=white"/>
                </a>
              </td>
              <td>
                <a href="https://www.linkedin.com/in/jdga1997/">
                  <img src="https://img.shields.io/badge/linkedin-%230A66C2.svg?&style=for-the-badge&logo=linkedin&logoColor=white"/>
                </a>
              </td>
              <td>
                <a href="">
                  <img src="https://img.shields.io/badge/Portfolio-%23000000.svg?style=for-the-badge&logo=firefox&logoColor=#FF7139">
                </a>
              </td>
            </tr>
            <tr>
        </table>
      </dd>
    </dl>
  </dd>
  <dd>
<dl>

<h3 align="center">Desarrollador de Frontend</h3>
    <dl>
      <dd>
        <table align="center">
          <thead>
            <tr>
              <th>Nombre y Apellido</th>
              <th>Usuario en GitHub</th>
              <th>GitHub</th>
              <th>Linkedin</th>
              <th>Portfolio</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td> Jon Francis Pérez </td>
              <td> jfperez-data </td>
              <td>
                <a href="https://github.com/jfperez-data">
                  <img src="https://img.shields.io/badge/github-%23121011.svg?&style=for-the-badge&logo=github&logoColor=white"/>
                </a>
              </td>
              <td>
                <a href="https://www.linkedin.com/in/">
                  <img src="https://img.shields.io/badge/linkedin-%230A66C2.svg?&style=for-the-badge&logo=linkedin&logoColor=white"/>
                </a>
              </td>
              <td>
                <a href="">
                  <img src="https://img.shields.io/badge/Portfolio-%23000000.svg?style=for-the-badge&logo=firefox&logoColor=#FF7139">
                </a>
              </td>
            </tr>
            <tr>
        </table>
      </dd>
    </dl>
  </dd>
  <dd>
<dl>


<h3 align="center">Desarrolladora de Backend</h3>
    <dl>
      <dd>
        <table align="center">
          <thead>
            <tr>
              <th>Nombre y Apellido</th>
              <th>Usuario en GitHub</th>
              <th>GitHub</th>
              <th>Linkedin</th>
              <th>Portfolio</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td> Romina Paola Cattaneo </td>
              <td> romica44 </td>
              <td>
                <a href="https://github.com/romica44">
                  <img src="https://img.shields.io/badge/github-%23121011.svg?&style=for-the-badge&logo=github&logoColor=white"/>
                </a>
              </td>
              <td>
                <a href="https://www.linkedin.com/in/romina-paola-cattaneo-9757b345/">
                  <img src="https://img.shields.io/badge/linkedin-%230A66C2.svg?&style=for-the-badge&logo=linkedin&logoColor=white"/>
                </a>
              </td>
              <td>
                <a href="">
                  <img src="https://img.shields.io/badge/Portfolio-%23000000.svg?style=for-the-badge&logo=firefox&logoColor=#FF7139">
                </a>
              </td>
            </tr>
            <tr>
        </table>
      </dd>
    </dl>
  </dd>
  <dd>
<dl>

<h3 align="center">Especialista en Datos</h3>
    <dl>
      <dd>
        <table align="center">
          <thead>
            <tr>
              <th>Nombre y Apellido</th>
              <th>Usuario en GitHub</th>
              <th>GitHub</th>
              <th>Linkedin</th>
              <th>Portfolio</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td> Tatiana Gisel Candia </td>
              <td> tati2222 </td>
              <td>
                <a href="https://github.com/tati2222">
                  <img src="https://img.shields.io/badge/github-%23121011.svg?&style=for-the-badge&logo=github&logoColor=white"/>
                </a>
              </td>
              <td>
                <a href="https://www.linkedin.com/in/">
                  <img src="https://img.shields.io/badge/linkedin-%230A66C2.svg?&style=for-the-badge&logo=linkedin&logoColor=white"/>
                </a>
              </td>
              <td>
                <a href="">
                  <img src="https://img.shields.io/badge/Portfolio-%23000000.svg?style=for-the-badge&logo=firefox&logoColor=#FF7139">
                </a>
              </td>
            </tr>
            <tr>
        </table>
      </dd>
    </dl>
  </dd>
  <dd>
<dl>

<h3 align="center">Tester y Documentación</h3>
    <dl>
      <dd>
        <table align="center">
          <thead>
            <tr>
              <th>Nombre y Apellido</th>
              <th>Usuario en GitHub</th>
              <th>GitHub</th>
              <th>Linkedin</th>
              <th>Portfolio</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td> Maximiliano Pantaleff </td>
              <td> Maxi-009 </td>
              <td>
                <a href="https://github.com/Maxi-009">
                  <img src="https://img.shields.io/badge/github-%23121011.svg?&style=for-the-badge&logo=github&logoColor=white"/>
                </a>
              </td>
              <td>
                <a href="https://www.linkedin.com/in/">
                  <img src="https://img.shields.io/badge/linkedin-%230A66C2.svg?&style=for-the-badge&logo=linkedin&logoColor=white"/>
                </a>
              </td>
              <td>
                <a href="">
                  <img src="https://img.shields.io/badge/Portfolio-%23000000.svg?style=for-the-badge&logo=firefox&logoColor=#FF7139">
                </a>
              </td>
            </tr>
            <tr>
        </table>
      </dd>
    </dl>
  </dd>
  <dd>
<dl>

</br>

<h1 align="center"> 
  Tecnologías
</h1>

<table align="center">
  <thead>
    <tr>
      <th>Lenguaje de programación</th>
      <th>IDE</th>
      <th>Gestión de Tareas</th>
      <th>Trabajo Colaborativo</th>
      <th>DataSet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <img alt="Python" src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54">
      </td>
      <td>
        <img alt="Jupyter Notebook" src="https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white">
      </td>
      <td>
        <img alt="Trello" src="https://img.shields.io/badge/Trello-%23026AA7.svg?style=for-the-badge&logo=Trello&logoColor=white">
      </td>
      <td>
        <img alt="Google Drive" src="https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white">
      </td>
      <td>
        <img alt="Kaggle" src="https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white">
      </td>      
    </tr>
  </tbody>
</table>
<br>

## GitHub

[Grupo-7-TSCDIA](https://github.com/JDGA1997/Grupo-7-TSCDIA)

</br>

## Dataset
[Glaucoma Detection](https://www.kaggle.com/datasets/sshikamaru/glaucoma-detection)

</br>

## Trello

[Grupo N°7](https://trello.com/b/eatyLr9U/grupo-n7-tscdia-2024)

</div>
