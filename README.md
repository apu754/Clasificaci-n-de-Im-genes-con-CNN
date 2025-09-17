# Clasificación de Imágenes con CNN + Grad-CAM (Colab)

Este proyecto implementa una **red neuronal convolucional (CNN)** en **TensorFlow/Keras**, entrenada en Google Colab para clasificar imágenes de la base **Caltech-101**.  
Incluye el uso de **Grad-CAM** para visualizar qué partes de la imagen el modelo utilizó para su predicción.

---

## Estructura del proyecto

```bash
├── Parcial_IA2.ipynb # Notebook principal (Google Colab)
├── test_images/ # Imágenes de prueba
│ ├── leopard.jpg
│ ├── airplane.jpg
│ └── stop_sign.jpg
│ └── motorcycle.jpg
└── README.md
```
---

## Pasos en el Notebook
1. Abre el cuaderno:  
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/apu754/Clasificaci-n-de-Im-genes-con-CNN/blob/main/Clasificaci%C3%B3n_de_Im%C3%A1genes_con_CNN.ipynb)

### A. Recolección de datos
- Se utiliza el dataset [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02).
- Se organizan las imágenes en carpetas `train/`, `val/`, `test/`.

### B. Selección de clases
- Se seleccionan 5 clases:  
  *`airplanes, Motorbikes, Faces_easy, Leopards, stop_sign`*  
- Se hace un split **70% train / 15% val / 15% test**.

### C. Definición de la CNN
- Se entrena una CNN desde cero con 3 bloques convolucionales + capa densa final.  
- Arquitectura:
  - Conv2D + BatchNorm + ReLU
  - MaxPooling
  - GlobalAveragePooling
  - Dropout
  - Dense (softmax)

### D. Aumentación de datos
- Se aplica `RandomFlip`, `RandomRotation`, `RandomZoom`, etc.  
- Se asegura que cada clase tenga al menos 400 imágenes.

### E. Entrenamiento y validación
- Se entrena con `SGD(learning_rate=0.01, momentum=0.9)`.  
- Se monitorea con callbacks:
  - `EarlyStopping`
  - `ReduceLROnPlateau`
  - `ModelCheckpoint`

- Se visualizan curvas de **accuracy** y **loss**, además de la **matriz de confusión**.

---

## Visualización de resultados

### Curvas de entrenamiento
- Accuracy y pérdida en train y validación.

### Matriz de confusión
- Distribución de aciertos y errores por clase.

### Grad-CAM
- Visualiza las **zonas de la imagen** que activaron más al modelo.  
- Ejemplo de salida: el mapa de calor resalta la cabeza y pelaje del *leopard*.

---

## Probar con nuevas imágenes

En el notebook (`Parcial_IA2.ipynb`), se puede probar con imágenes externas:

```python
from tensorflow.keras.preprocessing import image
import numpy as np

img_path = "/content/test_images/leopard.jpg"
img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)
pred_class = np.argmax(pred, axis=1)[0]

print("Predicción:", class_names[pred_class], "-", f"{pred[0][pred_class]*100:.2f}%")
Resultado esperado:


Predicción: Leopards - 98.73%
```

---

## Notas
- El proyecto es multiclase (5 clases) y puede ampliarse a más.
- Grad-CAM no es un detector, pero ayuda a entender qué partes de la imagen fueron importantes.
- Los resultados finales alcanzan ~98% de accuracy en validación.



---
