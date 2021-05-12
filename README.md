# AAPV_Clasif_Frutas

Software del trabajo "Clasificación de frutas: implementación hardware"desarrollado para la asignatura de Arquitecturas de Altas Prestaciones para Visión. 

Está compuesto por dos apartados independientes: Pytorch-Brevitas y FINN. Respectivamente, estos se corresponden con el desarrollo de una CNN cuantizada para la clasificación de frutas del dataset "Fruit360" y con su optimización, implementación y despliegue en hardware.

## Pytorch-Brevitas

Se incluyen dos scripts:

- __ClasificadorFrutas.py__: Script usado para entrenar, testear y exportar el modelo de CNN. Si se quiere reproducir la ejecución bastaría con indicar los directorios de training y test de Fruit360 mediante la modificación de las variables "*carpetaTraining*" y "*carpetaTest*". Los modelos y los ficheros ONNX se guardarán en la carpeta "modelos".
- __commonQuant.py__: Script que define un conjunto de wrappers para diferentes capas cuantizadas de la librería de Brevitas.

### Dependencias usadas:

- Brevitas 0.4.0.
- ONNX 1.5.0.
- Fruit360: https://github.com/Horea94/Fruit-Images-Dataset
- Scikit-Learn 0.21.3.

