# AAPV_Clasif_Frutas

Software del trabajo "Clasificación de frutas: implementación hardware"desarrollado para la asignatura de Arquitecturas de Altas Prestaciones para Visión. 

Está compuesto por dos apartados independientes: Pytorch-Brevitas y FINN. Respectivamente, estos se corresponden con el desarrollo de una CNN cuantizada para la clasificación de frutas del dataset "Fruit360" y con su optimización, implementación y despliegue en hardware.

## Pytorch-Brevitas

### Scripts

- __ClasificadorFrutas.py__: Script usado para entrenar, testear y exportar el modelo de CNN. Si se quiere reproducir la ejecución bastaría con indicar los directorios de training y test de Fruit360 mediante la modificación de las variables "*carpetaTraining*" y "*carpetaTest*". Los modelos y los ficheros ONNX se guardarán en la carpeta "modelos".
- __commonQuant.py__: Script que define un conjunto de wrappers para diferentes capas cuantizadas de la librería de Brevitas.

En este caso, solo haría falta ejecutar el script __ClasificadorFrutas.py__ desde consola para la realización de los pasos indicados.

### Dependencias usadas:

- Brevitas 0.4.0.
- ONNX 1.5.0.
- Fruit360: https://github.com/Horea94/Fruit-Images-Dataset
- Scikit-Learn 0.21.3.

## FINN

### Scripts

- __notebook_final.py__: Notebook modificado a partir del ejemplo "end2end_example/bnn-pynq" del proyecto FINN. Se modifica para poder adaptarlo a la red convolucional cuantizada generada por los scripts de Pytorch-Brevitas. Se han verificado todos los pasos hasta el apartado "5: Deployment and Remote Execution", por lo que solo se incluyen los pasos anteriores a dicho apartado.

Para la ejecución de este notebook, cabe realizar los pasos especificados en https://finn.readthedocs.io/en/latest/getting_started.html, y colocar el notebook en la carpeta "finn/notebooks" para su ejecución vía el servicio Jupyter que despliega el Docker de FINN. Además, habría que colocar el modelo ONNX exportado anteriormente en el directorio "finn/", así como indicar el nombre de dicho fichero ONNX en el segundo bloque de código del notebook.

Tras la ejecución completa del notebook, se encontrarán todos los ficheros generados en el directorio "/tmp/finn_dev_<USUARIO>/vivado_zynq_proj_<...>" del contenedor que ejecuta FINN. 
  
NOTA: Si alguno de los pasos falla durante la ejecución del notebook, por favor, póngase en contacto con el autor de este repositorio.

### Dependencias usadas:

- Las indicadas en https://finn.readthedocs.io/en/latest/getting_started.html
