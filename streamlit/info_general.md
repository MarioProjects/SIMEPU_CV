## Información General
La siguiente página sirve de aplicación para explorar las predicciones del modelo utilizado en SIMEPU para la 
clasificación multi etiqueta y segmentación ante muestras no vistas anteriormente por la red. Para ello,
se ha llevado a cabo un procedimiento de validación cruzada dividiendo los datos disponibles en 5 particiones o *folds*. 
Se utilizan 4 *folds* para entrenar la red y el restante para validar el funcionamiento obteniendo distintas métricas.
Así, se realizan los 5 posibles entrenamientos al variar el *fold* de validación a lo largo de los datos.

### Clasificación Multi-Etiqueta
Para el problema de clasificación multi-etiqueta considerando correcta la predicción si es **exactamente igual** 
al *ground truth*, es decir, contiene exactamente las clases del *ground truth*, ni más ni menos.

### Segmentación
Para el caso de la segmentación, para la cuantificación de las severidades, se utilizan diferentes redes:

  - Grietas: Se procesan todas las grietas longitudinales y transversales (volteadas) juntas. 
  - Huecos.
  - Parcheo.