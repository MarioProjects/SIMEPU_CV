# SIMEPU
## SISTEMA INTEGRAL DE MANTENIMIENTO EFICIENTE DE PAVIMENTOS URBANOS

La red vial es uno de los mayores bienes patrimoniales de un país y proporciona una base fundamental 
para su desarrollo económico y social. Al mismo tiempo, su construcción, mantenimiento y uso produce un 
significativo impacto medioambiental.

Por ello, el mantenimiento de una red vial en buen estado es vital para reducir costes de transporte de personas 
y bienes, así como para no incurrir en sobrecostes por mantenimientos tardíos que obligan a una 
rehabilitación o reconstrucción.

En el siguiente repositorio trataremos de dar solución a la clasificación por imágen de los diferentes estados
en los que podemos encontrar el pavimento. Como el proyecto puede avanzar hacia la clasificación de un número
incremental de estados, dividiremos cada etapa experimental en consecuencia. 

Para visualizar todos los experimentos realizados ejecutar: `tensorboard --logdir=results/logs`

``
pip install git+https://github.com/ildoonet/cutmix
``

## Clasificación: Todas las clases

En esta primera etapa tratamos de dar solución a la clasificación de 9 diferentes estados iniciales:
  - Alcantarillado
  - Grietas en forma de piel de cocodrilo
  - Grietas longitudinales
  - Grietas transversales
  - Huecos
  - Marca vial
  - Meteorización y desprendimiento
  - Parcheo
  - Sin daño

|     Model    | Criterion  | Optimizer |  Img Size  |  LR strategy  | Data Augmentation |      Extra       | Val Accuracy |
|:------------:|:----------:|:---------:|:----------:|:-------------:|:-----------------:|:----------------:|:------------:|
| resnet34*    |     ce     |    adam   |  224x224   |  steps 0.0001 |         Si        |   WeightedLoss   |    97.88%    |
| resnet50*    |     ce     |    adam   |  224x224   |  steps 0.0001 |         Si        |   WeightedLoss   |    98.21%    |

- [RESNET34] Train Analysis
  - Accuracy of Parcheo : 100.00% 
  - Accuracy of Marca vial : 99.48% 
  - Accuracy of Sin daño : 98.47% 
  - Accuracy of Grietas transversales : 97.77% 
  - Accuracy of Huecos : 91.25% 
  - Accuracy of Grietas longitudinales : 97.75% 
  - Accuracy of Meteorización y desprendimiento : 85.71% 
  - Accuracy of Grietas en forma de piel de cocodrilo : 100.00% 
  - Accuracy of Alcantarillado : 98.48% 

- [RESNET50] Train Analysis
  - Accuracy of Parcheo : 100.00% 
  - Accuracy of Marca vial : 100.00% 
  - Accuracy of Sin daño : 98.47% 
  - Accuracy of Grietas transversales : 98.32% 
  - Accuracy of Huecos : 90.00% 
  - Accuracy of Grietas longitudinales : 97.75% 
  - Accuracy of Meteorización y desprendimiento : 91.84% 
  - Accuracy of Grietas en forma de piel de cocodrilo : 100.00% 
  - Accuracy of Alcantarillado : 100.00% 

*: Preentrenado en Imagenet

Matriz de confusión del mejor modelo:
![Best Model Confusion Matrix](results/resnet50_adam_256to224_lr0.0001_DA_pretrained_Full/confusion_matrix.png "Best Model Confusion Matrix")

## Daño vs. No Daño

En esta segunda etapa tratamos de dar solución a la clasificación de las clases `Daño` y `No Daño`:
  - Daño: Grietas en forma de piel de cocodrilo / Grietas longitudinales / Grietas transversales / Huecos / Meteorización y desprendimiento / Parcheo
  - No Daño: Alcantarillado / Marca vial / Sin Daño

|     Model    | Criterion  | Optimizer |  Img Size  |  LR strategy  | Data Augmentation |      Extra       |   Accuracy   |
|:------------:|:----------:|:---------:|:----------:|:-------------:|:-----------------:|:----------------:|:------------:|
| resnet34*    |     ce     |    adam   |  224x224   |  steps 0.0001 |         Si        |   ------------   |    99.18%    |
| resnet50*    |     ce     |    adam   |  224x224   |  steps 0.0001 |         Si        |   ------------   |    99.34%    |

*: Preentrenado en Imagenet


- [RESNET34] Area bajo curva ROC: 0.9992
  - VP 100.00% para FP ratio 10.00%
  - VP 99.83% para FP ratio 5.00%
  - VP 99.31% para FP ratio 2.50%
  - VP 99.14% para FP ratio 1.00%
  - VP 97.75% para FP ratio 0.50%
  - VP 97.75% para FP ratio 0.40%
  - VP 93.44% para FP ratio 0.30%
  - VP 87.56% para FP ratio 0.20%
  - VP 87.56% para FP ratio 0.10%

- [RESNET50] Area bajo curva ROC: 0.9994
  - VP 100.00% para FP ratio 10.00%
  - VP 99.83% para FP ratio 5.00%
  - VP 99.83% para FP ratio 2.50%
  - VP 99.14% para FP ratio 1.00%
  - VP 98.27% para FP ratio 0.50%
  - VP 98.27% para FP ratio 0.40%
  - VP 95.51% para FP ratio 0.30%
  - VP 91.36% para FP ratio 0.20%
  - VP 91.36% para FP ratio 0.10%

Esto es, para un ratio de Falsos positivos (NoDaño clasificado como Daño) del 0.4% (4 de cada 1000 muestras), 
en el cual molestaríamos a un operario experto, los casos de Daño clasificados como tal (VP) son del 97.75%,
colandosenos por lo tanto alrededor de 22 muestras dañadas de cada 1000.

![Best Model ROC](results/resnet50_adam_256to224_lr0.0001_DA_pretrained_Binary/curva_ROC.png "Best Model ROC")

![Best Model ROC Zoom](results/resnet50_adam_256to224_lr0.0001_DA_pretrained_Binary/roc_zoom.png "Best Model ROC Zoom")
 
## Daños

Por otra parte, queremos estudiar cómo sería la clasificación de los daños al tratarlos estos de forma aislada; Clasificación de las clases `Daño`:
  - Daño: Grietas en forma de piel de cocodrilo / Grietas longitudinales / Grietas transversales / Huecos / Meteorización y desprendimiento / Parcheo
  
|     Model    | Criterion  | Optimizer |  Img Size  |  LR strategy  | Data Augmentation |      Extra       |   Accuracy   |
|:------------:|:----------:|:---------:|:----------:|:-------------:|:-----------------:|:----------------:|:------------:|
| resnet34*    |     ce     |    adam   |  224x224   |  steps 0.0001 |         Si        |   ------------   |    98.80%    |
| resnet50*    |     ce     |    adam   |  224x224   |  steps 0.0001 |         Si        |   ------------   |    99.14%    |

*: Preentrenado en Imagenet

- [RESNET34] Train Analysis
  - Accuracy of Parcheo : 100.00% 
  - Accuracy of Grietas transversales : 97.85% 
  - Accuracy of Huecos : 95.18% 
  - Accuracy of Grietas longitudinales : 100.00% 
  - Accuracy of Meteorización y desprendimiento : 100.00% 
  - Accuracy of Grietas en forma de piel de cocodrilo : 100.00%

- [RESNET50] Train Analysis  
  - Accuracy of Parcheo : 98.48% 
  - Accuracy of Grietas transversales : 98.39% 
  - Accuracy of Huecos : 97.59% 
  - Accuracy of Grietas longitudinales : 99.46% 
  - Accuracy of Meteorización y desprendimiento : 100.00% 
  - Accuracy of Grietas en forma de piel de cocodrilo : 100.00% 
  
 ![Best Model Damaged Confusion Matrix](results/resnet50_adam_256to224_lr0.0001_DA_pretrained_OnlyDamaged/confusion_matrix.png "Best Model Damaged Confusion Matrix")