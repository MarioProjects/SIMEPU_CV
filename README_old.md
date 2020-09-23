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
| resnet18     |     ce     |    sgd    |  224x224   |  steps 0.1    |         No        |   ------------   |    90.05%    |
| resnet18     |     ce     |    sgd    |  224x224   |  steps 0.01   |         No        |   ------------   |    90.55%    |
| resnet18*    |     ce     |    adam   |  224x224   |  steps 0.001  |         No        |   ------------   |    92.80%    |
| resnet18*    |     ce     |    adam   |  224x224   |  steps 0.001  |         Si        |   ------------   |    94.00%    |
| resnet18*    |     ce     |    adam   |  224x224   |  steps 0.001  |         Si        |   WeightedLoss   |    94.36%    |
| resnet34*    |     ce     |    adam   |  224x224   |  steps 0.001  |         Si        |   WeightedLoss   |    93.14%    |
| resnet34*    |     ce     |    adam   |  224x224   |  steps 0.0001 |         Si        |   WeightedLoss   |    93.65%    |
| resnet50*    |     ce     |    adam   |  224x224   |  steps 0.001  |         Si        |   WeightedLoss   |    93.05%    |
| seresnext50* |     ce     |    adam   |  224x224   |  steps 0.001  |         Si        |   WeightedLoss   |    94.12%    |

*: Preentrenado en Imagenet

Matriz de confusión del mejor modelo:
![Best Model Confusion Matrix](results/2019/resnet18_adam_256to224_lr0.001_DA_pretrained_weightedLoss/confusion_matrix.jpg "Best Model Confusion Matrix")

## Daño vs. No Daño

En esta segunda etapa tratamos de dar solución a la clasificación de las clases `Daño` y `No Daño`:
  - Daño: Grietas en forma de piel de cocodrilo / Grietas longitudinales / Grietas transversales / Huecos / Meteorización y desprendimiento / Parcheo
  - No Daño: Alcantarillado / Marca vial / Sin Daño

|     Model    | Criterion  | Optimizer |  Img Size  |  LR strategy  | Data Augmentation |      Extra       |   Accuracy   |
|:------------:|:----------:|:---------:|:----------:|:-------------:|:-----------------:|:----------------:|:------------:|
| resnet18*    |     ce     |    adam   |  512x512   |  steps 0.001  |         Si        |   ------------   |    97.25%    |
| resnet18*    |     ce     |    adam   |  224x224   |  steps 0.01   |         Si        |   ------------   |    70.86%    |
| resnet18*    |     ce     |    adam   |  224x224   |  steps 0.001  |         Si        |   ------------   |    97.19%    |
| resnet18*    |     ce     |    adam   |  224x224   |  steps 0.0001 |         Si        |   ------------   |    97.54%    |
| resnet34*    |     ce     |    adam   |  224x224   |  steps 0.01   |         Si        |   ------------   |    68.10%    |
| resnet34*    |     ce     |    adam   |  224x224   |  steps 0.001  |         Si        |   ------------   |    87.07%    |
| resnet34*    |     ce     |    adam   |  224x224   |  steps 0.0001 |         Si        |   ------------   |    97.68%    |

*: Preentrenado en Imagenet

- Area bajo curva ROC: 0.9954
  -  True Positive 99.31% para FP ratio 10.00%
  -  True Positive 96.96% para FP ratio 1.00%
  -  True Positive 95.15% para FP ratio 0.50%
  -  True Positive 94.18% para FP ratio 0.40%
  -  True Positive 92.18% para FP ratio 0.30%
  -  True Positive 79.16% para FP ratio 0.20%
  -  True Positive 44.12% para FP ratio 0.10%