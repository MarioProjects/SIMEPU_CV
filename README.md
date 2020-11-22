# SIMEPU
## AUTOMATIC IDENTIFICATION AND CLASSIFICATION OF URBAN PAVEMENT DISTRESSES THROUGH CONVOLUTIONAL NEURAL NETWORKS

The road network is one of the largest assets of a country and provides a fundamental basis
for its economic and social development. At the same time, its construction, maintenance, and use produce a
significant environmental impact.

Therefore, maintaining a road network in good condition is vital to reduce the cost of transporting people
and goods, as well as to avoid incurring additional costs for late maintenance that require a
rehabilitation or reconstruction.

In the following repository, we will try to solve the classification by the image of the different states
in which we can find the pavement. How the project can move towards classifying several incremental states, we will divide each experimental stage accordingly.

- Distresses: Alligator cracks / Longitudinal cracks / Transverse cracks / Raveling / Potholes / Patches
- No Distresses: Manholes / Road markings / Without Distresses

## Results Replication and Model Usage

## Classification

### Binary: Distresses vs No Distresses

In this stage, we want perform 'Damage' vs 'No Damage' binary classification. Models were pretrained with Imagenet. 

|   Model  | Fold | Accuracy | Precision | Recall |   F1   |
|:--------:|:----:|:--------:|:---------:|:------:|:------:|
| Resnet34 |   0  |  0.9950  |   0.9948  | 0.9948 | 0.9948 |
| Resnet34 |   1  |  0.9918  |   0.9895  | 0.9934 | 0.9915 |
| Resnet34 |   2  |  0.9899  |   0.9883  | 0.9908 | 0.9895 |
| Resnet34 |   3  |  0.9924  |   0.9905  | 0.9932 | 0.9918 |
| Resnet34 |   4  |  0.9943  |   0.9947  | 0.9933 | 0.9940 |
| Resnet34 | Mean |  0.9926  |   0.9915  | 0.9930 | 0.9923 |



### Distresses

In this stage, we want classify only the distresses. 
  
|   Model  | Fold | Accuracy | Balanced Accuracy |
|:--------:|:----:|:--------:|:-----------------:|
| Resnet34 |   0  |  0.9762  |       0.9759      |
| Resnet34 |   1  |  0.9836  |       0.9798      |
| Resnet34 |   2  |  0.9775  |       0.9804      |
| Resnet34 |   3  |  0.9828  |       0.9746      |
| Resnet34 |   4  |  0.9801  |       0.9747      |
| Resnet34 | Mean |  0.9804  |       0.9770      |

 

### All classes

We want compare our proposed 2 network framework with only 1 model that classifies all 9 different classes.

|   Model  | Fold | Accuracy | Balanced Accuracy |
|:--------:|:----:|:--------:|:-----------------:|
| Resnet34 |   0  |  0.9836  |       0.9787      |
| Resnet34 |   1  |  0.9824  |       0.9795      |
| Resnet34 |   2  |  0.9729  |       0.9757      |
| Resnet34 |   3  |  0.9798  |       0.9713      |
| Resnet34 |   4  |  0.9861  |       0.9797      |
| Resnet34 | Mean |  0.9809  |       0.9769      |

