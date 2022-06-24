# FeatureCloud imaging
### Imaging challenge solution for FeatureCloud.ai hackathon

This repo contains federated learning application for multi-label X-ray images classification. 

### Data
The dataset we used was ChestMNIST (part of medMNIST), which consists of 28x28 grayscaled images. 
Data was splitted into two parts in order to test application for 2 clients. 
Unfourtunately, we didn't manage to put data in `mnt/input/` folder inside docker, 
so we've put it inside the `fc_test/data/client_0` and `fc_test/data/client_1` directories.

### Model
We tried to tune some famous image classification models, ResNet50 and VGG16. In order to fit on small 28x28 images, 
some of max-pooling layers from original model architectures were removed. 
As optimizer we used `Adam`, and `BCEWithLogitsLoss` as loss-function.


### Some results

Here are some results in single-client mode for modified ResNet and VGG
model|dataset|AUC|ACC|
--------|-----|-------|-------|
ResNet50|train|0.82755|0.95046|
VGG16|train|0.78464|0.94955|
ResNet50|val|0.77206|0.94935|
VGG16|val|0.75625|0.94932|
ResNet50|test|0.77013|0.94748|
VGG16|test|0.75493|0.94797|

Some of them are better than official medMNIST benchmarks. 

We also tried to fit our modified VGG model in multi-clients mode and get average weights from them.

model|dataset|AUC|ACC|
--------|-----|-------|-------|
VGG16|train|0.44267|0.94887|
VGG16|val|0.43676|0.94917|
VGG16|test|0.44639|0.94726|
