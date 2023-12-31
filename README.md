# FeatureCloud imaging
[![FeatureCloud logo](https://featurecloud.eu/wp-content/uploads/2019/04/fc_logo.svg)](https://featurecloud.eu)

This repo contains federated learning application for multi-label X-ray images classification. This app was the solution to Imaging challenge of 1st FeatureCloud.ai hackathon, held in 2022 in Hamburg. 

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

We also tried to fit our modified VGG model in 2 clients mode and get average weights from them.

model|dataset|AUC|ACC|
--------|-----|-------|-------|
VGG16|train|0.44267|0.94887|
VGG16|val|0.43676|0.94917|
VGG16|test|0.44639|0.94726|

Here we can see strange AUC values, maybe we made some mistakes while computing average weights.

These are metrics for local model from `client_2` on 2 parts of data.

model|dataset|AUC|ACC|
--------|-----|-------|-------|
VGG_client_2|train_part_1|0.73221|0.94918
VGG_client_2|val_part_1|0.74091|0.94920
VGG_client_2|test_part_1|0.73307|0.94756


model|dataset|AUC|ACC|
--------|-----|-------|-------|
VGG_client_2|train_part_2|0.76809|0.94902
VGG_client_2|val_part_2|0.73734|0.94910
VGG_client_2|test_part_2|0.73090|0.94776

As we can see, local model predictions are better than predictions of an aggregated model.

## Contributors

- [Tim Vaitko](https://github.com/timvoytko)
- [Akim Malyshchyk](https://github.com/akimich11)
