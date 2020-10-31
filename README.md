<div class="row">
  <div class="column">
    <img src="badges/artifacts_evaluated_functional.jpg" align="right" width="15%" height="15%">
  </div>
  <div class="column">
    <img src="badges/artifacts_available.jpg" align="right" width="15%" height="15%">
  </div>
</div>

# FLeet

Author: Georgios Damaskinos (georgios.damaskinos@gmail.com)

_FLeet_ is a middleware introduced in [FLeet: Online Federated Learning via Staleness
Awareness and Performance Prediction](https://arxiv.org/pdf/2006.07273.pdf).

## Quick Start

The following steps train a CNN on the non-IID MNIST dataset.
1. Setup
    * Ubuntu 18.04.3 LTS
    * gcc 7.5
    * Python 3.6
    * Java 1.8.0_77
    * Maven 3.6.3
    * Gradle 4.10.1
    * Android API level 25
    * [Create and start an Android emulator](client_cmd_deployment/README.md)
    * Export variables:
      * JAVA_HOME
      * ANDROID_HOME
    * Download Python numpy package 
2. Download and extract the [4 files of the MNIST dataset](http://yann.lecun.com/exdb/mnist/)
3. Compile, deploy and run FLeet
   ```
   ./local_deploy.sh 5554 9992 ~/log /path/to/mnist/ 0 0 1000 3000 0.0005 1 1 0 0 0 0 0
   ```

Detailed information about the deployment options for FLeet is available [here](deployment.md).

## Usage

The following contains information about each module of FLeet along with instructions for implementing a new application.

### Server
Performs the descent computation and updates the model
* Edit [_SPUpdater.java_](Server/src/main/java/apps/SPUpdater.java) and [_SPSampler.java_](Server/src/main/java/apps/SPSampler.java)
* Set profiling method in [_MasterOrchestrator.java_](Server/src/main/java/coreComponents/MasterOrchestrator.java#L94)

### Driver
Initializes the Server and performs a periodic evaluation
* Edit [_Driver.java_](Driver/src/main/java/coreComponents/Driver.java)

### Android Client
Performs the gradient computation
* Edit [_SPGradientGenerator.java_](Client/app/src/main/java/apps/SPGradientGenerator.java)

## Existing applications

### cppNN
Image classification application based on Convolutional Neural Networks and written in C++.

* Enable/disable compression 
  * Enable compressed SGD: ```#define DISTILLATION_MODE 1```
  * Disable compressed SGD: ```#define DISTILLATION_MODE 0```
  * Files: [Server cppNN_backend.cpp](Server/src/main/c%2B%2B/cppNN_backend.cpp), [Driver cppNN_backend.cpp](Driver/src/main/c%2B%2B/cppNN_backend.cpp), [Client cppNN-lib.cpp](Client/app/src/main/cpp/cppNN-lib.cpp), [network.h](commonLib/cppNN/network.h)

### MLP
Image classification application based on multilayer perceptron and written in Java.

### LR
Facebook check-in prediction application based on logistic regression and written in Java.

### DL4J
Port of [DL4J library](https://deeplearning4j.org/) on FLeet.

## Datasets

### MNIST
See [Quick Start](#quick-start).

### EMNIST
* Download Python dependencies:
  * Sklearn
  * Scipy
  * Numpy
* ```python emnistParser.py --help```
* For parsing into FLeet change to the DL4J app:
  * See all `TODO` notes in the [Driver](Driver/src/main/java/coreComponents/Driver.java).
  * Change the application in the Server and Client (see [Usage](#usage)).
  * Give the output of the parser as input dataset for FLeet.

### CIFAR

* Download CIFAR-10 from [here](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz) or CIFAR-100 from [here](https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz) and extract.
* For CIFAR-100:
  ```bash
  cd cifar-100-binary/
  mv train.bin data_batch_1.bin
  mv test.bin test_batch.bin
  ```
* For parsing into FLeet:
  * Comment out [these lines](Driver/src/main/c%2B%2B/cppNN_backend.cpp#L91-L117) and comment in [these lines](Driver/src/main/c%2B%2B/cppNN_backend.cpp#L119-L138). Choose [this line](Driver/src/main/c%2B%2B/cppNN_backend.cpp#L135) for CIFAR-100 or [this line](Driver/src/main/c%2B%2B/cppNN_backend.cpp#L136) for CIFAR-10.
  * Comment out [these lines](Server/src/main/c%2B%2B/cppNN_backend.cpp#L400-L401) and comment in [these lines](Server/src/main/c%2B%2B/cppNN_backend.cpp#L404-L405) for CIFAR-100 or [these lines](Server/src/main/c%2B%2B/cppNN_backend.cpp#L406-L407) for CIFAR-10.
  * Comment in [this line](Server/src/main/c%2B%2B/cppNN_backend.cpp#L408) for both datasets.
