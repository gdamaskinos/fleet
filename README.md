# FLeet

Author: Georgios Damaskinos (georgios.damaskinos@gmail.com)

_FLeet_ is a middleware introduced in [FLeet: Online Federated Learning via Staleness
Awareness and Performance Prediction](https://arxiv.org/pdf/2006.07273.pdf).

## Quick Start

The following steps train a CNN on the non-IID MNIST dataset.
1. Setup
    * Ubuntu 18.04.3 LTS
    * Python 3.6
    * Java 1.8.0_77
    * Maven 3.6.3
    * Gradle 4.10.1
    * Android API level 25
    * [Create and start an Android emulator](client_cmd_deployment/README.md)
    * Export variables:
      * JAVA_HOME
      * ANDROID_HOME
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
* Set profiling method in [_MasterOrchestrator.java_](Server/src/main/java/coreComponents/MasterOrchestrator.java#L85)

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

