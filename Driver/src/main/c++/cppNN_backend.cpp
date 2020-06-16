/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <jni.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include "../../../../commonLib/cpp_utils/Base64.h"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <time.h>
//#include <tchar.h>

//#define MOJO_CV3

#include "../../../../commonLib/cppNN/mojo.h"
#include "../../../../commonLib/cppNN/util.h"
#include "../../../../commonLib/cppNN/cost.h"
#include "../../../../commonLib/cppNN/mnist_parser.h"
#include "../../../../commonLib/cppNN/cifar_parser.h"

#ifndef DISTILLATION_MODE
#define DISTILLATION_MODE 0
#endif

std::string solver = "sgd";
// ==== setup the network  - when you train you must specify an optimizer ("sgd", "rmsprop", "adagrad", "adam")
mojo::network cnn(solver.c_str());

std::vector<std::vector<float>> test_images;
std::vector<int> test_labels;
std::vector<std::vector<float>> train_images;
std::vector<int> train_labels;

static float accuracy, error;

bool outlier = false;

// performs validation testing
// returns <error, accuracy>
std::tuple<float, float> test(mojo::network &cnn, const std::vector<std::vector<float>> &test_images, const std::vector<int> &test_labels, int sample = -1)
{
    // use progress object for simple timing and status updating

    int out_size = cnn.out_size(); // we know this to be 10 for MNIST
    int correct_predictions = 0;
    int record_cnt = (int)test_images.size();
    if (sample > -1)
        record_cnt = sample;

    float error = 0;
    #pragma omp parallel for reduction(+:correct_predictions) schedule(dynamic)
    for (int k = 0; k<record_cnt; k++)
    {
        const std::tuple<int, double> result = cnn.predict_class(test_images[k].data());
        error += mojo::mse::cost(std::get<1>(result), test_labels[k],0,0);
        const double prediction = std::get<0>(result);
        if (prediction == test_labels[k]) correct_predictions += 1;
    }
    error /= (float) record_cnt;
    float accuracy = (float)correct_predictions / record_cnt*100.f;
    return std::make_tuple(error, accuracy);
}

extern "C"
JNIEXPORT void JNICALL Java_apps_cppNN_CppNNModel_initializeNative(JNIEnv * env, jobject, jstring prefix) {
    srand(1);

    const char *nativeString = env->GetStringUTFChars(prefix, 0);
    std::string data_path(nativeString);

    env->ReleaseStringUTFChars(prefix, nativeString);

	// !! the threading must be enabled with thread count prior to loading or creating a model !!
	cnn.enable_external_threads();
	cnn.set_smart_training(false); // automate training

    // TODO change with the appropriate dataset
	/* MNIST dataset */
	// augment data random shifts only
	cnn.set_random_augmentation(1,1,0,0,mojo::edge);

	// calls MNIST::parse_test_data
	if (!mnist::parse_test_data(data_path, test_images, test_labels)) { std::cerr << "error: could not parse data.\n"; return; }

	if (outlier) {
		// keep only outlier class 0 for the evaluation
		for (int i=0; i<test_labels.size(); i++)
			if (test_labels[i] != 0) {
				test_images.erase(test_images.begin() + i);
				test_labels.erase(test_labels.begin() + i);
				i--;
			}
	}

	// to construct the model through API calls...
	cnn.push_back("I1", "input 28 28 1");				// MNIST is 28x28x1
	cnn.push_back("C1", "convolution 5 8 1 elu");		// 5x5 kernel, 20 maps. stride 1. out size is 28-5+1=24
	cnn.push_back("P1", "semi_stochastic_pool 3 3");	// pool 3x3 blocks. stride 3. outsize is 8
	cnn.push_back("C2i", "convolution 1 16 1 elu");		// 1x1 'inception' layer
	cnn.push_back("C2", "convolution 5 48 1 elu");		// 5x5 kernel, 200 maps.  out size is 8-5+1=4
	cnn.push_back("P2", "semi_stochastic_pool 2 2");	// pool 2x2 blocks. stride 2. outsize is 2x2
	cnn.push_back("FC2", "softmax 10");					// 'flatten' of 2x2 input is inferred
	// connect all the layers. Call connect() manually for all layer connections if you need more exotic networks.
	cnn.connect_all();

	/* CIFAR dataset */
	// augment data random shifts only
// 	cnn.set_random_augmentation(2,2,0,0,mojo::edge);
// 
// 	// calls CIFAR::parse_test_data depending on 'using'
// 	int testImagesPerFile = 10000;
// 	if (!cifar::parse_test_data(data_path, test_images, test_labels, testImagesPerFile)) { std::cerr << "error: could not parse data.\n"; return; }
// 	
// 	// to construct the model through API calls...
//     cnn.push_back("I1", "input 32 32 3");
//     cnn.push_back("C1", "convolution 3 16 1 elu");
//     cnn.push_back("P1", "max_pool 3 2");
//     cnn.push_back("C2", "convolution 3 64 1 elu");
//     cnn.push_back("P2", "max_pool 4 4");
//     cnn.push_back("local4", "fully_connected 384 relu");
//     cnn.push_back("local5", "fully_connected 192 relu");
//     //cnn.push_back("FC2", "softmax 100"); // CIFAR-100
//     cnn.push_back("FC2", "softmax 10"); // CIFAR-10
// 
// 	cnn.connect_all();

	std::cout << "==  Network Configuration  ====================================================" << std::endl;
	std::cout << cnn.get_configuration() << std::endl;

	printf("Test data size: %lu\n", test_labels.size());
	
	error = -1;
	accuracy = -1;
}

extern "C"
JNIEXPORT jbyteArray JNICALL Java_apps_cppNN_CppNNModel_getParamsNative(JNIEnv * env, jobject) {

	std::string params = cnn.getParams();

    const char* response = params.c_str();

    jbyteArray array = env->NewByteArray(strlen(response));
    env->SetByteArrayRegion(array, 0, strlen(response), (jbyte*)response);

    return array;
}

extern "C"
JNIEXPORT void JNICALL Java_apps_cppNN_CppNNModel_printParamsNative(JNIEnv * env, jobject, jbyteArray input) {

    jbyte* buffer = env->GetByteArrayElements(input, NULL);
    jsize size = env->GetArrayLength(input);
    std::string encoded = (char *) buffer;

    std::vector<float> ret =  Base64::decodeFloat(encoded.substr(0, size));

    printf("Model parameters: %lu\n", ret.size());
    fflush(stdout);

    env->ReleaseByteArrayElements(input, buffer, 0);

}

extern "C"
JNIEXPORT void JNICALL Java_apps_cppNN_CppNNModel_fetchParamsNative(JNIEnv * env, jobject, jbyteArray input, jint x, jint y, jint z) {

    jbyte* buffer = env->GetByteArrayElements(input, NULL);
    jsize size = env->GetArrayLength(input);
    std::string encoded = (char *) buffer;

    // Needed for continuing training on this model

    if (DISTILLATION_MODE)
        cnn.start_epoch("distillation");    
    else
        cnn.start_epoch("cross_entropy");
    cnn.set_random_augmentation(1,1,0,0,mojo::edge);
    cnn.clear();

    std::istringstream ss(encoded);
    cnn.read(ss);

    env->ReleaseByteArrayElements(input, buffer, 0);
}

extern "C"
JNIEXPORT void JNICALL Java_apps_cppNN_CppNNModel_evaluateNative(JNIEnv * env, jobject) {
	std::tuple<float, float> res = test(cnn, test_images, test_labels);
	error = std::get<0>(res);
	accuracy = std::get<1>(res);
}

extern "C"
JNIEXPORT float JNICALL Java_apps_cppNN_CppNNModel_accuracyNative(JNIEnv * env, jobject) {
	return accuracy;
}

extern "C"
JNIEXPORT float JNICALL Java_apps_cppNN_CppNNModel_errorNative(JNIEnv * env, jobject) {
	return error;
}

