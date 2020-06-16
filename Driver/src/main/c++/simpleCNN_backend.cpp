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
#include "../../../../commonLib/cpp_utils/Base64.h"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <time.h>
#include "../../../../commonLib/simpleCNN/CNN/cnn.h"
#include "../../../../commonLib/simpleCNN/simple.h"

vector<layer_t *> layers;
conv_layer_t *layer1;
relu_layer_t *layer2;
pool_layer_t *layer3;
fc_layer_t *layer4;
vector<tensor_t<float>> train_x, train_y, val_x, val_y;

extern "C"
JNIEXPORT void JNICALL Java_apps_simpleCNN_SimpleCNNModel_initializeNative(JNIEnv * env, jobject, jint x, jint y, jint z) {
    srand(1);

    LEARNING_RATE = 0.01;
    MOMENTUM = 0.9;
    WEIGHT_DECAY = 0;

    val_x = load_csv_data("datasets/mnist/mnist_small_features.csv", 28, 28, 1);
    val_y = load_csv_data("datasets/mnist/mnist_small_labels.csv", 10, 1, 1);

    layer1 = new conv_layer_t(1, 5, 8, {x,y,z});        // 28 * 28 * 1 -> 24 * 24 * 8
    layer2 = new relu_layer_t(layer1->out.size);
    layer3 = new pool_layer_t(2, 2, layer2->out.size);                // 24 * 24 * 8 -> 12 * 12 * 8
    layer4 = new fc_layer_t(layer3->out.size, 10);                    // 4 * 4 * 16 -> 10

    layers.clear();
    layers.push_back((layer_t *) layer1);
    layers.push_back((layer_t *) layer2);
    layers.push_back((layer_t *) layer3);
    layers.push_back((layer_t *) layer4);
    int evalCount = 0;

    clock_t begin = clock();
 	float acc;
 	long ep;
    for (ep = 0; ep < 10; ep++) {

         train_SGD_minibatch(layers, val_x, val_y, 1);
 //        train_SGD_step(layers, x_mini_batch[0], y_mini_batch[0]);

     }
	printf("TRAINED!\n");
	fflush(stdout);
	double elapsed_time = double(clock() - begin) / CLOCKS_PER_SEC * 1000;
	auto val_acc = compute_accuracy(layers, val_x, val_y);
    auto val_loss = compute_mae_loss(layers, val_x, val_y);

    printf("eval:%d,%ld,-1,-1,%.4f,%.4f,%.2f\n", evalCount++, ep, val_loss, val_acc, elapsed_time);
    fflush(stdout);
}

extern "C"
JNIEXPORT jbyteArray JNICALL Java_apps_simpleCNN_SimpleCNNModel_getParamsNative(JNIEnv * env, jobject) {

    vector<float> netParams;
    vector<float> tempP1 = layer1->flatParams();
    vector<float> tempP2 = layer4->flatParams();
    netParams.insert(netParams.end(), tempP1.begin(), tempP1.end());
    netParams.insert(netParams.end(), tempP2.begin(), tempP2.end());

    std::string encoded = Base64::encode(netParams);

    const char* response = encoded.c_str();

    jbyteArray array = env->NewByteArray(strlen(response));
    env->SetByteArrayRegion(array, 0, strlen(response), (jbyte*)response);

    return array;

}


extern "C"
JNIEXPORT void JNICALL Java_apps_NativeCNN_NativeCNNUpdater_printParamsNative(JNIEnv * env, jobject, jbyteArray input) {

    jbyte* buffer = env->GetByteArrayElements(input, NULL);
    jsize size = env->GetArrayLength(input);
    std::string encoded = (char *) buffer;

    std::vector<float> ret =  Base64::decodeFloat(encoded.substr(0, size));


    printf("Got Numbers: ");
    for (int i=0; i<ret.size(); i++)
    	printf("%.6f ", ret[i]);
    printf("\n");
    fflush(stdout);

}

extern "C"
JNIEXPORT void JNICALL Java_apps_simpleCNN_SimpleCNNModel_fetchParamsNative(JNIEnv * env, jobject, jbyteArray input, jint x, jint y, jint z) {

    jbyte* buffer = env->GetByteArrayElements(input, NULL);
    jsize size = env->GetArrayLength(input);
    std::string encoded = (char *) buffer;

    std::vector<float> netParams =  Base64::decodeFloat(encoded.substr(0, size));

    free(layer1);
    free(layer2);
    free(layer3);
    free(layer4);

    layer1 = new conv_layer_t(1, 5, 8, {x,y,z});        // 28 * 28 * 1 -> 24 * 24 * 8
    layer2 = new relu_layer_t(layer1->out.size);
    layer3 = new pool_layer_t(2, 2, layer2->out.size);                // 24 * 24 * 8 -> 12 * 12 * 8
    layer4 = new fc_layer_t(layer3->out.size, 10);                    // 4 * 4 * 16 -> 10

    int s1 = layer1->setParams(netParams);
    netParams.erase(netParams.begin(), netParams.begin() + s1);
    layer4->setParams(netParams);

    layers.clear();
    layers.push_back((layer_t *) layer1);
    layers.push_back((layer_t *) layer2);
    layers.push_back((layer_t *) layer3);
    layers.push_back((layer_t *) layer4);

}

extern "C"
JNIEXPORT float JNICALL Java_apps_simpleCNN_SimpleCNNModel_accuracyNative(JNIEnv * env, jobject) {
	return compute_accuracy(layers, val_x, val_y);
}

extern "C"
JNIEXPORT float JNICALL Java_apps_simpleCNN_SimpleCNNModel_errorNative(JNIEnv * env, jobject) {
	return compute_mae_loss(layers, val_x, val_y);
}

