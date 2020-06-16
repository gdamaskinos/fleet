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

vector<layer_t *> layers2;
conv_layer_t *temp1;
relu_layer_t *temp2;
pool_layer_t *temp3;
fc_layer_t *temp4;

extern "C"
JNIEXPORT void JNICALL Java_apps_simpleCNN_SimpleCNNUpdater_initUpdater(JNIEnv * env, jobject, jdouble lrate, jdouble momentum) {
    srand(1);
    LEARNING_RATE = lrate;
    MOMENTUM = momentum;

	printf("Native Learning rate: %.5f\n", LEARNING_RATE);
	printf("Native Momentum: %.1f\n", MOMENTUM);
	fflush(stdout);
}


extern "C"
JNIEXPORT jbyteArray JNICALL Java_apps_simpleCNN_SimpleCNNUpdater_getParametersNative(JNIEnv * env, jobject) {

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
JNIEXPORT void JNICALL Java_apps_simpleCNN_SimpleCNNUpdater_fetchParamsNative(JNIEnv * env, jobject, jbyteArray input, jint x, jint y, jint z) {

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
JNIEXPORT void JNICALL Java_apps_simpleCNN_SimpleCNNUpdater_printParamsNative(JNIEnv * env, jobject, jbyteArray input) {

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
JNIEXPORT void JNICALL Java_apps_simpleCNN_SimpleCNNOfflineSampler_initSampler(JNIEnv * env, jobject, jstring prefix) {
    srand(1);

	const char *nativeString = env->GetStringUTFChars(prefix, 0);
	char trainxPath[200], trainyPath[200], valxPath[200], valyPath[200];

	strcpy(trainxPath, nativeString);
	strcat(trainxPath, "training_features.csv");

	strcpy(trainyPath, nativeString);
	strcat(trainyPath, "training_labels.csv");

	strcpy(valxPath, nativeString);
	strcat(valxPath, "small_features.csv");

	strcpy(valyPath, nativeString);
	strcat(valyPath, "small_labels.csv");

	env->ReleaseStringUTFChars(prefix, nativeString);

    val_x = load_csv_data(valxPath, 28, 28, 1);
    val_y = load_csv_data(valyPath, 10, 1, 1);


}

extern "C"
JNIEXPORT jbyteArray JNICALL Java_apps_simpleCNN_SimpleCNNOfflineSampler_getMiniBatch(JNIEnv * env, jobject, jint batch_size) {
    srand(1);

    vector<tensor_t<float>> x_mini_batch;
    vector<tensor_t<float>> y_mini_batch;

	// Select a random mini batch
    for (int j = 0; j < 2; j++) {
        int max = val_x.size() - 1;
        int min = 0;
        int index = min + (rand() % (max - min + 1));
        index = j;

        x_mini_batch.push_back(val_x[index]);
        y_mini_batch.push_back(val_y[index]);
    }

    /*
     * miniBatch[0] = batchSize
     * miniBatch[1] = featureSize.x
     * miniBatch[2] = featureSize.y
     * miniBatch[3] = featureSize.z
     * miniBatch[4] = numLabels
     * miniBatch[5..5+featureSize] -> example1
     * miniBatch[5+featureSize] -> label1
     * ...
     */
    vector<float> miniBatch, temp;
    miniBatch.push_back(2); // TODO batch_size);
    miniBatch.push_back(x_mini_batch[0].size.x);
    miniBatch.push_back(x_mini_batch[0].size.y);
    miniBatch.push_back(x_mini_batch[0].size.z);
    miniBatch.push_back(y_mini_batch[0].getSize());
    for (int i=0; i<x_mini_batch.size(); i++) {
    	// features
    	temp = x_mini_batch[i].flatData();
    	miniBatch.insert(miniBatch.end(), temp.begin(), temp.end());

    	//label
    	temp = y_mini_batch[i].flatData(); // one-hot vector
    	int label;
    	for (label=0; label<temp.size(); label++)
    		if (temp[label] == 1)
    			break;
    	miniBatch.push_back(label);
    }

    std::string encoded = Base64::encode(miniBatch);

    const char* response = encoded.c_str();

    jbyteArray array = env->NewByteArray(strlen(response));
    env->SetByteArrayRegion(array, 0, strlen(response), (jbyte*)response);

    return array;

}
