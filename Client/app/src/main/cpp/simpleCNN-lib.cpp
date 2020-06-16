/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include <string.h>
#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <android/log.h>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include "../../../../../commonLib/cpp_utils/Base64.h"
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <time.h>
#include "../../../../../commonLib/simpleCNN/CNN/cnn.h"
#include "../../../../../commonLib/simpleCNN/simple.h"

vector<layer_t *> layers;
conv_layer_t *layer1;
relu_layer_t *layer2;
pool_layer_t *layer3;
fc_layer_t *layer4;
vector<tensor_t<float>> x_mini_batch, y_mini_batch;

extern "C"
JNIEXPORT jbyteArray JNICALL Java_apps_simpleCNN_SimpleCNNGradientGenerator_getGradients(JNIEnv * env, jobject) {

    std::vector<float> v;
    __android_log_print(ANDROID_LOG_DEBUG, "INFO", "Sending numbers: ");
    for (int i=30; i<40; i++) {
        v.push_back(i + 0.123456);
        __android_log_print(ANDROID_LOG_DEBUG, "INFO", "%.6f ", v.back());
    }

    std::string encoded = Base64::encode(v);

    const char* response = encoded.c_str();

    jbyteArray array = env->NewByteArray(strlen(response));
    env->SetByteArrayRegion(array, 0, strlen(response), (jbyte*)response);

    return array;
}

extern "C"
JNIEXPORT void JNICALL Java_apps_simpleCNN_SimpleCNNGradientGenerator_fetchNative(JNIEnv * env, jobject, jbyteArray input) {

    jbyte* buffer = env->GetByteArrayElements(input, NULL);
    jsize size = env->GetArrayLength(input);
    std::string encoded = (char *) buffer;

    std::vector<float> netParams =  Base64::decodeFloat(encoded.substr(0, size));

    free(layer1);
    free(layer2);
    free(layer3);
    free(layer4);

    layer1 = new conv_layer_t(1, 5, 8, x_mini_batch[0].size);        // 28 * 28 * 1 -> 24 * 24 * 8
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
JNIEXPORT int JNICALL Java_apps_simpleCNN_SimpleCNNGradientGenerator_fetchMiniBatch(JNIEnv * env, jobject, jbyteArray input) {
    srand(1);

    jbyte* buffer = env->GetByteArrayElements(input, NULL);
    jsize size = env->GetArrayLength(input);
    std::string encoded = (char *) buffer;

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
    std::vector<float> ret =  Base64::decodeFloat(encoded.substr(0, size));
    int idx = 0;
    int batchSize = ret[idx++];
    int x = ret[idx++];
    int y = ret[idx++];
    int z = ret[idx++];
    int numLabels = ret[idx++];
    __android_log_print(ANDROID_LOG_DEBUG, "INFO", "batchSize: %d", batchSize);
    __android_log_print(ANDROID_LOG_DEBUG, "INFO", "featureSize: %d %d %d", x, y, z);
    __android_log_print(ANDROID_LOG_DEBUG, "INFO", "numLabels: %d", numLabels);

    x_mini_batch.clear();
    y_mini_batch.clear();
    for (int example=0; example<batchSize; example++) {
        // get features
        tensor_t<float> temp(x, y, z);
        for (int i=0; i<temp.getSize(); i++)
            temp.data[i] = ret[idx++];
        x_mini_batch.push_back(temp);

        // get label
        tensor_t<float> templ(numLabels, 1, 1); // one-hot vector
        for (int i=0; i<templ.getSize(); i++)
            templ.data[i] = 0;
        templ.data[(int) ret[idx++]] = 1;
        y_mini_batch.push_back(templ);

    }
    return batchSize;
}

extern "C"
JNIEXPORT void JNICALL Java_apps_simpleCNN_SimpleCNNGradientGenerator_printParamsNative(JNIEnv * env, jobject, jbyteArray input) {

    jbyte* buffer = env->GetByteArrayElements(input, NULL);
    jsize size = env->GetArrayLength(input);
    std::string encoded = (char *) buffer;

    std::vector<float> ret =  Base64::decodeFloat(encoded.substr(0, size));


    __android_log_print(ANDROID_LOG_DEBUG, "INFO", "Got numbers: ");
    for (int i=0; i<ret.size(); i++)
        __android_log_print(ANDROID_LOG_DEBUG, "INFO", "%.6f ", ret[i]);

}
