/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include <android/log.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <jni.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include "../../../../../commonLib/cpp_utils/Base64.h"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <time.h>
#include <thread>
#include <pthread.h>
#include <sys/resource.h>
#include <unistd.h>
#include <atomic>
//#include <tchar.h>
#ifndef DISTILLATION_MODE
#define DISTILLATION_MODE 0
#endif
//#define MOJO_CV3
//#define CALOREE

#include "../../../../../commonLib/cppNN/mojo.h"
#include "../../../../../commonLib/cppNN/util.h"
#include "../../../../../commonLib/cppNN/cost.h"
#include "../../../../../commonLib/cppNN/mnist_parser.h"

#ifdef CALOREE
#include <sched.h>
#include "caloree.h"
#else
// need to manually set the number of threads according to static policy for resource allocation
int NUMBER_OF_THREADS = 4;//std::thread::hardware_concurrency();
#endif
int E;
double lrate, sigma, C;
int mini_batch_size, numLabels, numFeatures;
std::string solver = "sgd";
mojo::network cnn(solver.c_str());

std::vector<std::vector<float>> x_mini_batch;
std::vector<std::vector<float>> x_pred_teacher_mini_batch;
std::vector<int> y_mini_batch;

// have a slot for 8-threads (maximum possible) and then unblock depending on the configuration
pthread_mutex_t block_lock[8];
pthread_cond_t block_nonzero[8];
bool block[8];

pthread_mutex_t master_lock;
pthread_cond_t master_nonzero;
bool master_block;

std::atomic<bool> *bitset;
int current_example_wait;
//std::atomic<int> active_threads = {0};

/* this function is run by the pthreads */
void *train_cnn(void *idx)
{
    int thread_index = (int)idx;
#ifdef CALOREE
    cpu_set_t my_set;
    CPU_ZERO(&my_set);
    CPU_SET(thread_index, &my_set);
    sched_setaffinity(0, sizeof(cpu_set_t), &my_set);

    nice(-20); // increase the thread priority; -20 should be enough to make it higher than everything else

    for (int k = 0; k < mini_batch_size; k++) {
        if (block[thread_index]) {
            pthread_mutex_lock(&block_lock[thread_index]);
            while (block[thread_index])
                pthread_cond_wait(&block_nonzero[thread_index], &block_lock[thread_index]);
            pthread_mutex_unlock(&block_lock[thread_index]);
        }

        if (!bitset[k].exchange(true)) {
            if (k == current_example_wait - 1) {
                pthread_mutex_lock(&master_lock);
                if (master_block)
                    pthread_cond_signal(&master_nonzero);
                master_block = false;
                pthread_mutex_unlock(&master_lock);
            }

            cnn.train_class(x_mini_batch[k].data(),
                            y_mini_batch[k],
                            &(x_pred_teacher_mini_batch[k]),
                            thread_index);
        }
    }
#else
    nice(-20);

    for (int k = 0; k < mini_batch_size; k++) {
        if (!bitset[k].exchange(true)) {
            cnn.train_class(x_mini_batch[k].data(), y_mini_batch[k],
                            &(x_pred_teacher_mini_batch[k]), thread_index);
        }
    }
#endif

    return NULL;
}

extern "C"
JNIEXPORT jintArray JNICALL Java_apps_cppNN_CppNNGradientGenerator_getClassDist(JNIEnv * env, jobject) {

    jint classDist[numLabels];

    for (int i=0; i<numLabels; i++)
        classDist[i] = 0;

    for (int i=0; i<mini_batch_size; i++)
        classDist[y_mini_batch[i]]++;

    for (int i=0; i<numLabels; i++)
        __android_log_print(ANDROID_LOG_DEBUG, "INFO", "class dist %d", classDist[i]);


    jintArray array = env->NewIntArray(numLabels);
    env->SetIntArrayRegion(array, (jsize) 0, (jsize) numLabels, classDist);

    return array;

}

extern "C"
JNIEXPORT jbyteArray JNICALL Java_apps_cppNN_CppNNGradientGenerator_getGradients(JNIEnv * env, jobject) {

    std::vector<float> prevGrads, grads;

    // gradient computation
	bitset = (std::atomic<bool>*) malloc(mini_batch_size * sizeof(std::atomic<bool>));
    for (int i = 0; i < mini_batch_size; i++) {
        bitset[i].store(false);
    }

#ifdef CALOREE
    pthread_t thread[8];

	// gradient computation
    current_example_wait = 0;
    for (int i = 0; i < 8; i++) {
        pthread_mutex_init(&block_lock[i], NULL);
        pthread_cond_init(&block_nonzero[i], NULL);
        block[i] = true;
        pthread_create(&thread[i], NULL, train_cnn, (void*)i);
        __android_log_print(ANDROID_LOG_WARN, "INFO", "pthread_create %lu", thread[i]);
    }

    long long exec_time;
    conf_tuple_t conf = get_config(get_average_speedup(), mini_batch_size / get_nrounds());
    config *c1 = std::get<0>(conf);
    config *c2 = std::get<1>(conf);
    int littleThreads1 = std::get<0>(c1->get_thread_conf());
    int bigThreads1 = std::get<1>(c1->get_thread_conf());
    int nexamples1 = c1->get_nexamples();
    int littleThreads2 = std::get<0>(c2->get_thread_conf());
    int bigThreads2 = std::get<1>(c2->get_thread_conf());
    int nexamples2 = c2->get_nexamples();
    delete c1;
    delete c2;
    exec_time = install_conf(bitset, &current_example_wait, littleThreads1, bigThreads1, nexamples1,
                             littleThreads2, bigThreads2, nexamples2, block_lock, block_nonzero,
                             block, &master_lock, &master_nonzero, &master_block);

    while (current_example_wait < mini_batch_size) {

        conf = get_config(compute_next_xup(exec_time), mini_batch_size / get_nrounds());
        c1 = std::get<0>(conf);
        c2 = std::get<1>(conf);
        littleThreads1 = std::get<0>(c1->get_thread_conf());
        bigThreads1 = std::get<1>(c1->get_thread_conf());
        nexamples1 = c1->get_nexamples();
        littleThreads2 = std::get<0>(c2->get_thread_conf());
        bigThreads2 = std::get<1>(c2->get_thread_conf());
        nexamples2 = c2->get_nexamples();
        delete c1;
        delete c2;
        exec_time = install_conf(bitset, &current_example_wait, littleThreads1, bigThreads1, nexamples1,
                     littleThreads2, bigThreads2, nexamples2, block_lock, block_nonzero, block,
                                 &master_lock, &master_nonzero, &master_block);

    }
    unlock_all(block_lock, block_nonzero, block);

#else
    pthread_t thread[NUMBER_OF_THREADS];
    for (int i = 0; i < NUMBER_OF_THREADS; i++) {
        pthread_create(&thread[i], NULL, train_cnn, (void*)i);
    }
#endif

    // Joining all threads
    for(int i = 0; i < 8; i++)
    {
        int ret = pthread_join(thread[i], NULL);
    }

    free(bitset);
    // get gradients
    if (sigma > 0)
        grads = cnn.DPgradients(C, sigma);
    else
        grads = cnn.gradients();

    if (prevGrads.empty())
        prevGrads = grads;
    else
        cnn.addGradients(prevGrads, grads);

    cnn.set_learning_rate(lrate);
    cnn.descent(grads);

    for (int i = 152; i < 159; ++i)
        __android_log_print(ANDROID_LOG_DEBUG, "INFO", "SEND GRAD: %.6f", prevGrads[i]);

    std::string encoded = Base64::encode(prevGrads);
    const char* response = encoded.c_str();

    jbyteArray array = env->NewByteArray(strlen(response));
    env->SetByteArrayRegion(array, 0, strlen(response), (jbyte*)response);

    return array;
}

extern "C"
JNIEXPORT void JNICALL Java_apps_cppNN_CppNNGradientGenerator_fetchNative(JNIEnv * env, jobject, jbyteArray input) {

    jbyte* buffer = env->GetByteArrayElements(input, NULL);
    jsize size = env->GetArrayLength(input);
    std::string encoded = (char *) buffer;

    // Needed for continuing training on this model
    if(DISTILLATION_MODE)
        cnn.start_epoch("distillation");
    else
        cnn.start_epoch("cross_entropy");
    cnn.enable_external_threads(8); // create MOJO net that can employ 8 threads
    cnn.set_mini_batch_size(mini_batch_size);
    cnn.set_random_augmentation(1,1,0,0,mojo::edge);

#ifdef CALOREE
    //__android_log_print(ANDROID_LOG_WARN, "INFO", "Batch size: %d\n", mini_batch_size);
    //init(mini_batch_size, (double) 5100 / 1000, (double) 1470 / mini_batch_size, 5, 0.5); //h10
    init(mini_batch_size, (double) 1092 / 1000, (double) 10000 / mini_batch_size, 5, 0.5); //s8
    //init(mini_batch_size, (double) 1112.0 / 1000, (double) 17400 / mini_batch_size, 5, 0.5); //S7
    //init(mini_batch_size, (double) 6478 / 1000, (double) 15500 / mini_batch_size, 5, 0.5); //Xperia
    //init(mini_batch_size, (double) 4586 / 1000, (double) 11200 / mini_batch_size, 5, 0.5); //S4mini
#endif

    cnn.clear();
    std::istringstream ss(encoded);
    cnn.read(ss);


}

extern "C"
JNIEXPORT int JNICALL Java_apps_cppNN_CppNNGradientGenerator_fetchMiniBatch(JNIEnv * env, jobject, jbyteArray input) {
    srand(1);

    jbyte* buffer = env->GetByteArrayElements(input, NULL);
    jsize size = env->GetArrayLength(input);
    std::string encoded = (char *) buffer;

    /*
     * miniBatch[0] = batchSize
     * miniBatch[1] = featureSize
     * miniBatch[2] = numLabels
     * miniBatch[3..3+featureSize] -> example1
     * miniBatch[3+featureSize] -> label1
     * ...
     */
    std::vector<float> ret =  Base64::decodeFloat(encoded.substr(0, size));

    int idx = 0;
    E = ret[idx++];
    sigma = ret[idx++];
    C = ret[idx++];
    lrate = ret[idx++];
    mini_batch_size = ret[idx++];
    numFeatures = ret[idx++];
    numLabels = ret[idx++];
    __android_log_print(ANDROID_LOG_DEBUG, "INFO", "E: %d", E);
    __android_log_print(ANDROID_LOG_DEBUG, "INFO", "Sigma: %g", sigma);
    __android_log_print(ANDROID_LOG_DEBUG, "INFO", "C: %g", C);
    __android_log_print(ANDROID_LOG_DEBUG, "INFO", "lrate: %g", lrate);
    __android_log_print(ANDROID_LOG_DEBUG, "INFO", "batchSize: %d", mini_batch_size);
    __android_log_print(ANDROID_LOG_DEBUG, "INFO", "featureSize: %d", numFeatures);
    __android_log_print(ANDROID_LOG_DEBUG, "INFO", "numLabels: %d", numLabels);

    x_mini_batch.clear();
    y_mini_batch.clear();
    if(DISTILLATION_MODE)
        x_pred_teacher_mini_batch.clear();
    for (int example=0; example<mini_batch_size; example++) {
        // get features
        std::vector<float> temp;
        std::vector<float> temp2;
        for (int i=0; i<numFeatures; i++)
            temp.push_back(ret[idx++]);
        x_mini_batch.push_back(temp);

        
        if(DISTILLATION_MODE){
            // get predictions
            for (int i=0; i<numLabels; i++)
                temp2.push_back(ret[idx++]);
            x_pred_teacher_mini_batch.push_back(temp2);
        }

        // get label
        y_mini_batch.push_back((int) ret[idx++]);

    }

    return mini_batch_size;
}

extern "C"
JNIEXPORT void JNICALL Java_apps_cppNN_CppNNGradientGenerator_printParamsNative(JNIEnv * env, jobject, jbyteArray input) {

    jbyte* buffer = env->GetByteArrayElements(input, NULL);
    jsize size = env->GetArrayLength(input);
    std::string encoded = (char *) buffer;

    std::vector<float> ret =  Base64::decodeFloat(encoded.substr(0, size));


    __android_log_print(ANDROID_LOG_DEBUG, "INFO", "Got numbers: ");
    for (int i=0; i<ret.size(); i++)
        __android_log_print(ANDROID_LOG_DEBUG, "INFO", "%.6f ", ret[i]);

}
