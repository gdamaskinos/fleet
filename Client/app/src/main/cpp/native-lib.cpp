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
#include <unistd.h>
#include <android/log.h>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <thread>
#include <time.h>

extern "C"
JNIEXPORT jbyteArray JNICALL
Java_coreComponents_MainActivity_transferBytes(JNIEnv * env, jobject instance, jbyteArray input) {
    //copy bytes from java
    jbyte* buffer = env->GetByteArrayElements(input, NULL);

    jsize size = env->GetArrayLength(input);

    buffer[size] = '\0';

    __android_log_print(ANDROID_LOG_DEBUG, "INFO", "Received: %s", buffer);
    env->ReleaseByteArrayElements(input, buffer, 0);


    // TEST HEAP SIZE ALLOCATION
    // https://github.com/hqt/heapsize-testing
    long testBytes = 1024*1024*100;
    char *blob = (char*) malloc(sizeof(char) * testBytes);
    for (long i=0; i<testBytes; i++)
        blob[i] = 'a';

    __android_log_print(ANDROID_LOG_DEBUG, "INFO", "Number of threads: %d\n", std::thread::hardware_concurrency());

    if (NULL == blob) {
        __android_log_print(ANDROID_LOG_DEBUG, "INFO", "Failed to allocate memory\n");
    } else {
        __android_log_print(ANDROID_LOG_DEBUG, "INFO", "Allocated %ld bytes", sizeof(char) * testBytes);
    }

    usleep(2000);
    free(blob);

    // copy bytes to java
    const char *response = "COPY DAT!";
    __android_log_print(ANDROID_LOG_DEBUG, "INFO", "Sending: %s", response);
    jbyteArray array = env->NewByteArray(strlen(response));
    env->SetByteArrayRegion(array, 0, strlen(response), (jbyte*)response);

    return array;
}
