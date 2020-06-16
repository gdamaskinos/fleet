/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include <jni.h>
#include <stdio.h>
#include "utils_JNITest.h"

JNIEXPORT void JNICALL Java_utils_JNITest_hello(JNIEnv *, jobject) {
	printf("Checking JNI...\n");
#ifdef __cplusplus
	printf("__cplusplus is defined\n");
#else
	printf("__cplusplus is NOT defined\n");
#endif
	fflush(stdout);
	return;
}
