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
#include <math.h>
//#include <tchar.h>

//#define MOJO_CV3

#include "../../../../commonLib/cppNN/mojo.h"
#include "../../../../commonLib/cppNN/util.h"
#include "../../../../commonLib/cppNN/cost.h"
#include "../../../../commonLib/cppNN/mnist_parser.h"
#include "../../../../commonLib/cppNN/cifar_parser.h"

/* mojo related definitions */
#ifndef DISTILLATION_MODE
#define DISTILLATION_MODE 0
#endif

std::string solver = "sgd";
// ==== setup the network  - when you train you must specify an optimizer ("sgd", "rmsprop", "adagrad", "adam")
mojo::network cnn(solver.c_str());
//if(DISTILLATION_MODE)
mojo::network teacher("sgd");

// local updates
int E;

// DP settings
double sigma, C;

/* dataset related definitions */
int numLabels;
std::vector<std::vector<float>> test_images;
std::vector<int> test_labels;
std::vector<std::vector<float>> train_images;
std::vector<int> train_labels;

bool iid = false;
bool outlier = false;
int numClients = 10;
int currClientID;
std::vector<std::vector<int>> buckets; // each bucket contains the shuffled dataset indices of each client
std::vector<int> bucketIdx; // current index for sampling the bucket (non-overlapping samples)
std::vector<std::vector<float>> sorted_images;
std::vector<int> sorted_labels;

jsize lrates_size;
double *lrates_vec;

int seed = 1;
/* staleness-aware learning related definitions */
/**
 * List of models for on-demand staleness updates
 * stale[list.size() -1] (i.e. end of the list) is the most recent version of the model
 * stale[list.size() -2] is the model before one update
 * ...
 */
std::vector<mojo::network*> models;
//mojo::network* oldestModel; // useful for kardam prev model sending

/**
 * determines the version of the model that is going to be sent to the next request
 */
int priority, currEpoch;
mojo::network tempNet;

// performs validation testing
// returns <error, accuracy>
std::tuple<float, float> test(mojo::network &cnn, const std::vector<std::vector<float>> &test_images, const std::vector<int> &test_labels, int sample = -1)
{
    // use progress object for simple timing and status updating
  //  mojo::progress progress((int)test_images.size(), "  testing: ");

    int out_size = cnn.out_size(); // we know this to be 10 for MNIST
    int correct_predictions = 0;
    int record_cnt = (int)test_images.size();
    if (sample > -1)
        record_cnt = sample;

    float error = 0;
    //#pragma omp parallel for reduction(+:correct_predictions) schedule(dynamic)
    for (int k = 0; k<record_cnt; k++)
    {
        const std::tuple<int, double> result = cnn.predict_class(test_images[k].data());
        error += mojo::mse::cost(std::get<1>(result), test_labels[k],0,0);
        const double prediction = std::get<0>(result);
        if (prediction == test_labels[k]) correct_predictions += 1;
  //      if (k % 1000 == 0) progress.draw_progress(k);
    }
    error /= (float) record_cnt;
    float accuracy = (float)correct_predictions / record_cnt*100.f;
    return std::make_tuple(error, accuracy);
}

template <typename T>
std::vector<size_t> sort_indexes(const std::vector<T> &v) {

	// initialize original index locations
	std::vector<size_t> idx(v.size());
	std::iota(idx.begin(), idx.end(), 0);

	// sort indexes based on comparing values in v
	std::sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

	return idx;
}

extern "C"
JNIEXPORT int JNICALL Java_apps_cppNN_CppNNUpdater_getNumLabels(JNIEnv * env, jobject) {
	return numLabels;
}

extern "C"
JNIEXPORT bool JNICALL Java_apps_cppNN_CppNNUpdater_hasOutlier(JNIEnv * env, jobject) {
	return outlier;
}

extern "C"
JNIEXPORT void JNICALL Java_apps_cppNN_CppNNUpdater_setCurrEpoch(JNIEnv * env, jobject, jint ep) {
	currEpoch = ep;
}
extern "C"
JNIEXPORT int JNICALL Java_apps_cppNN_CppNNUpdater_getCurrEpoch(JNIEnv * env, jobject) {
	return currEpoch;
}
extern "C"
JNIEXPORT void JNICALL Java_apps_cppNN_CppNNUpdater_setPriority(JNIEnv * env, jobject, jint p) {
	priority = p;
}
extern "C"
JNIEXPORT int JNICALL Java_apps_cppNN_CppNNUpdater_getPriority(JNIEnv * env, jobject) {
	return priority;
}

extern "C"
JNIEXPORT double JNICALL Java_apps_cppNN_CppNNUpdater_getLrate(JNIEnv * env, jobject) {
	return cnn.get_learning_rate();
}

extern "C"
JNIEXPORT void JNICALL Java_apps_cppNN_CppNNUpdater_initUpdater(JNIEnv * env, jobject, jdoubleArray vec, jint E_, jdouble sigma_, jdouble C_) {
    srand(seed);
    printf("Seed: %d\n", seed);

    lrates_size = env->GetArrayLength(vec);
    lrates_vec = env->GetDoubleArrayElements(vec, NULL);

    E = E_;
    sigma = sigma_;
    C = C_;
	cnn.set_learning_rate(lrates_vec[0]);
	printf("Native Learning rate: %.5f\n", cnn.get_learning_rate());
	fflush(stdout);

    mojo::network *cnnNew = new mojo::network(solver.c_str());

    if(DISTILLATION_MODE)
        cnnNew->start_epoch("distillation");
    else
        cnnNew->start_epoch("cross_entropy");
//              cnn2.enable_external_threads();
//              cnn2.set_smart_training(false); // automate training
    //cnn2.set_learning_rate(initial_learning_rate);
    cnnNew->set_random_augmentation(1,1,0,0,mojo::edge);

    std::string params = cnn.getParams();
    std::istringstream ss(params);
    cnnNew->clear();
    cnnNew->read(ss);

    models.push_back(cnnNew);
    priority = 0;
    currEpoch = 0;

    if(DISTILLATION_MODE){
        printf("Added initial model to models\n"); fflush(stdout);

        float* teacher_layer;
        std::vector<float> temp;

        //Forward run on teacher for the specific sample
        teacher_layer = teacher.forward(train_images[0].data(), TEMPERATURE, -1, 1);

        //Find the last layer of the teacher model to extract the probabilities

        //printf("BEFORE INITIALIZATION OF UPDATER %d %d numLabels: %d\n",teacher_layer_cnt,teacher_last_layer_index,numLabels); fflush(stdout);

        for (int j = 0; j < numLabels; j++) {
            temp.push_back(teacher_layer[j]);
             //   printf("jTH: %d class is: %f\n",j,teacher_layer[j]);
        }

     //    printf("BEFORE INITIALIZATION OF UPDATER\n"); fflush(stdout);

        cnn.train_class(train_images[0].data(),train_labels[0],&temp); // train with one example for initializing the dW_sets

    }
    else{
       printf("Added initial model to models\n"); fflush(stdout);

	   cnn.train_class(train_images[0].data(), train_labels[0],NULL); // train with one example for initializing the dW_sets
    }

}

extern "C"
JNIEXPORT jbyteArray JNICALL Java_apps_cppNN_CppNNUpdater_getModelParametersNative(JNIEnv * env, jobject, jint p) {

	std::vector<float> modelParams;
	printf("Sending priority: %d\n", p); fflush(stdout);
	modelParams = models[p]->getModelParams();

    std::string encoded = Base64::encode(modelParams);

    const char* response = encoded.c_str();

    jbyteArray array = env->NewByteArray(strlen(response));
    env->SetByteArrayRegion(array, 0, strlen(response), (jbyte*)response);

    return array;
}

extern "C"
JNIEXPORT jbyteArray JNICALL Java_apps_cppNN_CppNNUpdater_getParametersNative(JNIEnv * env, jobject, jint p) {


    if(DISTILLATION_MODE){
        std::string params;
        std::vector<mojo::matrix*> unquantized_W;

        printf("Sending priority: %d\n", p); fflush(stdout);
        models[p]->save_model_weights(&unquantized_W);
        //Quantize the model's weights
        models[p]->quantization_weight_model();

        // Create the dictionary and encoding of the quantized params 
        params = models[p]->getParams();
        models[p]->load_model_weights(unquantized_W);

        const char* response = params.c_str();

        jbyteArray array = env->NewByteArray(strlen(response));
        env->SetByteArrayRegion(array, 0, strlen(response), (jbyte*)response);

        return array;
    }
    else{
        std::string params;
        printf("Sending priority: %d\n", p); fflush(stdout);
        params = models[p]->getParams();

        const char* response = params.c_str();

        jbyteArray array = env->NewByteArray(strlen(response));
        env->SetByteArrayRegion(array, 0, strlen(response), (jbyte*)response);

        return array;
    }
}

extern "C"
JNIEXPORT void JNICALL Java_apps_cppNN_CppNNUpdater_fetchParamsNative(JNIEnv * env, jobject, jbyteArray input) {

    jbyte* buffer = env->GetByteArrayElements(input, NULL);
    jsize size = env->GetArrayLength(input);
    std::string encoded = (char *) buffer;

    if(DISTILLATION_MODE)
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
JNIEXPORT void JNICALL Java_apps_cppNN_CppNNUpdater_printParamsNative(JNIEnv * env, jobject, jbyteArray input) {

    jbyte* buffer = env->GetByteArrayElements(input, NULL);
    jsize size = env->GetArrayLength(input);
    std::string encoded = (char *) buffer;

    std::vector<int> ret =  Base64::decodeInt(encoded);


    printf("Got Numbers: ");
    for (int i=0; i<ret.size(); i++)
    	printf("%d ", ret[i]);
    printf("\n");
    fflush(stdout);

    env->ReleaseByteArrayElements(input, buffer, 0);

}

extern "C"
JNIEXPORT int JNICALL Java_apps_cppNN_CppNNUpdater_modelsSize(JNIEnv * env, jobject) {
	return models.size();
}

extern "C"
JNIEXPORT void JNICALL Java_apps_cppNN_CppNNUpdater_descentNative(JNIEnv * env, jobject, jbyteArray input, jint clientBatchSize, jint staleSize) {

    jbyte* buffer = env->GetByteArrayElements(input, NULL);
    jsize size = env->GetArrayLength(input);
    std::string encoded = (char *) buffer;

    std::vector<float> ret =  Base64::decodeFloat(encoded.substr(0, size));


    printf("Updating gradient: ");
    for (int i = 152; i < 159; ++i)
    		printf("%.6f ", ret[i]);
    printf("\n");
    fflush(stdout);

    cnn.set_mini_batch_size(clientBatchSize);

    if(currEpoch < lrates_size){
    	cnn.set_learning_rate(lrates_vec[currEpoch]);
    	printf("Set new learning rate: %.6f\n", lrates_vec[currEpoch]);
    }

    cnn.descent(ret);
    currEpoch++;

    mojo::network *cnnNew = new mojo::network(solver.c_str());

    if(DISTILLATION_MODE)
        cnnNew->start_epoch("distillation");
    else
        cnnNew->start_epoch("cross_entropy");
    cnnNew->set_mini_batch_size(clientBatchSize);
//              cnn2.enable_external_threads();
//              cnn2.set_smart_training(false); // automate training
    //cnn2.set_learning_rate(initial_learning_rate);
    cnnNew->set_random_augmentation(1,1,0,0,mojo::edge);

    std::string params = cnn.getParams();
    std::istringstream ss(params);
    cnnNew->clear();
    cnnNew->read(ss);

    models.push_back(cnnNew);
    printf("Added new model to models...Current size:%lu\n", models.size()); fflush(stdout);

    if (models.size() > staleSize) { // keep at least one more version for requests on the same model version (due to kardam filtering)
    		printf("Removing oldest model as size = %lu\n", models.size()); fflush(stdout);
    		delete models[0];
    		models.erase(models.begin());
    }

    env->ReleaseByteArrayElements(input, buffer, 0);

}

extern "C"
JNIEXPORT void JNICALL Java_apps_cppNN_CppNNOfflineSampler_initSampler(JNIEnv * env, jobject, jstring prefix) {
    srand(seed);

	printf("IID: %d\n", iid);
	printf("Outlier: %d\n", outlier);
	fflush(stdout);
	
    const char *nativeString = env->GetStringUTFChars(prefix, 0);
    std::string data_path(nativeString);

    env->ReleaseStringUTFChars(prefix, nativeString);

    // TODO change with the appropriate dataset
	/* MNIST */
	if (!mnist::parse_train_data(data_path, train_images, train_labels)) { std::cerr << "error: could not parse data.\n"; return; }
	numLabels = 10;

	/* CIFAR */
//	int trainImagesPerFile = 50000; // CIFAR-100
//	numLabels = 100;
// 	int trainImagesPerFile = 10000; // CIFAR-10
// 	numLabels = 10;
// 	if (!cifar::parse_train_data(data_path, train_images, train_labels, trainImagesPerFile)) { std::cerr << "error: could not parse data.\n"; return; }

	
	if (!iid) {
	
		currClientID = 0;
	
		// sort data by label
		std::vector<size_t> indices = sort_indexes(train_labels);
	
		std::vector<int> range;
		for (int i=0; i< (int) train_images.size(); i++) {
			sorted_images.push_back(train_images[indices[i]]);
			sorted_labels.push_back(train_labels[indices[i]]);
			range.push_back(i);
		}
	
		int offset = 0;
		int numOutliers = 0;
		if (outlier) {
			// create outlier bucket
			numOutliers = 1;
			while (train_labels[indices[offset]] == 0) offset++;
	
			std::vector<int> bucket;
			bucket.insert(bucket.end(), range.begin(), range.begin() + offset);
			std::random_shuffle(bucket.begin(), bucket.end(), [](int i) -> int {return std::rand()%i;});
			buckets.push_back(bucket);
			bucketIdx.push_back(0); // start sampling from first element of bucket
			
			// print outlier buckets
			//for (int j=0; j< offset; j++)
			//	std::cout << sorted_labels[buckets[0][j]] << " ";
			//std::cout << std::endl;
			//std::cout << std::endl;
		}
	
		// create non-outlier shards
		std::vector<int> shards;
		for (int i=0; i< 2 * (numClients - numOutliers); i++)
			shards.push_back(i);
		std::random_shuffle(shards.begin(), shards.end(), [](int i) -> int {return std::rand()%i;});
	
		// create non-outlier buckets
		int bucketSize = (int) (train_images.size() - offset) / (numClients - numOutliers);
		int shardSize = (int) bucketSize / 2;
		printf("Num clients: %d\nOutlier Bucket size: %d\nBucket size: %d\nShard size: %d\n", numClients, offset, bucketSize, shardSize);
		for (int i=0; i<shards.size(); i=i+2) {
			std::vector<int> bucket;
			// insert shard1
			bucket.insert(bucket.end(), range.begin() + offset + shards[i]*shardSize,
					range.begin() + offset + (shards[i]+1)*shardSize);
			// insert shard2
			bucket.insert(bucket.end(), range.begin() + offset + shards[i+1]*shardSize,
					range.begin() + offset + (shards[i+1]+1)*shardSize);
	
			std::random_shuffle(bucket.begin(), bucket.end(), [](int i) -> int {return std::rand()%i;});
			buckets.push_back(bucket);
	
			bucketIdx.push_back(0); // each client starts sampling from first element of bucket
		}
	
	
		// print non-outlier buckets
		//for (int i=numOutliers; i< numClients; i++) {
		//	for (int j=0; j< bucketSize; j++)
		//		std::cout << sorted_labels[buckets[i][j]] << " ";
		//	std::cout << std::endl;
		//	std::cout << std::endl;
		//}
	
	}
	
    if(DISTILLATION_MODE){
        //
        //######## Teacher Network #############
        //
        if (!mnist::parse_test_data(data_path, test_images, test_labels)) { std::cerr << "error: could not parse test data.\n"; return; }

        //teacher.enable_external_threads();
    //  cnn.set_mini_batch_size(mini_batch_size);
        teacher.set_smart_training(false); // automate training
        teacher.set_learning_rate(0.01f);

        /* MNIST dataset */
        // augment data random shifts only
        teacher.set_random_augmentation(1,1,0,0,mojo::edge);

        teacher.push_back("I1", "input 28 28 1");              // MNIST is 28x28x1
        teacher.push_back("C1", "convolution 5 8 1 elu");      // 5x5 kernel, 20 maps. stride 1. out size is 28-5+1=24
        teacher.push_back("P1", "semi_stochastic_pool 3 3");   // pool 3x3 blocks. stride 3. outsize is 8
        teacher.push_back("C2i", "convolution 1 16 1 elu");        // 1x1 'inceptoin' layer
        teacher.push_back("C2", "convolution 5 48 1 elu");     // 5x5 kernel, 200 maps.  out size is 8-5+1=4
        teacher.push_back("P2", "semi_stochastic_pool 2 2");   // pool 2x2 blocks. stride 2. outsize is 2x2
        teacher.push_back("FC2", "softmax 10");                    // 'flatten' of 2x2 input is inferred
        //connect all the layers. Call connect() manually for all layer connections if you need more exotic networks.

        teacher.connect_all();
        int train_samples = (int)train_images.size();
        int max_steps = 3; // reaches ~80% accuracy
        int teacher_batch_size = 100;
        printf("Training teacher for %d steps. DO NOT MAKE COMPUTATION REQUESTS!\n", max_steps);
        fflush(stdout);

        float accuracy=0, error=0, old_accuracy = 0;
        int trainedCount = 0, epoch = 0, evalCount = 1;
        while (epoch < max_steps) {
            teacher.start_epoch("cross_entropy");
            // manually loop through data. batches are handled internally. if data is to be shuffled, the must be performed externally
            std::tuple<float, float> res;
            //std::cout <<" What happens? "<< train_samples <<std::endl;
            //#pragma omp parallel for schedule(dynamic) 
            for (int k = 0; k < train_samples; k++) {
                //std::cout<<"New train sample: "<< k <<" out of "<<train_samples<<std::endl;
                int max = train_images.size() - 1;
                int min = 0;
                int index = min + (rand() % (max - min + 1));
                //.data() is the pointer to the floats that represent the image in the vector C++11 stuff
                teacher.train_class(train_images[index].data(),train_labels[index],NULL);
                //if (k % 1000 == 0) progress.draw_progress(k);
                trainedCount++;
                if (trainedCount % teacher_batch_size == 0) {
                    epoch = trainedCount / teacher_batch_size;
                    teacher.sync_mini_batch(); // accumulate gradients
                    teacher.descent(); // update model
                    if (epoch % 10 == 0) {
                        res = test(teacher, test_images, test_labels);
                        error = std::get<0>(res);
                        accuracy = std::get<1>(res);
                        std::cout << "eval:" << evalCount++ << "," << epoch << ",-1,-1," << error << "," << accuracy <<  std::endl;
                    }
                    if (epoch >= max_steps)
                    	break;
                }
            }
            teacher.end_epoch();
        }
    }

    int train_samples = (int)train_images.size();
    std::cout << "Train data size: " << train_samples << std::endl;


}

std::vector<float> uniformSample(int batchSize) {

    std::vector<std::vector<float>> x_mini_batch;
    std::vector<int> y_mini_batch;

	// Select a random mini batch
    for (int j = 0; j < batchSize; j++) {
        int max = train_images.size() - 1;
        int min = 0;
        int index = min + (rand() % (max - min + 1));

        x_mini_batch.push_back(train_images[index]);
        y_mini_batch.push_back(train_labels[index]);
    }

    /*
     * miniBatch[0] = E
     * miniBatch[1] = sigma
     * miniBatch[2] = C
     * miniBatch[3] = lrate
     * miniBatch[4] = batchSize
     * miniBatch[5] = featureSize
     * miniBatch[6] = numLabels
     * miniBatch[7..7+featureSize] -> example1
     * miniBatch[7+featureSize] -> label1
    OR IN CASE OF DISTILLATION WE NEED THE LABELS NUMBER
     * miniBatch[7+featureSize..7+featureSize+numLabels]
     * miniBatch[7+featureSize+numLabels] -> label1

     * ...
     */
    std::vector<float> miniBatch, temp;
    miniBatch.push_back(E);
	miniBatch.push_back(sigma);
    miniBatch.push_back(C);
    miniBatch.push_back(cnn.get_learning_rate());
    miniBatch.push_back(batchSize);
    miniBatch.push_back(x_mini_batch[0].size());
    miniBatch.push_back(numLabels);

    if(DISTILLATION_MODE){

        float* teacher_layer;

        for (int i=0; i<x_mini_batch.size(); i++) {
                // features
                temp = x_mini_batch[i];
                miniBatch.insert(miniBatch.end(), temp.begin(), temp.end());

                //Forward run on teacher for the specific sample
                teacher_layer = teacher.forward(temp.data(), TEMPERATURE, -1, 1);
                
                //Probabilities of teacher model for each class
                for (int j = 0; j < numLabels; j++) {
                    miniBatch.push_back(teacher_layer[j]);
                }

		if(i==x_mini_batch.size()-1){
			for(int j=0 ; j< numLabels; j++){
                		std::cout<<"teacher pred:"<< teacher_layer[j] << std::endl;
		        }

		}

                //label
                miniBatch.push_back(y_mini_batch[i]);
        }
	miniBatch.push_back(1234567);
    }
    else {
        for (int i=0; i<x_mini_batch.size(); i++) {
        		// features
        		temp = x_mini_batch[i];
        		miniBatch.insert(miniBatch.end(), temp.begin(), temp.end());

        		//label
        		miniBatch.push_back(y_mini_batch[i]);
        }
    }

    return miniBatch;
}

std::vector<float> nonIIDSample(int batchSize, int clientID) {
	printf("Getting sample for client: %d\n", clientID); fflush(stdout);

// 	if (clientID == 2) 
// 		batchSize = 1; // weak worker
		
	std::vector<std::vector<float>> x_mini_batch;
    	std::vector<int> y_mini_batch;

	// Select a mini batch from the bucket of the client
    for (int j = 0; j < batchSize; j++) {
	int idx = buckets[clientID][bucketIdx[clientID]];
       	x_mini_batch.push_back(sorted_images[idx]);
       	y_mini_batch.push_back(sorted_labels[idx]);

       	bucketIdx[clientID] = (bucketIdx[clientID] + 1) % buckets[clientID].size();
   	}

   	std::vector<float> miniBatch, temp;
    miniBatch.push_back(E);
    miniBatch.push_back(sigma);
    miniBatch.push_back(C);
    miniBatch.push_back(cnn.get_learning_rate());
   	miniBatch.push_back(batchSize);
   	miniBatch.push_back(x_mini_batch[0].size());
   	miniBatch.push_back(numLabels);

	// TODO no support for DISTILLATION_MODE 1
   	for (int i=0; i<x_mini_batch.size(); i++) {
   		// features
   		temp = x_mini_batch[i];
   		miniBatch.insert(miniBatch.end(), temp.begin(), temp.end());

   		//label
   		miniBatch.push_back(y_mini_batch[i]);
   	}

   	return miniBatch;

}

extern "C"
JNIEXPORT jbyteArray JNICALL Java_apps_cppNN_CppNNOfflineSampler_getMiniBatch(JNIEnv * env, jobject, jint batch_size) {

	std::vector<float> miniBatch;
	batch_size *= E;
	if (iid)
		miniBatch = uniformSample((int) batch_size);
	else
		miniBatch = nonIIDSample((int) batch_size, currClientID);

	currClientID = (currClientID + 1) % numClients;

    std::string encoded = Base64::encode(miniBatch);

    const char* response = encoded.c_str();

//    env->DeleteLocalRef(array);
    jbyteArray array = env->NewByteArray(strlen(response));
    env->SetByteArrayRegion(array, 0, strlen(response), (jbyte*)response);

    return array;

}

extern "C"
JNIEXPORT jbyteArray JNICALL Java_apps_cppNN_CppNNUpdater_getFlatGradient(JNIEnv * env, jobject, jbyteArray input) {
    jbyte* buffer = env->GetByteArrayElements(input, NULL);
    jsize size = env->GetArrayLength(input);
    std::string encoded = (char *) buffer;

    std::vector<float> ret =  Base64::decodeFloat(encoded.substr(0, size));

    encoded = Base64::encode(cnn.flatGrad(ret));

    const char* response = encoded.c_str();

//    env->DeleteLocalRef(array);
    jbyteArray array = env->NewByteArray(strlen(response));
    env->SetByteArrayRegion(array, 0, strlen(response), (jbyte*)response);

    env->ReleaseByteArrayElements(input, buffer, 0);

    return array;
}

extern "C"
JNIEXPORT jbyteArray JNICALL Java_apps_cppNN_CppNNUpdater_mergeFlatGradient(JNIEnv * env, jobject, jbyteArray g, jbyteArray flatG) {
    jbyte* buffer = env->GetByteArrayElements(g, NULL);
    jsize size = env->GetArrayLength(g);
    std::string encoded = (char *) buffer;

    std::vector<float> grad =  Base64::decodeFloat(encoded.substr(0, size));

    jbyte* buffer2 = env->GetByteArrayElements(flatG, NULL);
    size = env->GetArrayLength(flatG);
    encoded = (char *) buffer2;

    std::vector<float> flatGrad =  Base64::decodeFloat(encoded.substr(0, size));

    cnn.mergeFlatGrad(grad, flatGrad);

    encoded = Base64::encode(grad);

    const char* response = encoded.c_str();

//    env->DeleteLocalRef(array);
    jbyteArray array = env->NewByteArray(strlen(response));
    env->SetByteArrayRegion(array, 0, strlen(response), (jbyte*)response);

    env->ReleaseByteArrayElements(g, buffer, 0);
    env->ReleaseByteArrayElements(flatG, buffer2, 0);

    return array;
}


extern "C"
JNIEXPORT jbyteArray JNICALL Java_utils_ByteVec_scalarMulNative(JNIEnv * env, jobject, jbyteArray input, jdouble a) {

    jbyte* buffer = env->GetByteArrayElements(input, NULL);
    jsize size = env->GetArrayLength(input);
    std::string encoded = (char *) buffer;

    std::vector<float> ret =  Base64::decodeFloat(encoded.substr(0, size));

    std::vector<float> res;
    for (int i=0; i<ret.size(); i++)
    		res.push_back(ret[i] * a);

    encoded = Base64::encode(res);

    const char* response = encoded.c_str();

//    env->DeleteLocalRef(array);
    jbyteArray array = env->NewByteArray(strlen(response));
    env->SetByteArrayRegion(array, 0, strlen(response), (jbyte*)response);

    env->ReleaseByteArrayElements(input, buffer, 0);

    return array;
}

extern "C"
JNIEXPORT double JNICALL Java_utils_ByteVec_getNorm(JNIEnv * env, jobject, jbyteArray input) {

    jbyte* buffer = env->GetByteArrayElements(input, NULL);
    jsize size = env->GetArrayLength(input);
    std::string encoded = (char *) buffer;

    std::vector<float> ret =  Base64::decodeFloat(encoded.substr(0, size));

    double s = 0;
    for (int i=0; i<ret.size(); i++)
    		s += ret[i] * ret[i];

    env->ReleaseByteArrayElements(input, buffer, 0);

    return sqrt(s);
}

extern "C"
JNIEXPORT jbyteArray JNICALL Java_utils_ByteVec_addNative(JNIEnv * env, jobject, jbyteArray a, jbyteArray b) {

    jbyte* buffer = env->GetByteArrayElements(a, NULL);
    jsize size = env->GetArrayLength(a);
    std::string encoded = (char *) buffer;

    std::vector<float> retA =  Base64::decodeFloat(encoded.substr(0, size));

    env->ReleaseByteArrayElements(a, buffer, 0);

    buffer = env->GetByteArrayElements(b, NULL);
    size = env->GetArrayLength(b);
    encoded = (char *) buffer;

    std::vector<float> retB = Base64::decodeFloat(encoded.substr(0, size));

    env->ReleaseByteArrayElements(b, buffer, 0);


    std::cout << "Adding size: " << retA.size() << std::endl;
    for (int i=152; i<159; i++)
    		printf("%.4f ", retA[i]);

    printf("\n");
    std::cout << "from size: " << retA.size() << std::endl;
    for (int i=152; i<159; i++)
    		printf("%.4f ", retB[i]);
    printf("\n");
    fflush(stdout);

    std::vector<float> res;
    for (int i=0; i<retA.size(); i++)
    		 res.push_back(retA[i] + retB[i]);

    for (int i=152; i<159; i++)
    		printf("%.4f ", res[i]);
    printf("\n");
    fflush(stdout);

    encoded = Base64::encode(res);

    const char* response = encoded.c_str();

//    env->DeleteLocalRef(array);
    jbyteArray array = env->NewByteArray(strlen(response));
    env->SetByteArrayRegion(array, 0, strlen(response), (jbyte*)response);

    return array;
}

extern "C"
JNIEXPORT jbyteArray JNICALL Java_utils_ByteVec_subtractNative(JNIEnv * env, jobject, jbyteArray a, jbyteArray b) {

    jbyte* buffer = env->GetByteArrayElements(a, NULL);
    jsize size = env->GetArrayLength(a);
    std::string encoded = (char *) buffer;

    std::vector<float> retA =  Base64::decodeFloat(encoded.substr(0, size));

    env->ReleaseByteArrayElements(a, buffer, 0);

    buffer = env->GetByteArrayElements(b, NULL);
    size = env->GetArrayLength(b);
    encoded = (char *) buffer;

    std::vector<float> retB = Base64::decodeFloat(encoded.substr(0, size));

    env->ReleaseByteArrayElements(b, buffer, 0);


    std::cout << "Subtracting size: " << retA.size() << std::endl;
    for (int i=152; i<159; i++)
    		printf("%.4f ", retA[i]);

    printf("\n");
    std::cout << "from size: " << retA.size() << std::endl;
    for (int i=152; i<159; i++)
    		printf("%.4f ", retB[i]);
    printf("\n");
    fflush(stdout);

    std::vector<float> res;
    for (int i=0; i<retA.size(); i++)
    		 res.push_back(retA[i] - retB[i]);

    encoded = Base64::encode(res);

    const char* response = encoded.c_str();

//    env->DeleteLocalRef(array);
    jbyteArray array = env->NewByteArray(strlen(response));
    env->SetByteArrayRegion(array, 0, strlen(response), (jbyte*)response);

    return array;
}
