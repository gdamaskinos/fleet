/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


/* Compile/Run with: 
 * ```g++ -std=c++11 -I/$JAVA_HOME/include -I/$JAVA_HOME/include/linux -I/$JAVA_HOME/include/darwin playground.cpp && ./a.out```
 * */

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include "cpp_utils/Base64.h"
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <algorithm>
#include <time.h>
#include <math.h>
//#include <tchar.h>

//#define MOJO_CV3

#include "cppNN/mojo.h"
#include "cppNN/util.h"
#include "cppNN/cost.h"
#include "cppNN/mnist_parser.h"
#include "cppNN/cifar_parser.h"

int numLabels;

std::vector<std::vector<float>> test_images;
std::vector<int> test_labels;
std::vector<std::vector<float>> train_images;
std::vector<int> train_labels;

bool iid = false;
int numClients = 10;
std::vector<std::vector<int>> buckets; // each bucket contains the shuffled dataset of each client
std::vector<int> bucketIdx; // current index for sampling the bucket (non-overlapping samples)
std::vector<std::vector<float>> sorted_images;
std::vector<int> sorted_labels;

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

    	for (int i=0; i<5; i++) {
    	        printf("%d\n", y_mini_batch[i]);
	}
    	    
    	std::vector<float> miniBatch, temp;
    	miniBatch.push_back(batchSize);
    	miniBatch.push_back(x_mini_batch[0].size());
    	miniBatch.push_back(numLabels);

    	    for (int i=0; i<x_mini_batch.size(); i++) {
    	    		// features
    	    		temp = x_mini_batch[i];
    	    		miniBatch.insert(miniBatch.end(), temp.begin(), temp.end());

    	    		//label
    	    		miniBatch.push_back(y_mini_batch[i]);
    	    }

    	return miniBatch;
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

std::vector<float> nonIIDSample(int batchSize, int clientID) {
	printf("Getting sample for client: %d\n", clientID); fflush(stdout);

	std::vector<std::vector<float>> x_mini_batch;
    	std::vector<int> y_mini_batch;

	// Select a mini batch from the bucket of the client
    	for (int j = 0; j < batchSize; j++) {
		int idx = buckets[clientID][bucketIdx[clientID]];
        	x_mini_batch.push_back(sorted_images[idx]);
        	y_mini_batch.push_back(sorted_labels[idx]);

		bucketIdx[clientID] = (bucketIdx[clientID] + 1) % buckets[clientID].size();
    	}

		for (int i=0; i<batchSize; i++) printf("%d ", y_mini_batch[i]);
		printf("\n");

    	std::vector<float> miniBatch, temp;
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


int main() {
	srand(1);
	
  	int prevClientID;
	int batch_size = 100;

	std::string data_path("path/to/mnist/original/");
	if (!mnist::parse_train_data(data_path, train_images, train_labels)) { std::cerr << "error: could not parse data.\n"; exit(0);}
	numLabels = 10;
	
	if (!iid) {

		// sort data by label
		std::vector<size_t> indices = sort_indexes(train_labels);
		std::vector<int> range;
		for (int i=0; i< (int) train_images.size(); i++) {
			sorted_images.push_back(train_images[indices[i]]);
			sorted_labels.push_back(train_labels[indices[i]]);
			range.push_back(i);
		}

		// create shards
		std::vector<int> shards;
		for (int i=0; i< 2 * numClients; i++)
			shards.push_back(i);
		std::random_shuffle(shards.begin(), shards.end(), [](int i) -> int {return std::rand()%i;});


		// create buckets
		int bucketSize = (int) train_images.size() / numClients;
		int shardSize = (int) bucketSize / 2;
		printf("Bucket size:%d\nShard size:%d\n", bucketSize, shardSize);
		for (int i=0; i<shards.size(); i=i+2) {
			std::vector<int> bucket;
			// insert shard1
			bucket.insert(bucket.end(), range.begin()+shards[i]*shardSize, 
					range.begin()+ (shards[i]+1)*shardSize);
			// insert shard2
			bucket.insert(bucket.end(), range.begin()+shards[i+1]*shardSize, 
					range.begin()+ (shards[i+1]+1)*shardSize);

			std::random_shuffle(bucket.begin(), bucket.end(), [](int i) -> int {return std::rand()%i;});
			buckets.push_back(bucket);

			bucketIdx.push_back(0); // each client starts sampling from first element of bucket
		}

	}

	int train_samples = (int)train_images.size();
    	std::cout << "Train examples: " << train_samples << std::endl;

    	prevClientID = -1;

	for (int i=0; i<120; i++) {
		prevClientID = (prevClientID + 1) % numClients;
		std::vector<float> miniBatch = nonIIDSample((int) batch_size, prevClientID);
	}
	

	exit(0);

}
