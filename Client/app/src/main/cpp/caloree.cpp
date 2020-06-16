/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include <android/log.h>
#include <pthread.h>
#include "caloree.h"

double per_example_base_latency;
double average_speedup;
double last_speedup;
int rounds;
double pole;
double per_round_base_latency;

//h10
/*cmap_t pht = {
        {0.272552, std::make_tuple(1,0)},
        {1.002950, std::make_tuple(1,3)},
        {1.008304, std::make_tuple(4,0)},
        {1.008503, std::make_tuple(0,4)},
        {1.011503, std::make_tuple(0,4)},
};*/

//S8
cmap_t pht = {
        {0.167356, std::make_tuple(3,0)},
        {0.214160, std::make_tuple(4,0)},
        {0.216280, std::make_tuple(4,0)},
        {0.892886, std::make_tuple(0,4)},
        {1.029218, std::make_tuple(3,4)},
};

//S7 warm
/*cmap_t pht = {
        {0.068394, std::make_tuple(1,0)},
        {0.533892, std::make_tuple(0,4)},
        {0.881761, std::make_tuple(2,4)},
        {0.934667, std::make_tuple(3,4)},
        {1, std::make_tuple(4,4)},
};*/

//Xperia
/*cmap_t pht = {
        {0.266092, std::make_tuple(1,0)},
        {0.986748, std::make_tuple(4,0)},
        {0.993406, std::make_tuple(4,0)},
        {1, std::make_tuple(4,0)},
};*/

//S4 mini
/*cmap_t pht = {
        {0.502686, std::make_tuple(1,0)},
        {0.981172, std::make_tuple(2,0)},
        {0.988575, std::make_tuple(2,0)},
        {0.996090, std::make_tuple(2,0)},
        {1, std::make_tuple(2,0)},
};*/

/*
 * n_examples mini-batch size
 * perex_base_latency computation latency when all resources are employed
 * perex_target_latency caloree deadline
 * r number of caloree iterations
 * p caloree pole
 */
void init(int n_examples, double perex_base_latency,
          double perex_target_latency,
          int r, double p){
    per_example_base_latency = perex_base_latency;
    rounds = r;
    pole = p;
    //pht = loadPHT(is);
    //__android_log_print(ANDROID_LOG_WARN, "INFO", "perex_target_latency: %lf\n", perex_target_latency);
    average_speedup = per_example_base_latency  / perex_target_latency;
    last_speedup = average_speedup;
    per_round_base_latency = (per_example_base_latency * n_examples) / rounds;
}

conf_tuple_t get_config(double next_speedup, int n_round_examples){
    double ceiling_speedup, floor_speedup;
    int floor_examples, ceiling_examples;

    auto floor = pht.lower_bound(next_speedup);
    auto ceiling = pht.upper_bound(next_speedup);
    if (floor == pht.begin()){
        __android_log_print(ANDROID_LOG_WARN, "INFO", "key smaller than min\n");

        ceiling_speedup = floor_speedup = ceiling->first;

        floor_examples = n_round_examples / 2;
        ceiling_examples = n_round_examples - floor_examples;
    } else if (ceiling == pht.end()) {
        __android_log_print(ANDROID_LOG_WARN, "INFO", "key larger than max\n");

        ceiling_speedup = floor_speedup = (--ceiling)->first;

        floor_examples = n_round_examples / 2;
        ceiling_examples = n_round_examples - floor_examples;
    } else {
        ceiling_speedup = ceiling->first;
        floor_speedup = (--ceiling)->first;

        floor_examples = (int) (n_round_examples * floor_speedup * ceiling_speedup -
                n_round_examples * floor_speedup * next_speedup) /
                             (ceiling_speedup * next_speedup - floor_speedup * next_speedup);
        ceiling_examples = n_round_examples - floor_examples;
    }

    int_tuple_t floor_conf = pht[floor_speedup];
    int_tuple_t ceiling_conf = pht[ceiling_speedup];

    config *c1 = new config(floor_conf, floor_examples);
    config *c2 = new config(ceiling_conf, ceiling_examples);

    __android_log_print(ANDROID_LOG_WARN, "INFO", "caloree output: %d %d %d %d %d %d\n",
                        std::get<0>(floor_conf), std::get<1>(floor_conf), floor_examples,
                        std::get<0>(ceiling_conf), std::get<1>(ceiling_conf), ceiling_examples);

    return std::make_tuple(c1, c2);
}

double compute_next_xup(double last_latency){
    //__android_log_print(ANDROID_LOG_WARN, "INFO", "Last latency: %lf\n", last_latency);
    double real_speedup = per_round_base_latency / last_latency;

    //__android_log_print(ANDROID_LOG_WARN, "INFO", "Real speedup: %lf\n", real_speedup);
    //__android_log_print(ANDROID_LOG_WARN, "INFO", "Avg speedup: %lf\n", average_speedup);

    double next_speedup = last_speedup + (1 - pole) * (average_speedup - real_speedup);
    last_speedup = next_speedup;
    //__android_log_print(ANDROID_LOG_WARN, "INFO", "Next speedup: %lf\n", next_speedup);

    return next_speedup;
}

long long install_conf(std::atomic<bool>* bitset, int* current_example_wait, int littleThreads1,
                  int bigThreads1, int nexamples1, int littleThreads2, int bigThreads2, int nexamples2,
                  pthread_mutex_t* block_lock, pthread_cond_t* block_nonzero, bool* block,
                  pthread_mutex_t* master_lock, pthread_cond_t* master_nonzero, bool* master_block){
    long long start = getTimeNsec();

    //__android_log_print(ANDROID_LOG_WARN, "INFO", "1");
    for (int i = 0; i < 4; i++) {
        if (i < littleThreads1) {
            //__android_log_print(ANDROID_LOG_WARN, "INFO", "iter %d block 11\n", i);
            pthread_mutex_lock(&block_lock[i]);
            if (block[i])
                pthread_cond_signal(&block_nonzero[i]);
            block[i] = false;
            pthread_mutex_unlock(&block_lock[i]);
        } else {
            //__android_log_print(ANDROID_LOG_WARN, "INFO", "iter %d block 12\n", i);
            pthread_mutex_lock(&block_lock[i]);
            block[i] = true;
            pthread_mutex_unlock(&block_lock[i]);
        }
    }

    //__android_log_print(ANDROID_LOG_WARN, "INFO", "2");
    for (int i = 4; i < 8; i++) {
        if (i < 4 + bigThreads1) {
            //__android_log_print(ANDROID_LOG_WARN, "INFO", "iter %d block 21\n", i);
            pthread_mutex_lock(&block_lock[i]);
            if (block[i])
                pthread_cond_signal(&block_nonzero[i]);
            block[i] = false;
            pthread_mutex_unlock(&block_lock[i]);
        } else {
            //__android_log_print(ANDROID_LOG_WARN, "INFO", "iter %d block 22\n", i);
            pthread_mutex_lock(&block_lock[i]);
            block[i] = true;
            pthread_mutex_unlock(&block_lock[i]);
        }
    }

    //__android_log_print(ANDROID_LOG_WARN, "INFO", "3");
    *current_example_wait += nexamples1;
    //__android_log_print(ANDROID_LOG_WARN, "INFO", "current_example_wait %d\n", *current_example_wait);

    if (!bitset[(*current_example_wait) - 25].load()) {
        pthread_mutex_lock(master_lock);
        *master_block = true;
        while (*master_block)
            pthread_cond_wait(master_nonzero, master_lock);
        pthread_mutex_unlock(master_lock);
    }
    //while (!bitset[(*current_example_wait) - 1].load()){}

    //__android_log_print(ANDROID_LOG_WARN, "INFO", "4");
    for (int i = 0; i < 4; i++) {
        if (i < littleThreads2) {
            pthread_mutex_lock(&block_lock[i]);
            if (block[i])
                pthread_cond_signal(&block_nonzero[i]);
            block[i] = false;
            pthread_mutex_unlock(&block_lock[i]);
        } else {
            pthread_mutex_lock(&block_lock[i]);
            block[i] = true;
            pthread_mutex_unlock(&block_lock[i]);
        }
    }
    //__android_log_print(ANDROID_LOG_WARN, "INFO", "5");
    for (int i = 4; i < 8; i++) {
        if (i < 4 + bigThreads2) {
            pthread_mutex_lock(&block_lock[i]);
            if (block[i] == 1)
                pthread_cond_signal(&block_nonzero[i]);
            block[i] = 0;
            pthread_mutex_unlock(&block_lock[i]);
        } else {
            pthread_mutex_lock(&block_lock[i]);
            block[i] = 1;
            pthread_mutex_unlock(&block_lock[i]);
        }
    }
    //__android_log_print(ANDROID_LOG_WARN, "INFO", "6");
    *current_example_wait += nexamples2;
    //__android_log_print(ANDROID_LOG_WARN, "INFO", "current_example_wait %d\n", *current_example_wait);

    if (!bitset[(*current_example_wait) - 25].load()) {
        pthread_mutex_lock(master_lock);
        *master_block = true;
        while (*master_block)
            pthread_cond_wait(master_nonzero, master_lock);
        pthread_mutex_unlock(master_lock);
    }
    //while (!bitset[(*current_example_wait) - 1].load()){}

    //__android_log_print(ANDROID_LOG_WARN, "INFO", "7");

    return (getTimeNsec() - start) / 1000000;
}

void unlock_all(pthread_mutex_t* block_lock, pthread_cond_t* block_nonzero, bool* block){

    for (int i = 0; i < 8; i++) {
        pthread_mutex_lock(&block_lock[i]);
        if (block[i])
            pthread_cond_signal(&block_nonzero[i]);
        block[i] = false;
        pthread_mutex_unlock(&block_lock[i]);
    }

}

double get_average_speedup() {
    return average_speedup;
}

int get_nrounds() {
    return rounds;
}

