/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


/*
 * =====================================================================================
 *
 *       Filename:  caloree.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  13/12/2019 16:56:21
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

#include <tuple>
#include <map>

typedef std::tuple<int, int> int_tuple_t;
typedef std::map<double, int_tuple_t> cmap_t;

class config {
public:
    int_tuple_t threadConf;
    int n_examples;

    config(int_tuple_t tConf, int n_ex){
        threadConf = tConf;
        n_examples = n_ex;
    }
    int_tuple_t get_thread_conf() {
        return threadConf;
    }
    int get_nexamples() {
        return n_examples;
    }
};
typedef std::tuple<config*, config*> conf_tuple_t;

long long getTimeNsec() {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (int64_t) now.tv_sec*1000000000LL + now.tv_nsec;
}

void init(int n_examples, double perex_base_latency,
          double perex_target_latency,
          int r, double p);

conf_tuple_t get_config(double next_speedup, int n_round_examples);

double compute_next_xup(double last_latency);

long long install_conf(std::atomic<bool>* bitset, int* current_example_wait, int littleThreads1,
                  int bigThreads1, int nexamples1, int littleThreads2, int bigThreads2, int nexamples2,
                  pthread_mutex_t* block_lock, pthread_cond_t* block_nonzero, bool* block,
                  pthread_mutex_t* master_lock, pthread_cond_t* master_nonzero, bool* master_block);

void unlock_all(pthread_mutex_t* block_lock, pthread_cond_t* block_nonzero, bool* block);

double get_average_speedup();

int get_nrounds();
