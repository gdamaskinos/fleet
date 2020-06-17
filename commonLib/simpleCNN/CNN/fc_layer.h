/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

#include <math.h>
#include <float.h>
#include <string.h>
#include "layer_t.h"

#pragma pack(push, 1)

struct fc_layer_t {
    layer_type type = layer_type::fc;
    tensor_t<float> grads_in;
    tensor_t<float> grads_acc;
    tensor_t<float> in;
    tensor_t<float> out;
    std::vector<float> input;
    tensor_t<float> weights;
    std::vector<gradient_t> gradients;
    std::vector<gradient_t> acc_grads;

    fc_layer_t(tdsize in_size, int out_size)
            :
            in(in_size.x, in_size.y, in_size.z),
            out(out_size, 1, 1),
            grads_in(in_size.x, in_size.y, in_size.z),
            grads_acc(in_size.x, in_size.y, in_size.z),
            weights(in_size.x * in_size.y * in_size.z, out_size, 1) {
        input = std::vector<float>(out_size);
        gradients = std::vector<gradient_t>(out_size);
        acc_grads= std::vector<gradient_t>(out_size);


        int maxval = in_size.x * in_size.y * in_size.z;

        for (int i = 0; i < out_size; i++)
            for (int h = 0; h < in_size.x * in_size.y * in_size.z; h++)
                weights(h, i, 0) = 2.19722f / maxval * rand() / float(RAND_MAX);
        // 2.19722f = f^-1(0.9) => x where [1 / (1 + exp(-x) ) = 0.9]
    }

    float activator_function(float x) {
        //return tanhf( x );
        float sig = 1.0f / (1.0f + exp(-x));
        return sig;
    }

    float activator_derivative(float x) {
        //float t = tanhf( x );
        //return 1 - t * t;
        float sig = 1.0f / (1.0f + exp(-x));
        return sig * (1 - sig);
    }

    void activate(tensor_t<float> &in) {
        this->in = in;
        activate();
    }

    int map(point_t d) {
        return d.z * (in.size.x * in.size.y) +
               d.y * (in.size.x) +
               d.x;
    }

    void init_acc_grads() {
	for (int n = 0; n < out.size.x; n++) {
	    acc_grads[n].grad=0;
	    acc_grads[n].oldgrad=0;
	}
    }

    void activate() {
        for (int n = 0; n < out.size.x; n++) {
            float inputv = 0;

            for (int i = 0; i < in.size.x; i++)
                for (int j = 0; j < in.size.y; j++)
                    for (int z = 0; z < in.size.z; z++) {
                        int m = map({i, j, z});
                        inputv += in(i, j, z) * weights(m, n, 0);
                    }

            input[n] = inputv;

            out(n, 0, 0) = activator_function(inputv);
        }
    }

    void fix_weights() {
        for (int n = 0; n < out.size.x; n++) {
            gradient_t &grad = acc_grads[n];
            for (int i = 0; i < in.size.x; i++)
                for (int j = 0; j < in.size.y; j++)
                    for (int z = 0; z < in.size.z; z++) {
                        int m = map({i, j, z});
                        float &w = weights(m, n, 0);
                        w = update_weight(w, grad, in(i, j, z));
                    }
	    update_gradient(grad);
        }
    }

    void calc_grads(tensor_t<float> &grad_next_layer) {
        memset(grads_in.data, 0, grads_in.size.x * grads_in.size.y * grads_in.size.z * sizeof(float));
        for (int n = 0; n < out.size.x; n++) {
            gradient_t &grad = gradients[n];
            grad.grad = grad_next_layer(n, 0, 0) * activator_derivative(input[n]);

            for (int i = 0; i < in.size.x; i++)
                for (int j = 0; j < in.size.y; j++)
                    for (int z = 0; z < in.size.z; z++) {
                        int m = map({i, j, z});
                        grads_in(i, j, z) += grad.grad * weights(m, n, 0);
                    }
	    acc_grads[n].grad += grad.grad;
        }
    }

    std::vector<float> grads_to_floats() {
        std::vector<float> floats;
        for(auto grad : acc_grads) {
            floats.push_back(grad.grad);
        }

        return floats;
    }

    void floats_to_grads(std::vector<float> &floats) {
        int i = 0;
        for(auto grad : acc_grads) {
            grad.grad = floats[i];
            i++;
        }

        floats.erase(floats.begin(), floats.begin() + acc_grads.size());
    }

	/*
	 *  Return a flatten vector representation of this layer
	 *  i.e., the weights flatten vector
	 */
	std::vector<float> flatParams() {
		return weights.flatData();
	}

	/*
	 * Initialize the parameters of this layer with a flatten vector
	 * returns the number of params read
	 */
	int setParams(std::vector<float> flatData) {
		int i;
		for (i=0; i<weights.getSize(); i++)
				weights.data[i] = flatData[i];
		return i;
	}
};
#pragma pack(pop)
