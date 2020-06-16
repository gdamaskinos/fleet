/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

#include "layer_t.h"

#pragma pack(push, 1)

struct conv_layer_t {
    layer_type type = layer_type::conv;
    tensor_t<float> grads_in;
    tensor_t<float> grads_acc;
    tensor_t<float> in;
    tensor_t<float> out;
    std::vector<tensor_t<float>> filters;
    std::vector<tensor_t<gradient_t>> filter_grads;
    std::vector<tensor_t<gradient_t>> acc_grads;
    uint16_t stride;
    uint16_t extend_filter;

    conv_layer_t(uint16_t stride, uint16_t extend_filter, uint16_t number_filters, tdsize in_size)
            :
            grads_in(in_size.x, in_size.y, in_size.z),
            grads_acc(in_size.x, in_size.y, in_size.z),
            in(in_size.x, in_size.y, in_size.z),
            out(
                    (in_size.x - extend_filter) / stride + 1,
                    (in_size.y - extend_filter) / stride + 1,
                    number_filters
            ) {
        this->stride = stride;
        this->extend_filter = extend_filter;
        assert((float(in_size.x - extend_filter) / stride + 1)
               ==
               ((in_size.x - extend_filter) / stride + 1));

        assert((float(in_size.y - extend_filter) / stride + 1)
               ==
               ((in_size.y - extend_filter) / stride + 1));

        for (int a = 0; a < number_filters; a++) {
            tensor_t<float> t(extend_filter, extend_filter, in_size.z);

            int maxval = extend_filter * extend_filter * in_size.z;

            for (int i = 0; i < extend_filter; i++)
                for (int j = 0; j < extend_filter; j++)
                    for (int z = 0; z < in_size.z; z++)
                        t(i, j, z) = 1.0f / maxval * rand() / float(RAND_MAX);
            filters.push_back(t);
        }
        for (int i = 0; i < number_filters; i++) {
            tensor_t<gradient_t> t(extend_filter, extend_filter, in_size.z);
            filter_grads.push_back(t);
	    acc_grads.push_back(t);
        }

    }

    point_t map_to_input(point_t out, int z) {
        out.x *= stride;
        out.y *= stride;
        out.z = z;
        return out;
    }

    struct range_t {
        int min_x, min_y, min_z;
        int max_x, max_y, max_z;
    };

    int normalize_range(float f, int max, bool lim_min) {
        if (f <= 0)
            return 0;
        max -= 1;
        if (f >= max)
            return max;

        if (lim_min) // left side of inequality
            return ceil(f);
        else
            return floor(f);
    }

    void init_acc_grads() {
        for (int k = 0; k < filter_grads.size(); k++) {
            for (int i = 0; i < extend_filter; i++)
                for (int j = 0; j < extend_filter; j++)
                    for (int z = 0; z < in.size.z; z++){
                        acc_grads[k].get(i, j, z).grad = 0;
                        acc_grads[k].get(i, j, z).oldgrad = 0;
		    }
        }
    }


    range_t map_to_output(int x, int y) {
        float a = x;
        float b = y;
        return
                {
                        normalize_range((a - extend_filter + 1) / stride, out.size.x, true),
                        normalize_range((b - extend_filter + 1) / stride, out.size.y, true),
                        0,
                        normalize_range(a / stride, out.size.x, false),
                        normalize_range(b / stride, out.size.y, false),
                        (int) filters.size() - 1,
                };
    }

    void activate(tensor_t<float> &in) {
        this->in = in;
        activate();
    }

    void activate() {
        for (int filter = 0; filter < filters.size(); filter++) {
            tensor_t<float> &filter_data = filters[filter];
            for (int x = 0; x < out.size.x; x++) {
                for (int y = 0; y < out.size.y; y++) {
                    point_t mapped = map_to_input({(uint16_t) x, (uint16_t) y, 0}, 0);
                    float sum = 0;
                    for (int i = 0; i < extend_filter; i++)
                        for (int j = 0; j < extend_filter; j++)
                            for (int z = 0; z < in.size.z; z++) {
                                float f = filter_data(i, j, z);
                                float v = in(mapped.x + i, mapped.y + j, z);
                                sum += f * v;
                            }
                    out(x, y, filter) = sum;
                }
            }
        }
    }

    void fix_weights() {
        for (int a = 0; a < filters.size(); a++)
            for (int i = 0; i < extend_filter; i++)
                for (int j = 0; j < extend_filter; j++)
                    for (int z = 0; z < in.size.z; z++) {
                        float &w = filters[a].get(i, j, z);
                        gradient_t &grad = acc_grads[a].get(i, j, z);
                        w = update_weight(w, grad);
			update_gradient(grad);
                    }
    }

    void calc_grads(tensor_t<float> &grad_next_layer) {

        for (int k = 0; k < filter_grads.size(); k++) {
            for (int i = 0; i < extend_filter; i++)
                for (int j = 0; j < extend_filter; j++)
                    for (int z = 0; z < in.size.z; z++)
                        filter_grads[k].get(i, j, z).grad = 0;
        }

        for (int x = 0; x < in.size.x; x++) {
            for (int y = 0; y < in.size.y; y++) {
                range_t rn = map_to_output(x, y);
                for (int z = 0; z < in.size.z; z++) {
                    float sum_error = 0;
                    for (int i = rn.min_x; i <= rn.max_x; i++) {
                        int minx = i * stride;
                        for (int j = rn.min_y; j <= rn.max_y; j++) {
                            int miny = j * stride;
                            for (int k = rn.min_z; k <= rn.max_z; k++) {
                                int w_applied = filters[k].get(x - minx, y - miny, z);
                                sum_error += w_applied * grad_next_layer(i, j, k);
                                filter_grads[k].get(x - minx, y - miny, z).grad +=
                                        in(x, y, z) * grad_next_layer(i, j, k);
                            }
                        }
                    }
                    grads_in(x, y, z) = sum_error;
                }
            }
        }

        for (int k = 0; k < filter_grads.size(); k++) {
            for (int i = 0; i < extend_filter; i++)
                for (int j = 0; j < extend_filter; j++)
                    for (int z = 0; z < in.size.z; z++){
                        acc_grads[k].get(i, j, z).grad += filter_grads[k].get(i, j, z).grad;
		    }
        }
    }

    std::vector<float> grads_to_floats() {
        std::vector<float> floats;

        for(auto filter : acc_grads) {
            for (int x = 0; x < filter.size.x; ++x) {
                for (int y = 0; y < filter.size.y; ++y) {
                    for (int z = 0; z < filter.size.z; ++z) {
                        floats.push_back(filter.get(x, y, z).grad);
                    }
                }
            }
        }

        return floats;
    }


    void floats_to_grads(std::vector<float> &floats) {
        int i = 0;
        for(auto filter : acc_grads) {
            for (int x = 0; x < filter.size.x; ++x) {
                for (int y = 0; y < filter.size.y; ++y) {
                    for (int z = 0; z < filter.size.z; ++z) {
                        filter.get(x, y, z).grad = floats[i];
                        i++;
                    }
                }
            }
        }

        floats.erase(floats.begin(), floats.begin() + acc_grads.size());
    }
	
	/*
	 *  Return a flatten vector representation of this layer
	 *	return[0]-return[size_filter0-1]: flat vector of filter0
	 *	return[size_filter0]-return[size_filter0+size_filter1-1]: flat vector of filter1
	 *	...
	 */
	std::vector<float> flatParams() {
		std::vector<float> ret, temp;
		for (int i=0; i<filters.size(); i++) {
			temp = filters[i].flatData();
			// TODO implement with move for no-copy (will make the current
			// filters unusable)
			ret.insert(ret.end(), temp.begin(), temp.end());
		}
		return ret;
	}

	/*
	 * Initialize the parameters of this layer with a flatten vector
	 * returns the number of params read
	 */
	int setParams(std::vector<float> flatData) {
		int idx = 0;
		for (int i=0; i<filters.size(); i++) 
			for (int j=0; j<filters[i].getSize(); j++)
				filters[i].data[j] = flatData[idx++];
		return idx;
	}
};

#pragma pack(pop)
