/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

#include "types.h"
#include "tensor_t.h"

#pragma pack(push, 1)
struct layer_t {
    layer_type type;
    tensor_t<float> grads_in;
    tensor_t<float> grads_acc;
    tensor_t<float> in;
    tensor_t<float> out;

    void accumulate_grads() {
        this->grads_acc.add(grads_in);
    }

    void grads_acc_to_in(float quotient) {
        for (int i = 0; i < this->grads_in.size.x * this->grads_in.size.y * this->grads_in.size.z; ++i) {
            this->grads_in.data[i] = grads_acc.data[i] / quotient;
            grads_acc.data[i] = 0;
        }
    }
};
#pragma pack(pop)
