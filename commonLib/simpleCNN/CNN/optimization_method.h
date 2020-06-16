/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

#include "gradient_t.h"

double LEARNING_RATE = 0.005;
double MOMENTUM = 0.8;
double WEIGHT_DECAY = 0.001;

static float update_weight(float w, gradient_t &grad, float multp = 1) {
    float m = (grad.grad + grad.oldgrad * MOMENTUM);
    w -= LEARNING_RATE * m * multp +
    	LEARNING_RATE * WEIGHT_DECAY * w;
    return w;
}

static void update_gradient(gradient_t &grad) {
    grad.oldgrad = (grad.grad + grad.oldgrad * MOMENTUM);
}
