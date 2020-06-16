/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


// == mojo ====================================================================
//
//    Copyright (c) gnawice@gnawice.com. All rights reserved.
//	  See LICENSE in root folder
//
//    Permission is hereby granted, free of charge, to any person obtaining a
//    copy of this software and associated documentation files(the "Software"),
//    to deal in the Software without restriction, including without 
//    limitation the rights to use, copy, modify, merge, publish, distribute,
//    sublicense, and/or sell copies of the Software, and to permit persons to
//    whom the Software is furnished to do so, subject to the following 
//    conditions :
//
//    The above copyright notice and this permission notice shall be included
//    in all copies or substantial portions of the Software.
//
//    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//    OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF 
//    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//    IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
//    CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
//    OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR
//    THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// ============================================================================
//    cost.h:  cost/loss function for training
// ==================================================================== mojo ==


#pragma once

#include <math.h>
#include <algorithm>
#include <string>

namespace mojo {

namespace mse 
{
inline float  cost(float out, float target, float temp , float temp2) {
	return 0.5f * (out - target) * (out - target);
};
inline float  d_cost(float out, float target, float temp, float temp2) {
	return (out - target);
};
const char name[] = "mse";
}
/*
namespace triplet_loss 
{
	inline float  E(float out1, float out2, float out3) {return 0.5f*(out-target)*(out-target);};
	inline float  dE(float out, float target) {return (out-target);};
	const char name[]="triplet_loss";
}
*/
namespace cross_entropy 
{
inline float  cost(float out, float target, float temp, float temp2) {
	return (-target * std::log(out) - (1.f - target) * std::log(1.f - out));
};
inline float  d_cost(float out, float target, float temp, float temp2) {
	return ((out - target) / (out * (1.f - out)));
};
const char name[] = "cross_entropy";
}

namespace distillation {
inline float  cost(float out, float target, float t_out , float temperature_dist) {

	//Distillation loss is a cost function that utilizes the softmax with higher temperature
	//for both the teacher and the student model output and then implements a Kullback-Leider Divergence
	//for these two distributions
	//lastly adds a factor of cross-entropy for the output of students

	float wtl = 0.7;
	float loss = 0.f;

	loss = wtl * pow(temperature_dist, 2) *(-t_out * std::log(out) - (1.f - t_out) * std::log(1.f - out));
	loss += (1 -  wtl) * (-target * std::log(out) - (1.f - target) * std::log(1.f - out));
	//loss = (t_out>0) ? wtl * pow(temperature_dist, 2) * t_out * std::log(std::log(t_out) / out) : 0 ;
	//loss += (1 -  wtl) * (-target * std::log(out) - (1.f - target) * std::log(1.f - out));
	return loss;
};
inline float  d_cost(float out, float target, float t_out , float temperature_dist) {
	float wtl = 0.7;
	float deriv;
	// deriv = wtl * pow(temperature_dist, 2)*(out*(std::log(out)-std::log(t_out)) +(t_out-out)/t_out);
	// deriv += (1.f - wtl) * (out - target);

	//deriv = wtl*pow(temperature_dist,2)*((1.f-out)*(log(out) - log(t_out) + out*(1.f-out)) - std::log(out)*(1-t_out));
	//std::cout<<"1st: "<<t_out<<"2nd: "<<out<<"3rd: "<<target<<std::endl;
	//deriv = wtl * pow(temperature_dist, 2)*((log(out/t_out)+1)*(out*(1-out))-out*(1-t_out));
	//deriv += (1.f - wtl) * (out - target);


	//deriv = wtl * pow(temperature_dist, 2) * ((out - target) * (std::log(out / t_out) + 1) - ((t_out - target) * out) / t_out);
	//deriv += (1.f - wtl) * (out - target);
	//deriv += (1.f - wtl) * ((out - target) / (out * (1.f - out)));

	//deriv = (t_out > 0 ) ? -1.f * t_out / out: 0;

	deriv =  wtl * pow(temperature_dist, 2)* ((out - target) / (out * (1.f - out))) + (1 -  wtl)* ((out - target) / (out * (1.f - out)));
	return deriv;
};
const char name[] = "distillation";
}

typedef struct 
{
public:
	float (*cost)(float, float, float, float);
	float (*d_cost)(float, float, float, float);
	const char *name;
} cost_function;

cost_function* new_cost_function(std::string loss)
{
	cost_function *p = new cost_function;
	if (loss.compare(cross_entropy::name) == 0) {
		p->cost = &cross_entropy::cost;
		p->d_cost = &cross_entropy::d_cost;
		p->name = cross_entropy::name; return p;
	}
	if (loss.compare(mse::name) == 0) {
		p->cost = &mse::cost;
		p->d_cost = &mse::d_cost;
		p->name = mse::name; return p;
	}
	if (loss.compare(distillation::name) == 0) {
		p->cost = &distillation::cost;
		p->d_cost = &distillation::d_cost;
		p->name = distillation::name;
		return p;
	}
	//if(loss.compare(triplet_loss::name)==0) { p->E = &triplet_loss::E; p->dE = &triplet_loss::dE; p->name=triplet_loss::name; return p;}
	delete p;
	return NULL;
}

cost_function* new_cost_function(const char *type)
{
	std::string loss(type);
	return new_cost_function(loss);
}

}
