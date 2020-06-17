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
//    network.h: The main artificial neural network graph for mojo
// ==================================================================== mojo ==

#pragma once

#include <string>
#include <iostream> // cout
#include <fstream>
#include <sstream>
#include <map>
#include <set>
#include <vector>
#include <atomic>

#include "layer.h"
#include "solver.h"
#include "activation.h"
#include "cost.h"
#define TEMPERATURE 2
#ifndef DISTILLATION_MODE
#define DISTILLATION_MODE 0
#endif

// hack for VS2010 to handle c++11 for(:)
#if (_MSC_VER  == 1600)
	#ifndef __for__
	#define __for__ for each
	#define __in__ in
	#endif
#else
	#ifndef __for__
	#define __for__ for
	#define __in__ :
	#endif
#endif



namespace mojo {

#if defined(MOJO_CV2) || defined(MOJO_CV3)
// forward declare these for data augmentation
cv::Mat matrix2cv(const mojo::matrix &m, bool uc8 = false);
mojo::matrix cv2matrix(cv::Mat &m);
mojo::matrix transform(const mojo::matrix in, const int x_center, const int y_center, int out_dim, float theta = 0, float scale = 1.f);
#endif


	// sleep needed for threading
#ifdef _WIN32
#include <windows.h>
	void mojo_sleep(unsigned milliseconds) { Sleep(milliseconds); }
#else
#include <unistd.h>
	void mojo_sleep(unsigned milliseconds) { usleep(milliseconds * 1000); }
#endif

#ifdef MOJO_PROFILE_LAYERS
#ifdef _WIN32
	//* used for profiling layers
	double PCFreq = 0.0;
	__int64 CounterStart = 0;

	void StartCounter()
	{
		LARGE_INTEGER li;
		if (!QueryPerformanceFrequency(&li)) return;
		PCFreq = double(li.QuadPart) / 1000.0;
		QueryPerformanceCounter(&li);
		CounterStart = li.QuadPart;
	}
	double GetCounter()
	{
		LARGE_INTEGER li;
		QueryPerformanceCounter(&li);
		return double(li.QuadPart - CounterStart) / PCFreq;
	}
#else
	void StartCounter(){}
	double GetCounter(){return 0;}
#endif
	
#endif
	//*/

	void replace_str(std::string& str, const std::string& from, const std::string& to) {
		if (from.empty())
			return;
		size_t start_pos = 0;
		while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
			str.replace(start_pos, from.length(), to);
			start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
		}
	}


// returns Energy (euclidian distance / 2) and max index
float match_labels(const float *out, const float *target, const int size, int *best_index = NULL)
{
	float E = 0;
	int max_j = 0;
	for (int j = 0; j<size; j++)
	{
		E += (out[j] - target[j])*(out[j] - target[j]);
		if (out[max_j]<out[j]) max_j = j;
	}
	if (best_index) *best_index = max_j;
	E *= 0.5;
	return E;
}
// returns index of highest value (argmax)
int arg_max(const float *out, const int size)
{
	int max_j = 0;
	for (int j = 0; j<size; j++) 
		if (out[max_j]<out[j])  
		{max_j = j; }//std::cout <<j<<",";}
	return max_j;
}

//----------------------------------------------------------------------
//  network  
//  - class that holds all the layers and connection information
//	- runs forward prediction

class network
{
	
public:	
	int _size;  // output size
	int _thread_count; // determines number of layer sets (copys of layers)
	int _internal_thread_count; // used for speeding up convolutions, etc..
	static const int MAIN_LAYER_SET = 0;

    std::atomic<int> batch_index = {0};

	// training related stuff
	int _batch_size;   // determines number of dW sets 
	float _skip_energy_level;
	bool _smart_train;
	std::vector <float> _running_E;
	double _running_sum_E;
	cost_function *_cost_function;
	solver *_solver;
	static const unsigned char BATCH_RESERVED = 1, BATCH_FREE = 0, BATCH_COMPLETE = 2;
	static const int BATCH_FILLED_COMPLETE = -2, BATCH_FILLED_IN_PROCESS = -1;
#ifdef MOJO_OMP
	omp_lock_t _lock_batch;
	void lock_batch() {omp_set_lock(&_lock_batch);}
	void unlock_batch() {omp_unset_lock(&_lock_batch);}
	void init_lock() {omp_init_lock(&_lock_batch);}
	void destroy_lock() {omp_destroy_lock(&_lock_batch);}
	int get_thread_num() {return omp_get_thread_num();}
#else
	void lock_batch() {}
	void unlock_batch() {}
	void init_lock(){}
	void destroy_lock() {}
	int get_thread_num() {return 0;}
#endif


	// training progress stuff
	int train_correct;
	int train_skipped;
	int stuck_counter;
	int train_updates;
	int train_samples;
	int epoch_count;
	int max_epochs;
	float best_estimated_accuracy;
	int best_accuracy_count;
	float old_estimated_accuracy;
	float estimated_accuracy;
// data augmentation stuff
	int use_augmentation; // 0=off, 1=mojo, 2=opencv
	int augment_x, augment_y;
	int augment_h_flip, augment_v_flip;
	mojo::pad_type augment_pad;
	float augment_theta;
	float augment_scale;



	// here we have multiple sets of the layers to allow threading and batch processing
	// a separate layer set is needed for each independent thread
	std::vector< std::vector<base_layer *>> layer_sets;
	
	std::map<std::string, int> layer_map;  // name-to-index of layer for layer management
	std::vector<std::pair<std::string, std::string>> layer_graph; // pairs of names of layers that are connected
	std::vector<matrix *> W; // these are the weights between/connecting layers 

	// these sets are needed because we need copies for each item in mini-batch
	std::vector< std::vector<matrix>> dW_sets; // only for training, will have _batch_size of these
	std::vector< std::vector<matrix>> dbias_sets; // only for training, will have _batch_size of these
	std::vector< unsigned char > batch_open; // only for training, will have _batch_size of these	
	

	network(const char* opt_name=NULL): _thread_count(1), _skip_energy_level(0.f), _batch_size(1) 
	{ 
		_internal_thread_count=1;
		_size=0;  
		_solver = new_solver(opt_name);
		_cost_function = NULL;
		//std::vector<base_layer *> layer_set;
		//layer_sets.push_back(layer_set);
		layer_sets.resize(1);
		dW_sets.resize(_batch_size);
		dbias_sets.resize(_batch_size);
		batch_open.resize(_batch_size);
		_running_sum_E = 0.;
		train_correct = 0;
		train_samples = 0;
		train_skipped = 0;
		epoch_count = 0; 
		max_epochs = 1000;
		train_updates = 0;
		estimated_accuracy = 0;
		old_estimated_accuracy = 0;
		stuck_counter = 0;
		best_estimated_accuracy=0;
		best_accuracy_count=0;
		use_augmentation=0;
		augment_x = 0; augment_y = 0; augment_h_flip = 0; augment_v_flip = 0; 
		augment_pad =mojo::edge; 
		augment_theta=0; augment_scale=0;

		init_lock();
#ifdef USE_AF
		af::setDevice(0);
        af::info();
#endif
	}
	
	~network() 
	{
		clear();
		if (_cost_function) delete _cost_function;
		if(_solver) delete _solver; 
		destroy_lock();	
	}

	// call clear if you want to load a different configuration/model
	void clear()
	{
		for(int i=0; i<(int)layer_sets.size(); i++)
		{
			__for__(auto l __in__ layer_sets[i]) delete l;
			layer_sets.clear();
		}
		layer_sets.clear();
		__for__(auto w __in__ W) if(w) delete w;  
		W.clear();
		layer_map.clear();
		layer_graph.clear();
	}

	// output size of final layer;
	int out_size() {return _size;}

	// get input size 
	bool get_input_size(int *w, int *h, int *c)
	{
		if(layer_sets[MAIN_LAYER_SET].size()<1) return false; 
		*w=layer_sets[MAIN_LAYER_SET][0]->node.cols;*h=layer_sets[MAIN_LAYER_SET][0]->node.rows;*c=layer_sets[MAIN_LAYER_SET][0]->node.chans;
		return true;
	}

	// sets up number of layer copies to run over multiple threads
	void build_layer_sets()
	{
		int layer_cnt = (int)layer_sets.size();
		if (layer_cnt<_thread_count) layer_sets.resize(_thread_count);
		// ToDo: add shrink back /  else if(layer_cnt>_thread_count)
		sync_layer_sets();
	}

	inline int get_thread_count() {return _thread_count;}
	// must call this with max thread count before constructing layers
	// value <1 will result in thread count = # cores (including hyperthreaded)
	void enable_external_threads(int threads = -1)
	{
//#ifdef MOJO_OMP
//		if (threads < 1) threads = omp_get_num_procs();
		_thread_count = threads;
//		if(_internal_thread_count<=_thread_count) omp_set_num_threads(_thread_count);

//#else
//		if (threads < 1) _thread_count = 1;
//		else _thread_count = threads;
//		if (threads > 1) bail("must define MOJO_OMP to used threading");
//#endif
		build_layer_sets();
	}

	void enable_internal_threads(int threads = -1)
	{
#ifdef MOJO_OMP
		if (threads < 1) {threads = omp_get_num_procs(); threads = threads-1;} // one less than core count
		if(threads<1) _internal_thread_count=1;
		else _internal_thread_count=threads;
#else
		_internal_thread_count=1;
#endif

	}

	// when using threads, need to get bias data synched between all layer sets, 
	// call this after bias update in main layer set to copy the bias to the other sets
	void sync_layer_sets()
	{
		for(int i=1; i<(int)layer_sets.size();i++)
			for(int j=0; j<(int)layer_sets[MAIN_LAYER_SET].size(); j++)
				for(int k=0; k<layer_sets[MAIN_LAYER_SET][j]->bias.size(); k++) 
					(layer_sets[i])[j]->bias.x[k]=(layer_sets[MAIN_LAYER_SET])[j]->bias.x[k];
	}

	// used to add some noise to weights
	void heat_weights()
	{
		__for__(auto w __in__ W)
		{
			if (!w) continue;
			matrix noise(w->cols, w->rows, w->chans);
			noise.fill_random_normal(1.f/ noise.size());
			//noise *= *w;
			*w += noise; 
		}
	}

	// used to add some noise to weights
	void remove_means()
	{
		__for__(auto w __in__ W)
			if(w) w->remove_mean();
	}

	// used to push a layer back in the ORDERED list of layers
	// if connect_all() is used, then the order of the push_back is used to connect the layers
	// when forward or backward propogation, this order is used for the serialized order of calculations 
	// Layer_name must be unique.
	bool push_back(const char *layer_name, const char *layer_config)
	{
		if(layer_map[layer_name]) return false; //already exists
		base_layer *l=new_layer(layer_name, layer_config);
		// set map to index

		// make sure there is a 'set' to add layers to
		if(layer_sets.size()<1)
		{
			std::vector<base_layer *> layer_set;
			layer_sets.push_back(layer_set);
		}
		// make sure layer_sets are created
		build_layer_sets();

		layer_map[layer_name] = (int)layer_sets[MAIN_LAYER_SET].size();
		layer_sets[MAIN_LAYER_SET].push_back(l);
		// upadate as potential last layer - so it sets the out size
		_size=l->fan_size();
		// add other copies needed for threading
		for(int i=1; i<(int)layer_sets.size();i++) layer_sets[i].push_back(new_layer(layer_name, layer_config));
		return true;
	}

	// connect 2 layers together and initialize weights
	// top and bottom concepts are reversed from literature
	// my 'top' is the input of a forward() pass and the 'bottom' is the output
	// perhaps 'top' traditionally comes from the brain model, but my 'top' comes
	// from reading order (information flows top to bottom)
	void connect(const char *layer_name_top, const char *layer_name_bottom) 
	{
		size_t i_top=layer_map[layer_name_top];
		size_t i_bottom=layer_map[layer_name_bottom];

		base_layer *l_top= layer_sets[MAIN_LAYER_SET][i_top];
		base_layer *l_bottom= layer_sets[MAIN_LAYER_SET][i_bottom];
		
		int w_i=(int)W.size();
		matrix *w = l_bottom->new_connection(*l_top, w_i);
		W.push_back(w);
		layer_graph.push_back(std::make_pair(layer_name_top,layer_name_bottom));
		// need to build connections for other batches/threads
		for(int i=1; i<(int)layer_sets.size(); i++)
		{
			l_top= layer_sets[i][i_top];
			l_bottom= layer_sets[i][i_bottom];
			delete l_bottom->new_connection(*l_top, w_i);
		}

		// we need to let solver prepare space for stateful information 
		if (_solver)
		{
			if (w)_solver->push_back(w->cols, w->rows, w->chans);
			else _solver->push_back(1, 1, 1);
		}

		int fan_in=l_bottom->fan_size();
		int fan_out=l_top->fan_size();

		// ToDo: this may be broke when 2 layers connect to one. need to fix (i.e. resnet)
		// after all connections, run through and do weights with correct fan count

		// initialize weights - ToDo: separate and allow users to configure(?)
		if (w && l_bottom->has_weights())
		{
			if (strcmp(l_bottom->p_act->name, "tanh") == 0)
			{
				// xavier : for tanh
				float weight_base = (float)(std::sqrt(6. / ((double)fan_in + (double)fan_out)));
				//		float weight_base = (float)(std::sqrt(.25/( (double)fan_in)));
				w->fill_random_uniform(weight_base);
			}
			else if ((strcmp(l_bottom->p_act->name, "sigmoid") == 0) || (strcmp(l_bottom->p_act->name, "sigmoid") == 0))
			{
				// xavier : for sigmoid
				float weight_base = 4.f*(float)(std::sqrt(6. / ((double)fan_in + (double)fan_out)));
				w->fill_random_uniform(weight_base);
			}
			else if ((strcmp(l_bottom->p_act->name, "lrelu") == 0) || (strcmp(l_bottom->p_act->name, "relu") == 0)
				|| (strcmp(l_bottom->p_act->name, "vlrelu") == 0) || (strcmp(l_bottom->p_act->name, "elu") == 0))
			{
				// he : for relu
				float weight_base = (float)(std::sqrt(2. / (double)fan_in));
				w->fill_random_normal(weight_base);
			}
			else
			{
				// lecun : orig
				float weight_base = (float)(std::sqrt(1. / (double)fan_in));
				w->fill_random_uniform(weight_base);
			}
		}
		else if (w) w->fill(0);
	}

	// automatically connect all layers in the order they were provided 
	// easy way to go, but can't deal with branch/highway/resnet/inception types of architectures
	void connect_all()
	{	
		for(int j=0; j<(int)layer_sets[MAIN_LAYER_SET].size()-1; j++) 
			connect(layer_sets[MAIN_LAYER_SET][j]->name.c_str(), layer_sets[MAIN_LAYER_SET][j+1]->name.c_str());
	}

	int get_layer_index(const char *name)
	{
		for (int j = 0; j < (int)layer_sets[MAIN_LAYER_SET].size(); j++)
			if (layer_sets[MAIN_LAYER_SET][j]->name.compare(name) == 0)
				return j;
		return -1;
	}

	// get the list of layers used (but not connection information)
	std::string get_configuration()
	{
		std::string str;
		// print all layer configs
		for (int j = 0; j<(int)layer_sets[MAIN_LAYER_SET].size(); j++) str+= "  "+ std::to_string((long long)j) +" : " +layer_sets[MAIN_LAYER_SET][j]->name +" : " + layer_sets[MAIN_LAYER_SET][j]->get_config_string();
		str += "\n";
		// print layer links
		if (layer_graph.size() <= 0) return str;
		
		for (int j = 0; j < (int)layer_graph.size(); j++)
		{
			if (j % 3 == 0) str += "  ";
			if((j % 3 == 1)|| (j % 3 == 2)) str += ", ";
			str +=layer_graph[j].first + "-" + layer_graph[j].second;
			if (j % 3 == 2) str += "\n";
		}
		return str;
	}

	// performs forward pass and returns <class index, probability>
	// do not delete or modify the returned pointer. it is a live pointer to the last layer in the network
	// if calling over multiple threads, provide the thread index since the interal data is not otherwise thread safe
	std::tuple<int, double> predict_class(const float *in, int _thread_number = -1)
	{
		const float* out = forward(in, 1.f, _thread_number);
		int argm = arg_max(out, out_size());
		return std::make_tuple(argm, out[argm]);
	}

	//----------------------------------------------------------------------------------------------------------
	// F O R W A R D
	//
	// the main forward pass 
	// if calling over multiple threads, provide the thread index since the interal data is not otherwise thread safe
	// train parameter is used to designate the forward pass is used in training (it turns on dropout layers, etc..)
	float* forward(const float *in, float temperature = 1.f, int _thread_number = -1, int _train = 0)
	{
		if(_thread_number<0) _thread_number=get_thread_num();
		if (_thread_number > _thread_count && _thread_count>0) bail("need to enable threading\n");
		if (_thread_number >= (int)layer_sets.size()) bail("need to enable threading\n");

		//std::cout << get_thread_num() << ",";
		// clear nodes to zero & find input layers
		std::vector<base_layer *> inputs;
		__for__(auto layer __in__ layer_sets[_thread_number])
		{
			if (dynamic_cast<input_layer*> (layer) != NULL)  inputs.push_back(layer);
			layer->set_threading(_internal_thread_count);
			layer->node.fill(0.f);
		}
		// first layer assumed input. copy input to it 
		const float *in_ptr = in;
		//base_layer * layer = layer_sets[_thread_number][0];

		//memcpy(layer->node.x, in, sizeof(float)*layer->node.size());
		
		__for__(auto layer __in__ inputs)
		{
			memcpy(layer->node.x, in_ptr, sizeof(float)*layer->node.size());
			in_ptr += layer->node.size();
		}
		//for (int i = 0; i < layer->node.size(); i++)
		//	layer_sets[_thread_number][0]->node.x[i] = in[i];
		// for all layers
		__for__(auto layer __in__ layer_sets[_thread_number])
		{
			// add bias and activate these outputs (they should all be summed up from other branches at this point)
			//for(int j=0; j<layer->node.chans; j+=10) for (int i=0; i<layer->node.cols*layer->node.rows; i+=10)	std::cout<< layer->node.x[i+j*layer->node.chan_stride] <<"|";
			layer->activate_nodes(temperature); 
			
			//for(int j=0; j<layer->node.chans; j++) for (int i=0; i<layer->node.cols*layer->node.rows; i+=10)	std::cout<< layer->node.x[i+j*layer->node.chan_stride] <<"|";
			// send output signal downstream (note in this code 'top' is input layer, 'bottom' is output - bucking tradition
			__for__ (auto &link __in__ layer->forward_linked_layers)
			{
				// instead of having a list of paired connections, just use the shape of W to determine connections
				// this is harder to read, but requires less look-ups
				// the 'link' variable is a std::pair created during the connect() call for the layers
				int connection_index = link.first; 
				base_layer *p_bottom = link.second;
				// weight distribution of the signal to layers under it
#ifdef MOJO_PROFILE_LAYERS
	StartCounter();
#endif
				p_bottom->accumulate_signal(*layer, *W[connection_index], _train);
				//if (p_bottom->has_weights())
			//for(int j=0; j<layer->node.chans; j++) 
			//int j=0;	for (int i=0; i<layer->node.cols*layer->node.rows; i+=10)	std::cout<< layer->node.x[i+j*layer->node.chan_stride] <<"|";

#ifdef MOJO_PROFILE_LAYERS
		std::cout << p_bottom->name << "\t" << GetCounter() << "ms\n";
#endif
			
			}

		}
		// return pointer to float * result from last layer
/*		std::cout << "out:";
		for (int i = 0; i < 10; i++)
		{
			std::cout << layer_sets[_thread_number][layer_sets[_thread_number].size() - 1]->node.x[i] <<",";
		}
		std::cout << "\n";
	*/
		return layer_sets[_thread_number][layer_sets[_thread_number].size()-1]->node.x;
	}

	int float_vector_find(std::vector<float>& indexed_unique_values, float element){

		bool find_element = false;
		for(int k = 0 ; k < indexed_unique_values.size(); k++){
			if(fabs(element - indexed_unique_values[k]) < 0.00000001f){
				find_element = true;
				return k;
			}

		}
		if(!find_element){
			indexed_unique_values.push_back(element);
		}
		return -1;
	}
	//----------------------------------------------------------------------------------------------------------
	// W R I T E
	std::string getParams() {
		std::ostringstream stream;
		// save layers
		int layer_cnt = (int)layer_sets[MAIN_LAYER_SET].size();
//		int ignore_cnt = 0;
//		for (int j = 0; j<(int)layer_sets[0].size(); j++)
//			if (dynamic_cast<dropout_layer*> (layer_sets[0][j]) != NULL)  ignore_cnt++;
		stream<<"mojo01" << std::endl;
		stream<<(int)(layer_cnt)<<std::endl;
		
		for(int j=0; j<(int)layer_sets[0].size(); j++)
			stream << layer_sets[MAIN_LAYER_SET][j]->name << std::endl << layer_sets[MAIN_LAYER_SET][j]->get_config_string();
//			if (dynamic_cast<dropout_layer*> (layer_sets[0][j]) != NULL)

		// save graph
		stream<<(int)layer_graph.size()<<std::endl;
		for(int j=0; j<(int)layer_graph.size(); j++)
			stream<<layer_graph[j].first << std::endl << layer_graph[j].second << std::endl;

			stream<<(int)0<<std::endl;
			// save bias info
			for(int j=0; j<(int)layer_sets[MAIN_LAYER_SET].size(); j++)
			{
				if (layer_sets[MAIN_LAYER_SET][j]->use_bias())
				{
					for (int k = 0; k < layer_sets[MAIN_LAYER_SET][j]->bias.size(); k++)  stream << layer_sets[MAIN_LAYER_SET][j]->bias.x[k] << " ";
					stream << std::endl;
				}
			}
			// save weights
			if(DISTILLATION_MODE){

				//Store the different weight values in a set so keep the unique
				//std::set<float> unique_values;
				std::vector<float> indexed_unique_values;
				int loop_counter =0;
				for(int j=0; j<(int)W.size(); j++)
				{
					if (W[j])
					{
						for (int i = 0; i < W[j]->size(); i++){
							float_vector_find(indexed_unique_values, W[j]->x[i]);
							//if (std::find(indexed_unique_values.begin(), indexed_unique_values.end(), W[j]->x[i]) == indexed_unique_values.end())
							//	indexed_unique_values.push_back(W[j]->x[i]);
							//unique_values.insert(W[j]->x[i]);
							loop_counter++;
						}
					}
				}


				//std::cout<<"Loop Counter" << loop_counter <<std::endl;
				//Create a vector from the set in order to have index values
				stream<<loop_counter<<std::endl;
				//std::cout<<"Unique_values size:"<<indexed_unique_values.size()<<std::endl;
				//std::vector<float> indexed_unique_values(unique_values.begin(), unique_values.end());

				//for(int index_counter=0; index_counter < indexed_unique_values.size(); index_counter++){
				//	std::cout<< index_counter << " " << indexed_unique_values[index_counter] <<std::endl;
				//}
                                //std::cout<<"INDEXED VALUE FROM GETPARAMS:"<<indexed_unique_values.size()<<std::endl;
				//Send the size of the mapping
				stream<<indexed_unique_values.size()<<std::endl;
				//std::cout<<"INDEXED VALUE FROM GETPARAMS:"<<indexed_unique_values.size()<<std::endl;
				for(int index_counter=0; index_counter < indexed_unique_values.size(); index_counter++){
					stream << index_counter << std::endl;
					stream << indexed_unique_values[index_counter] << std::endl;
				}

				for(int j=0; j<(int)W.size(); j++)
				{
					if (W[j])
					{
						for (int i = 0; i < W[j]->size(); i++){
							stream << float_vector_find(indexed_unique_values,W[j]->x[i]) << " ";//(int)std::distance(indexed_unique_values.begin(),std::find(indexed_unique_values.begin(),indexed_unique_values.end(),W[j]->x[i])) << " ";
							//std::cout<< (int)std::distance(indexed_unique_values.begin(),std::find(indexed_unique_values.begin(),indexed_unique_values.end(),W[j]->x[i]))<< W[j]->x[i] << std::endl;
							//std::cout<< float_vector_find(indexed_unique_values,W[j]->x[i])<<" "<< W[j]->x[i]<<std::endl;
						}
						stream << std::endl;
					}
				}
			}
			else{
				//stream << (int)0 <<std::endl;
				for(int j=0; j<(int)W.size(); j++)
				{
					if (W[j])
					{
						for (int i = 0; i < W[j]->size(); i++) stream << W[j]->x[i] << " ";
						stream << std::endl;
					}
				}	
			}
		std::string res = stream.str();
		return res; 
	}
	
	std::vector<float> getModelParams() {
		std::vector<float> ret;
		// save graph
		for(int j=0; j<(int)layer_graph.size(); j++)
			// save bias info
			for(int j=0; j<(int)layer_sets[MAIN_LAYER_SET].size(); j++)
				if (layer_sets[MAIN_LAYER_SET][j]->use_bias())
					for (int k = 0; k < layer_sets[MAIN_LAYER_SET][j]->bias.size(); k++)  
						ret.push_back(layer_sets[MAIN_LAYER_SET][j]->bias.x[k]);
			// save weights
			for(int j=0; j<(int)W.size(); j++)
				if (W[j])
					for (int i = 0; i < W[j]->size(); i++) 
						ret.push_back(W[j]->x[i]);
		return ret; 
	}

	//
	// write parameters to stream/file
	// note that this does not persist intermediate training information that could be needed to 'pickup where you left off'
	bool write(std::ofstream& ofs, bool binary = false, bool final = false)
	{
		// save layers
		int layer_cnt = (int)layer_sets[MAIN_LAYER_SET].size();
//		int ignore_cnt = 0;
//		for (int j = 0; j<(int)layer_sets[0].size(); j++)
//			if (dynamic_cast<dropout_layer*> (layer_sets[0][j]) != NULL)  ignore_cnt++;
		ofs<<"mojo01" << std::endl;
		ofs<<(int)(layer_cnt)<<std::endl;
		
		for(int j=0; j<(int)layer_sets[0].size(); j++)
			ofs << layer_sets[MAIN_LAYER_SET][j]->name << std::endl << layer_sets[MAIN_LAYER_SET][j]->get_config_string();
//			if (dynamic_cast<dropout_layer*> (layer_sets[0][j]) != NULL)

		// save graph
		ofs<<(int)layer_graph.size()<<std::endl;
		for(int j=0; j<(int)layer_graph.size(); j++)
			ofs<<layer_graph[j].first << std::endl << layer_graph[j].second << std::endl;

		if(binary)
		{
			ofs<<(int)1<<std::endl; // flags that this is binary data
			// binary version to save space if needed
			// save bias info
			for(int j=0; j<(int)layer_sets[MAIN_LAYER_SET].size(); j++)
				if(layer_sets[MAIN_LAYER_SET][j]->use_bias())
					ofs.write((char*)layer_sets[MAIN_LAYER_SET][j]->bias.x, layer_sets[MAIN_LAYER_SET][j]->bias.size()*sizeof(float));
			// save weights
			for (int j = 0; j < (int)W.size(); j++)
			{
				if (W[j])
					ofs.write((char*)W[j]->x, W[j]->size()*sizeof(float));
			}
		}
		else
		{
			ofs<<(int)0<<std::endl;
			// save bias info
			for(int j=0; j<(int)layer_sets[MAIN_LAYER_SET].size(); j++)
			{
				if (layer_sets[MAIN_LAYER_SET][j]->use_bias())
				{
					for (int k = 0; k < layer_sets[MAIN_LAYER_SET][j]->bias.size(); k++)  ofs << layer_sets[MAIN_LAYER_SET][j]->bias.x[k] << " ";
					ofs << std::endl;
				}
			}
			// save weights
			for(int j=0; j<(int)W.size(); j++)
			{
				if (W[j])
				{
					for (int i = 0; i < W[j]->size(); i++) ofs << W[j]->x[i] << " ";
					ofs << std::endl;
				}
			}
		}
		ofs.flush();
		
		return true;
	}
	bool write(std::string &filename, bool binary = false, bool final = false) { 
		std::ofstream temp((const char *)filename.c_str(), std::ios::binary);
		return write(temp, binary, final);
	}//, std::ofstream::binary);

	bool write(char *filename, bool binary = false, bool final = false) 
	{
		std::string str= filename;
		return write(str, binary, final); 
	}

	// read network from a file/stream
	
	std::string getcleanline(std::istream& ifs)
	{
		std::string s;

		// The characters in the stream are read one-by-one using a std::streambuf.
		// That is faster than reading them one-by-one using the std::istream.
		// Code that uses streambuf this way must be guarded by a sentry object.
		// The sentry object performs various tasks,
		// such as thread synchronization and updating the stream state.

		std::istream::sentry se(ifs, true);
		std::streambuf* sb = ifs.rdbuf();

		for (;;) {
			int c = sb->sbumpc();
			switch (c) {
			case '\n':
				return s;
			case '\r':
				if (sb->sgetc() == '\n') sb->sbumpc();
				return s;
			case EOF:
				// Also handle the case when the last line has no line ending
				if (s.empty()) ifs.setstate(std::ios::eofbit);
				return s;
			default:
				s += (char)c;
			}
		}
	}
	
	//----------------------------------------------------------------------------------------------------------
	// R E A D
	//
	void fetchParams(std::string &params) {
		std::istringstream ss(params);
		read(ss);
	}
		
	bool read(std::istream &ifs)
	{
		if(!ifs.good()) return false;
		std::string s;
		s = getcleanline(ifs);
		int layer_count;
		int version = 0;
		if (s.compare("mojo01")==0)
		{
			s = getcleanline(ifs);
			layer_count = atoi(s.c_str());
			version = 1;
			//std::cout<<"READ: Layer count:"<<layer_count<<std::endl;

		}
		else if (s.compare("mojo:") == 0)
		{
			version = -1;
			int cnt = 1;

			while (!ifs.eof())
			{
				s = getcleanline(ifs);
				if (s.empty()) continue;
				push_back(int2str(cnt).c_str(), s.c_str());
				cnt++;
			}
			connect_all();

			// copies batch=0 stuff to other batches
			sync_layer_sets();
			return true;
		}
		else
			layer_count = atoi(s.c_str());
		// read layer def
		std::string layer_name;
		std::string layer_def;
		for (auto i=0; i<layer_count; i++)
		{
			layer_name = getcleanline(ifs);
			layer_def = getcleanline(ifs);
			push_back(layer_name.c_str(),layer_def.c_str());
		}

		//std::cout<<"READ: Layer Name Successful"<<std::endl;

		// read graph
		int graph_count;
		ifs>>graph_count;
		//std::cout<<"Read: Graph Count"<<graph_count<<std::endl;
		getline(ifs,s); // get endline
		if (graph_count <= 0)
		{
			connect_all();
		}
		else
		{
		std::string layer_name1;
		std::string layer_name2;
		for (auto i=0; i<graph_count; i++)
		{
			layer_name1= getcleanline(ifs);
			layer_name2 = getcleanline(ifs);
			connect(layer_name1.c_str(),layer_name2.c_str());
		}
		}
		//std::cout<<"Read: Layer Connection Successful"<<std::endl;

		int binary;
		s=getcleanline(ifs); // get endline
		binary = atoi(s.c_str());
		//std::cout<<"Read: Binary value:"<<binary<<std::endl;

		int distillation_compression;
		// binary version to save space if needed
		if(binary==1)
		{
			for(int j=0; j<(int)layer_sets[MAIN_LAYER_SET].size(); j++)
				if (layer_sets[MAIN_LAYER_SET][j]->use_bias())
				{
					//int c = layer_sets[MAIN_LAYER_SET][j]->bias.chans;
					//int cs = layer_sets[MAIN_LAYER_SET][j]->bias.chan_stride;
					//for (int i = 0; i < layer_sets[MAIN_LAYER_SET][j]->bias.size; i++)
						ifs.read((char*)layer_sets[MAIN_LAYER_SET][j]->bias.x, layer_sets[MAIN_LAYER_SET][j]->bias.size()*sizeof(float));
				}
			
			for (int j = 0; j < (int)W.size(); j++)
			{

				if (W[j])
				{
					ifs.read((char*)W[j]->x, W[j]->size()*sizeof(float));
				}
			}

		}
		else if(binary==0)// text version
		{
			// read bias
			for(int j=0; j<layer_count; j++)
			{
				if (layer_sets[MAIN_LAYER_SET][j]->use_bias())
				{
				//	int c = layer_sets[MAIN_LAYER_SET][j]->bias.chans;
				//	int cs = layer_sets[MAIN_LAYER_SET][j]->bias.chan_stride;

				//	for (int i = 0; i < c; i++)
					for (int k = 0; k < layer_sets[MAIN_LAYER_SET][j]->bias.size(); k++)
					{
						ifs >> layer_sets[MAIN_LAYER_SET][j]->bias.x[k];
						//std::cout << layer_sets[MAIN_LAYER_SET][j]->bias.x[k] << ",";
					}
					ifs.ignore();// getline(ifs, s); // get endline
				}
			}
			std::map<int, float> unique_mapping;
			//Empty line after the layers
			if(DISTILLATION_MODE){
				ifs.ignore();
				int loop_counter;
				s = getcleanline(ifs);
				loop_counter = atoi(s.c_str());
				//std::cout<<"Read Loop Counter is:"<<loop_counter<<std::endl;
				int unique_values_size;
				s = getcleanline(ifs);
				unique_values_size = atoi(s.c_str());

				//std::cout<<"INDEXED VALUE FROM READ:"<<unique_values_size<<std::endl;
				int vector_idx;
				float vector_val;
				for(int idx_counter=0; idx_counter < unique_values_size; idx_counter++){
					ifs >> vector_idx;
					ifs.ignore();
					ifs >> vector_val;
					ifs.ignore();
					//s = getcleanline(ifs);
					//vector_idx = atoi(s.c_str());
					//s = getcleanline(ifs);
					//vector_val = atof(s.c_str());
					//std::cout<<vector_idx<< " "<<vector_val<<std::endl;
					unique_mapping.insert({vector_idx,vector_val});
				}
				//std::cout<<"SET OF PAIRS SIZE:"<<unique_mapping.size()<<std::endl;
				int translator;
				for (auto j=0; j<(int)W.size(); j++)
				{
					if (W[j])
					{
						for (int i = 0; i < W[j]->size(); i++){
							 ifs >> translator;
							 //std::cout<<"Translator:"<<translator<<" "<<unique_mapping[translator]<<std::endl;
							 W[j]->x[i] = unique_mapping[translator];
						}
						ifs.ignore(); //getline(ifs, s); // get endline
					}
				}
			}
			else{
				for (auto j=0; j<(int)W.size(); j++)
                  		{
                               		if (W[j])
                               		{
                                        	for (int i = 0; i < W[j]->size(); i++) ifs >> W[j]->x[i];
                                        	ifs.ignore(); //getline(ifs, s); // get endline
                                	}
                        	}
			}
		}
	
		// copies batch=0 stuff to other batches
		sync_layer_sets();
		return true;
	}
	bool read(std::string filename)
	{
		std::ifstream fs(filename.c_str(),std::ios::binary);
		if (fs.is_open())
		{
			bool ret = read(fs);
			fs.close();
			return ret;
		}
		else return false;
	}
	bool read(const char *filename) { return  read(std::string(filename)); }

	// returns a vector representation of dw_sets[0] and dbias_sets[0]
	// !must be invoked right after sync_mini_batch() invoke 
	// (i.e., at the end of mini-batch processing so that all gradients are
	// summed to dw_sets[0] and dbias_sets[0])
	// Returns: ret[0]: size of dw_sets[0]
	// 			ret[1]: size of dw_sets[0][0]
	// 			ret[2]: dw_sets[0][0][0]
	// 			ret[3]: dw_sets[0][0][1]
	// 			...
	// 			ret[n]: size of dbias[0]
	// 			...
	std::vector<float> gradients() {
		std::vector<float> ret;
		// weight gradients
        	sync_mini_batch();
		ret.push_back(dW_sets[0].size());
		for (int i=0; i<dW_sets[0].size(); i++) {
			ret.push_back(dW_sets[0][i].size());
			for (int j=0; j<dW_sets[0][i].size(); j++)
				ret.push_back(dW_sets[0][i].x[j]);
		}
		// bias gradients
		ret.push_back(dbias_sets[0].size());
		for (int i=0; i<dbias_sets[0].size(); i++) {
			ret.push_back(dbias_sets[0][i].size());
			for (int j=0; j<dbias_sets[0][i].size(); j++)
				ret.push_back(dbias_sets[0][i].x[j]);
		}
		return ret;
	}

		int sizeOfGradients() {
			int params = 0;
			// weight gradients
			for (int i=0; i<dW_sets[0].size(); i++)
				params += dW_sets[0][i].size();
			// bias gradients
			for (int i=0; i<dbias_sets[0].size(); i++)
				params += dbias_sets[0][i].size();
			return params;
		}

	// adds g2 to g1
	void addGradients(std::vector<float> &g1, std::vector<float> g2) {
		int idx=0;
		// weight gradients
		idx++;
		for (int i=0; i<dW_sets[0].size(); i++) {
			idx++;
			for (int j=0; j<dW_sets[0][i].size(); j++) {
				g1[idx] += g2[idx];
				idx++;
			}
		}
		// bias gradients
		idx++;
		for (int i=0; i<dbias_sets[0].size(); i++) {
			idx++;
			for (int j=0; j<dbias_sets[0][i].size(); j++) { 
				g1[idx] += g2[idx];
				idx++;
			}
		}
	}

	void scale_gradients(float C) {
        int layer_cnt = (int)layer_sets[MAIN_LAYER_SET].size();
        base_layer *layer;

        for (int b = 1; b< _batch_size; b++) {
            float sqsum = 0;
            for (int k = layer_cnt - 1; k >= 0; k--) {
                layer = layer_sets[MAIN_LAYER_SET][k];

                __for__ (auto &link __in__ layer->backward_linked_layers) {
                    int w_index = (int) link.first;
                    for (int j=0; j<dW_sets[b][w_index].size(); j++)
                        sqsum += dW_sets[b][w_index].x[j] * dW_sets[b][w_index].x[j];
                }
                for (int j=0; j<dbias_sets[b][k].size(); j++)
                    sqsum += dbias_sets[b][k].x[j] * dbias_sets[b][k].x[j];
            }
            float norm = sqrt(sqsum);
            //__android_log_print(ANDROID_LOG_DEBUG, "INFO", "Norm: %6.4lf", norm);
            float scale = 1 / std::max<float>(1.0, norm / C);

            for (int k = layer_cnt - 1; k >= 0; k--) {
                __for__ (auto &link __in__ layer->backward_linked_layers) {
                    int w_index = (int) link.first;
                    for (int j=0; j<dW_sets[b][w_index].size(); j++)
                        dW_sets[b][w_index].x[j] *= scale;
                }
                for (int j=0; j<dbias_sets[b][k].size(); j++)
                    dbias_sets[b][k].x[j] *= scale;
            }
        }
	}

	void addGaussian(std::vector<float> &v, float sigma) {
	    std::default_random_engine generator;
	    std::normal_distribution<double> dist(0.0, sigma);
	    for (int i = 0; i < v.size(); i++)
	        v[i] += dist(generator);
	}

	// same as gradients() but adds noise for differential privacy
	std::vector<float> DPgradients(float C, float sigma) {

		std::vector<float> ret;
		std::vector<int> sizes;
		std::vector<int> indices;
		int idx = 0;

        	scale_gradients(C);
        	sync_mini_batch();

		// weight gradients
		sizes.push_back(dW_sets[0].size());
		indices.push_back(idx);
		ret.push_back(0); // setting 0 instead of the size will not affect the norm computation
		idx++;
		for (int i=0; i<dW_sets[0].size(); i++) {
			sizes.push_back(dW_sets[0][i].size());
			indices.push_back(idx);
			ret.push_back(0);
			idx++;
			for (int j=0; j<dW_sets[0][i].size(); j++) {
				ret.push_back(dW_sets[0][i].x[j]);
				idx++;
			}
		}
		// bias gradients
		sizes.push_back(dbias_sets[0].size());
		indices.push_back(idx);
		ret.push_back(0);
		idx++;
		for (int i=0; i<dbias_sets[0].size(); i++) {
			sizes.push_back(dbias_sets[0][i].size());
			indices.push_back(idx);
			ret.push_back(0);
			idx++;
			for (int j=0; j<dbias_sets[0][i].size(); j++) { 
				ret.push_back(dbias_sets[0][i].x[j]);
				idx++;
			}
		}

        	addGaussian(ret, C * sigma);

		// copy back sizes to ret
		for (int i=0; i<sizes.size(); i++)
			ret[indices[i]] = sizes[i];

		return ret;
	}

	// performs model update (descent) with the given gradients in a vector
	// representation (see gradients() method)
	void descent(std::vector<float> g) {
		int idx = 0;
		int dWSize = g[idx++];
		for (int i=0; i<dWSize; i++) {
			int size = g[idx++];
			for (int j=0; j<size; j++) 
				dW_sets[0][i].x[j] = g[idx++];
		}

		int dbiasSize= g[idx++];
		for (int i=0; i<dbiasSize; i++) {
			int size = g[idx++];
			for (int j=0; j<size; j++) 
				dbias_sets[0][i].x[j] = g[idx++];
		}

		descent();
	}

	// returns a flat representation of the gradient without additional info
	// (e.g., sizes)
	std::vector<float> flatGrad(std::vector<float> g) {
		int idx = 0;
		std::vector<float> ret;
		int dWSize = g[idx++];
		for (int i=0; i<dWSize; i++) {
			int size = g[idx++];
			for (int j=0; j<size; j++) 
				ret.push_back(g[idx++]);
		}

		int dbiasSize= g[idx++];
		for (int i=0; i<dbiasSize; i++) {
			int size = g[idx++];
			for (int j=0; j<size; j++) 
				ret.push_back(g[idx++]);
		}
		return ret;
	}

	// merges flat gradient to given gradient
	void mergeFlatGrad(std::vector<float> &g, std::vector<float> flat) {
		int idx = 0;
		int idx2 = 0;
		int dWSize = g[idx++];
		for (int i=0; i<dWSize; i++) {
			int size = g[idx++];
			for (int j=0; j<size; j++) 
				g[idx++] = flat[idx2++];
		}

		int dbiasSize= g[idx++];
		for (int i=0; i<dbiasSize; i++) {
			int size = g[idx++];
			for (int j=0; j<size; j++) 
				g[idx++] = flat[idx2++];
		}
	}

#ifndef MOJO_NO_TRAINING  // this is surely broke by now and will need to be fixed

	// ===========================================================================
	// training part
	// ===========================================================================

	// resets the state of all batches to 'free' state
	void reset_mini_batch() { memset(batch_open.data(), BATCH_FREE, batch_open.size()); }
	
	// sets up number of mini batches (storage for sets of weight deltas)
	void set_mini_batch_size(int batch_cnt)
	{
		if (batch_cnt<1) batch_cnt = 1;
		_batch_size = batch_cnt;
		dW_sets.resize(_batch_size);
		dbias_sets.resize(_batch_size);
		batch_open.resize(_batch_size); 
		reset_mini_batch();
	}
	
	int get_mini_batch_size() { return _batch_size; }

	// return index of next free batch
	// or returns -2 (BATCH_FILLED_COMPLETE) if no free batches - all complete (need a sync call)
	// or returns -1 (BATCH_FILLED_IN_PROCESS) if no free batches - some still in progress (must wait to see if one frees)
	int get_next_open_batch()
	{
		int reserved = 0;
		int filled = 0;
		for (int i = 0; i<batch_open.size(); i++)
		{
			if (batch_open[i] == BATCH_FREE) return i;
			if (batch_open[i] == BATCH_RESERVED) reserved++;
			if (batch_open[i] == BATCH_COMPLETE) filled++;
		}
		if (reserved>0) return BATCH_FILLED_IN_PROCESS; // all filled but wainting for reserves
		if (filled == batch_open.size()) return BATCH_FILLED_COMPLETE; // all filled and complete
		
		bail("threading error"); // should not get here  unless threading problem
	}

	//----------------------------------------------------------------------------------------------------------
	// s y n c   m i n i   b a t c h
	//
	// apply all weights to first set of dW, then apply to model weights 
	void sync_mini_batch()
	{
		// need to ensure no batches in progress (reserved)
//		int next = get_next_open_batch();
//		if (next == BATCH_FILLED_IN_PROCESS) bail("thread lock");

		int layer_cnt = (int)layer_sets[MAIN_LAYER_SET].size();

		base_layer *layer;

		 // sum contributions 
		for (int k = layer_cnt - 1; k >= 0; k--)
		{
			layer = layer_sets[MAIN_LAYER_SET][k];
			__for__(auto &link __in__ layer->backward_linked_layers)
			{
				int w_index = (int)link.first;
				// if batch free, then make sure it is zero'd out because we will increment dW set [0]
				//if (batch_open[0] == BATCH_FREE) dW_sets[0][w_index].fill(0);
				for (int b = 1; b< _batch_size; b++)
				{
					/*if (batch_open[b] == BATCH_COMPLETE)*/ dW_sets[0][w_index] += dW_sets[b][w_index];
				}
			}
			if (dynamic_cast<convolution_layer*> (layer) != NULL)  continue;

			// bias stuff... that needs to be fixed for conv layers perhaps
			//if (batch_open[0] == BATCH_FREE) dbias_sets[0][k].fill(0);
			for (int b = 1; b< _batch_size; b++)
			{
				/*if (batch_open[b] == BATCH_COMPLETE)*/ dbias_sets[0][k] += dbias_sets[b][k];
			}
		}

//		descent();

		// prepare to start mini batch over
//		reset_mini_batch();
		batch_index = 0;
//		train_updates++; // could have no updates .. so this is not exact
//		sync_layer_sets();

	}
	
	// performs the descent operation (i.e., updates weights and gradients)
	void descent() {
		int layer_cnt = (int)layer_sets[MAIN_LAYER_SET].size();
		base_layer *layer;

		// update weights
		for (int k = layer_cnt - 1; k >= 0; k--)
		{
			layer = layer_sets[MAIN_LAYER_SET][k];
			__for__(auto &link __in__ layer->backward_linked_layers)
			{
				int w_index = (int)link.first;
				if (dW_sets[MAIN_LAYER_SET][w_index].size() > 0)
					if(W[w_index]) _solver->increment_w(W[w_index], w_index, dW_sets[MAIN_LAYER_SET][w_index]);  // -- 10%

			}
			layer->update_bias(dbias_sets[0][k], _solver->learning_rate);
		}
		train_updates++;
		sync_layer_sets();
	}

	// reserve_next.. is used to reserve a space in the minibatch for the existing training sample
	int reserve_next_batch()
	{
		lock_batch();
		int my_batch_index = -3;
		while (my_batch_index < 0)
		{
			my_batch_index = get_next_open_batch();

			if (my_batch_index >= 0) // valid index
			{
				batch_open[my_batch_index] = BATCH_RESERVED;
				unlock_batch();
				return my_batch_index;
			}
			else if (my_batch_index == BATCH_FILLED_COMPLETE) // all index are complete
			{
//				sync_mini_batch(); // MY edit commented out/ resets _batch_index to 0
				my_batch_index = get_next_open_batch();
				batch_open[my_batch_index] = BATCH_RESERVED;
				unlock_batch();
				return my_batch_index;
			}
			// need to wait for ones in progress to finish
			unlock_batch();
			mojo_sleep(1);
			lock_batch();
		}
		return -3;
	}

	float get_learning_rate() {if(!_solver) bail("set solver"); return _solver->learning_rate;}
	void set_learning_rate(float alpha) {if(!_solver) bail("set solver"); _solver->learning_rate=alpha;}
	void reset_solver() {if(!_solver) bail("set solver"); _solver->reset();}
	bool get_smart_training() {return _smart_train;}
	void set_smart_training(bool _use_train) { _smart_train = _use_train;}
	float get_smart_train_level() { return _skip_energy_level; }
	void set_smart_train_level(float _level) { _skip_energy_level = _level; }
	void set_max_epochs(int max_e) { if (max_e <= 0) max_e = 1; max_epochs = max_e; }
	int get_epoch() { return epoch_count; }

	// goal here is to update the weights W. 
	// use w_new = w_old - alpha dE/dw
	// E = sum: 1/2*||y-target||^2
	// note y = f(x*w)
	// dE = (target-y)*dy/dw = (target-y)*df/dw = (target-y)*df/dx* dx/dw = (target-y) * df * y_prev  
	// similarly for cross entropy

// ===========================================================================
// training part
// ===========================================================================

	void set_random_augmentation(int translate_x, int translate_y,
		int flip_h, int flip_v, mojo::pad_type padding = mojo::edge)
	{
		use_augmentation = 1;
		augment_x = translate_x;
		augment_y = translate_y;
		augment_h_flip = flip_h;
		augment_v_flip = flip_v;
		augment_pad = padding;
		augment_theta = 0;
		augment_scale = 0;

	}
	void set_random_augmentation(int translate_x, int translate_y,
		int flip_h, int flip_v, float rotation_deg, float scale, mojo::pad_type padding = mojo::edge)
	{
		use_augmentation = 2;
		augment_x = translate_x;
		augment_y = translate_y;
		augment_h_flip = flip_h;
		augment_v_flip = flip_v;
		augment_pad = padding;
		augment_theta = rotation_deg;
		augment_scale = scale;

	}

	// call before starting training for current epoch
	void start_epoch(std::string loss_function="mse")
	{
		_cost_function=new_cost_function(loss_function);
		train_correct = 0;
		train_skipped = 0;
		train_updates = 0;
		train_samples = 0;
		if (epoch_count == 0) reset_solver();
	
		// accuracy not improving .. slow learning
		if(_smart_train &&  (best_accuracy_count > 4))
		{
			stuck_counter++;
			set_learning_rate((0.5f)*get_learning_rate());
			if (get_learning_rate() < 0.000001f)
			{
//				heat_weights();
				set_learning_rate(0.000001f);
				stuck_counter++;// end of the line.. so speed up end
			}
			best_accuracy_count = 0;
		}

		old_estimated_accuracy = estimated_accuracy;
		estimated_accuracy = 0;
		//_skip_energy_level = 0.05;
		_running_sum_E = 0;
	}
	
	// time to stop?
	bool elvis_left_the_building()
	{
		// 2 stuck x 4 non best accuracy to quit = 8 times no improvement 
		if ((epoch_count>max_epochs) || (stuck_counter > 3)) return true;
		else return false;
	}

	// call after putting all training samples through this epoch
	bool end_epoch()
	{
		// run leftovers through mini-batch
//		sync_mini_batch();  // MY edit
		epoch_count++;

		// estimate accuracy of validation run 
		estimated_accuracy = 100.f*train_correct / train_samples;

		if (train_correct > best_estimated_accuracy)
		{
			best_estimated_accuracy = (float)train_correct;
			best_accuracy_count = 0;
			stuck_counter = 0;
		}
		else best_accuracy_count++;

		return elvis_left_the_building();
	}

	// if smart training was thinking about exiting, calling reset will make it think everything is OK
	void reset_smart_training()
	{
		stuck_counter=0;
		best_accuracy_count = 0;
		best_estimated_accuracy = 0;
	}

	//----------------------------------------------------------------------------------------------------------
	// u p d a t e _ s m a r t _ t r a i n
	//
	void update_smart_train(const float E, bool correct)
	{

#ifdef MOJO_OMP	
#pragma omp critical
#endif
		{
			train_samples++;
			if (correct) train_correct++;

			if (_smart_train)
			{
				_running_E.push_back(E);
				_running_sum_E += E;
				const int SMART_TRAIN_SAMPLE_SIZE = 1000;

				int s = (int)_running_E.size();
				if (s >= SMART_TRAIN_SAMPLE_SIZE)
				{
					_running_sum_E /= (double)s;
					std::sort(_running_E.begin(), _running_E.end());
					float top_fraction = (float)_running_sum_E*10.f; //10.
					const float max_fraction = 0.75f;
					const float min_fraction = 0.075f;// 0.03f;

					if (top_fraction > max_fraction) top_fraction = max_fraction;
					if (top_fraction < min_fraction) top_fraction = min_fraction;
					int index = s - 1 - (int)(top_fraction*(s - 1));

					if (_running_E[index] > 0) _skip_energy_level = _running_E[index];

					_running_sum_E = 0;

					_running_E.clear();
				}
			}
			if (E > 0 && E < _skip_energy_level)
			{
				//std::cout << "E=" << E;
				train_skipped++;
			}

		}  // omp critical


	}
	// finish back propogation through the hidden layers
	void backward_hidden(const int my_batch_index, const int thread_number, float temperature)
	{
		const int layer_cnt = (int)layer_sets[thread_number].size();
		const int last_layer_index = layer_cnt - 1;
		base_layer *layer;// = layer_sets[thread_number][last_layer_index];

		// update hidden layers
		// start at lower layer and push information up to previous layer
		// handle dropout first

		for (int k = last_layer_index; k >= 0; k--)
		{
			layer = layer_sets[thread_number][k];
			// all the signals should be summed up to this layer by now, so we go through and take the grad of activiation
			int nodes = layer->node.size();
			// already did last layer, so skip it
			if (k< last_layer_index)
				for (int i = 0; i< nodes; i++)
					layer->delta.x[i] *= layer->df(layer->node.x, i, nodes, temperature);

			// now pass that signal upstream
			__for__(auto &link __in__ layer->backward_linked_layers) // --- 50% of time this loop
			{
				base_layer *p_top = link.second;
				// note all the delta[connections[i].second] should have been calculated by time we get here
				layer->distribute_delta(*p_top, *W[link.first]);
			}
		}


		// update weights - shouldn't matter the direction we update these 
		// we can stay in backwards direction...
		// it was not faster to combine distribute_delta and increment_w into the same loop
		int size_W = (int)W.size();
		dW_sets[my_batch_index].resize(size_W);
		dbias_sets[my_batch_index].resize(layer_cnt);
		for (int k = last_layer_index; k >= 0; k--)
		{
			layer = layer_sets[thread_number][k];

			__for__(auto &link __in__ layer->backward_linked_layers)
			{
				base_layer *p_top = link.second;
				int w_index = (int)link.first;
				//if (dynamic_cast<max_pooling_layer*> (layer) != NULL)  continue;
				layer->calculate_dw(*p_top, dW_sets[my_batch_index][w_index]);// --- 20%
																			  // moved this out to sync_mini_batch();
																			  //_solver->increment_w( W[w_index],w_index, dW_sets[_batch_index][w_index]);  // -- 10%
			}
			if (dynamic_cast<convolution_layer*> (layer) != NULL)  continue;

			dbias_sets[my_batch_index][k] = layer->delta;
		}
		// if all batches finished, update weights
//		lock_batch();
//		batch_open[my_batch_index] = BATCH_COMPLETE;
//		int next_index = get_next_open_batch();
//		if (next_index == BATCH_FILLED_COMPLETE) // all complete
//			sync_mini_batch(); // MY edit commented out/ resets _batch_index to 0
//		unlock_batch();
	}


	mojo::matrix make_input(float *in, const int _thread_number)
	{
		mojo::matrix augmented_input;// = auto_augmentation();

		std::vector<base_layer *> inputs;
		int in_size = 0;
		__for__(auto layer __in__ layer_sets[_thread_number])
		{
			if (dynamic_cast<input_layer*> (layer) != NULL)
			{
				inputs.push_back(layer);
				in_size += layer->node.size();
			}
		}


		if (use_augmentation > 0)
		{

			augmented_input.resize(in_size, 1, 1);
			float s = ((float)(rand() % 101) / 50.f - 1.f)*augment_scale;
			float t = ((float)(rand() % 101) / 50.f - 1.f)*augment_theta;
			bool flip_h = ((rand() % 2)*augment_h_flip) ? true: false;
			bool flip_v = ((rand() % 2)*augment_v_flip) ? true: false;
			int shift_x = (rand() % (augment_x * 2 + 1)) - augment_x;
			int shift_y = (rand() % (augment_y * 2 + 1)) - augment_y;
			int offset = 0;
			__for__(auto layer __in__ inputs)
			{
				//memcpy(layer->node.x, in_ptr, sizeof(float)*layer->node.size());
				//in_ptr += layer->node.size();
				// copy input to matrix type
				mojo::matrix m(layer->node.cols, layer->node.rows, layer->node.chans, in + offset);
				if (m.rows > 1 && m.cols > 1)
				{
#if defined(MOJO_CV2) || defined(MOJO_CV3)
					if ((augment_theta > 0 || augment_scale > 0))
						m = transform(m, m.cols / 2, m.rows / 2, m.cols, t, 1 + s);
#endif
					if (flip_v)m = m.flip_cols();
					if (flip_h)	m = m.flip_rows();
					mojo::matrix aug = m.shift(shift_x, shift_y, augment_pad);
					memcpy(augmented_input.x + offset, aug.x, sizeof(float)*aug.size());
					offset += aug.size();

				}
				else
				{
					memcpy(augmented_input.x + offset, m.x, sizeof(float)*m.size());
					offset += m.size();
				}
			}
//			input = augmented_input.x;
		}
		else
		{
			augmented_input.resize(in_size, 1, 1);
			memcpy(augmented_input.x, in, sizeof(float)*in_size);
		}
		return augmented_input;
	}

	//QUANTIZATION
	// Here we introduce our quantization. The quantization process follows the one from the paper
	// https://arxiv.org/pdf/1802.05668.pdf , since its an experimental analysis we just implement:
	// 1. Only the linear scaling function sc(u)
	// 2. Only the uniform quantization function Q
	// 3. Also quantization function is the determenistic rounding and not the stohastic one

	void quantization_weight_model(int num_bits = 8, int bucket_size = 128) {	//num_bits = 2 for testing

		//Here we introduce the quantization of our model
		//s = 2 ** num_bits for the linear case
		std::vector<std::vector<float>> weight_map;
		int s = pow (2, num_bits);

		//scale down the weights of the network as a first step
		float alpha, beta;
		s =  s - 1;

		bool bucketing = false;


		if (bucketing) {
			//Scale down the vector in buckets to maintain the variance and not so many 0 values
			for (int j = 0; j < (int)W.size(); j++) {
				if (W[j]) {
					// std::cout<<"The size of the processing matrix is: "<<W[j]->size() <<std::endl;
					// for (int i = 0; i < W[j]->size(); i++) std::cout << W[j]->x[i] << " ";
					// std::cout << std::endl;
					// std::cout<<"/////////////////////////////////////////////"<<std::endl;
					W[j]->bucket_scaling(bucket_size, s);
					// for (int i = 0; i < W[j]->size(); i++) std::cout << W[j]->x[i] << " ";
					// std::cout << std::endl;
				}
			}
			//std::cout<<"\n\n\n~~~~~~~~~~~~~~~~FINITO~~~~~~~~~~~~~~~\n\n\n";
		} else {
			float min, max;
			for (int j = 0; j < (int)W.size(); j++) {
				if (W[j]) {

					//SCALING PROCEDURE
					//std::cout << "The W[j] is : " << *W[j] <<std::endl;
					W[j]->min_max(&min, &max);
					alpha = max - min;
					beta = min;

					//std::cout<<"MIN: "<<min<<" MAX: "<<max<<" ALPHA: "<<alpha<<" BETA: "<<beta<<std::endl;

					*W[j] = *W[j] + (-1.0) * beta;		//substraction with beta
					// std::cout<<"AFTER THE BETA TRANSFORMATION"<<std::endl;
					// std::cout<<getParams()<<std::endl;

					*W[j] = *W[j] * (1.0 / alpha);		//division with alpha
					// std::cout<<"AFTER THE ALPHA TRANSFORMATION"<<std::endl;
					// std::cout<<getParams()<<std::endl;


					//QUANTIZATION PROCEDURE

					//*W[j] = *W[j] * s;
					//std::cout<<"AFTER THE S TRANSFORMATION"<<std::endl;
					//std::cout<<getParams()<<std::endl;

					W[j]->round_matrix(s);
					//std::cout<<"AFTER THE ROUND TRANSFORMATION"<<std::endl;


					/*This procedure will create a dictionary to map the float values
					in integers to save some bandwidth*/
					/*std::vector<float> temp;
					for (float i = 0.0 ; i <= 1.0 ; i += 1 / s) {
						temp.push(i);
						temp.apply(lambda x: x * 1 / s);
						temp.apply(lambda x: x * alpha);
						temp.apply(lambda x: x + beta);
					}
*/
					//W[j]->int_mapping(temp);

					// 	*dict *= (1.0 / s);
					// 	*dict *= alpha;
					// 	*dict += beta;
					// }



					//std::cout<<getParams()<<std::endl;
					//*W[j] = *W[j] * (1.0 / s);


					//INVERSING SCALING DOWN
					*W[j] = *W[j] * alpha;
					*W[j] = *W[j] + beta;		//substraction with beta

//					W[j]->int_mapping(temp);
				}
			}
		}
	}

	void save_model_weights(std::vector<matrix*>* old_weights) {
		for (int j = 0; j < (int)W.size(); j++) {
			if (W[j]) {
				matrix* old_weight = new matrix(*W[j]);
				old_weights->push_back(old_weight);
			} else {
				old_weights->push_back(NULL);
			}
		}
	}
	void load_model_weights(std::vector<matrix*>& unquantized_weights) {

		for (std::vector<matrix *>::iterator it = W.begin() ; it != W.end(); ++it) {
			delete (*it);
		}
		W.clear();

		for (int j = 0; j < (int)unquantized_weights.size(); j++) {
			W.push_back(unquantized_weights[j]);
		}
	}


	//----------------------------------------------------------------------------------------------------------
	// T R A I N   C L A S S 
	//
	// after starting epoch, call this to train against a class label
	// label_index must be 0 to out_size()-1
	// for thread safety, you must pass in the thread_index if calling from different threads

	bool train_class(float *in, int label_index, std::vector<float>* teacher_prob = NULL, int _thread_number = -1, bool debug=false)
	{

		if (_solver == NULL) bail("set solver");
		if (_thread_number < 0) _thread_number = get_thread_num();
		if (_thread_number > _thread_count)  bail("call allow_threads()");

		const int thread_number = _thread_number;
/*
		mojo::matrix augmented_input = make_input(in, thread_number);

/*/
		float *input = in;
		mojo::matrix augmented_input;
		if (use_augmentation > 0)
		{
			//augment_h_flip = flip_h;
			//augment_v_flip = flip_v;
			// copy input to matrix type
			mojo::matrix m(layer_sets[thread_number][0]->node.cols, layer_sets[thread_number][0]->node.rows, layer_sets[thread_number][0]->node.chans, in);
#if defined(MOJO_CV2) || defined(MOJO_CV3)
			if (augment_theta > 0 || augment_scale > 0)
			{
				float s = ((float)(rand() % 101) / 50.f - 1.f)*augment_scale;
				float t = ((float)(rand() % 101) / 50.f - 1.f)*augment_theta;
				m = transform(m, m.cols / 2, m.rows / 2, m.cols, t, 1+s);
			}
#endif
			if (augment_h_flip)
				if ((rand() % 2) == 0)
					m = m.flip_cols();
			if (augment_v_flip)
				if ((rand() % 2) == 0)
					m = m.flip_rows();
			augmented_input = m.shift((rand() % (augment_x * 2 + 1)) - augment_x, (rand() % (augment_y * 2 + 1)) - augment_y, augment_pad);
			
			input = augmented_input.x;
		}


//*/

		// get next free mini_batch slot
		// this is tied to the current state of the model
		int my_batch_index = batch_index++;
		// out of data or an error if index is negative
		if (my_batch_index < 0) return false;


		// run through forward to get nodes activated

		if (teacher_prob == NULL) {
			forward(input, 1, thread_number, 1);
		} else {
			forward(input, TEMPERATURE, thread_number, 1);
		}


		//forward(input, thread_number, 1, debug);


		// set all deltas to zero
		__for__(auto layer __in__ layer_sets[thread_number]) layer->delta.fill(0.f);

		int layer_cnt = (int)layer_sets[thread_number].size();

		// calc delta for last layer to prop back up through network
		// d = (target-out)* grad_activiation(out)
		const int last_layer_index = layer_cnt - 1;
		base_layer *layer = layer_sets[thread_number][last_layer_index];
		const int layer_node_size = layer->node.size();
		const int layer_delta_size = layer->delta.size();

		if (dynamic_cast<dropout_layer*> (layer) != NULL) bail("can't have dropout on last layer");

		float E = 0;
		int max_j_out = 0;
		int max_j_target = label_index;

		// was passing this in, but may as well just create it on the fly
		// a vector mapping the label index to the desired target output node values
		// all -1 except target node 1
		std::vector<float> target;

		//One Hot Encoding

		if ((std::string("sigmoid").compare(layer->p_act->name) == 0) ||
		        (std::string("softmax").compare(layer->p_act->name) == 0) ||
		        (std::string("logsoftmax").compare(layer->p_act->name) == 0)) {

			target = std::vector<float>(layer_node_size, 0);
		} else {

			target = std::vector<float>(layer_node_size, -1);
		}
		if (label_index >= 0 && label_index < layer_node_size) {
			target[label_index] = 1;
		}


        //const float grad_fudge = 1.0f;
		// because of numerator/demoninator cancellations which prevent a divide by zero issue, 
		// we need to handle some things special on output layer
		float cost_activation_type = 0;
		if ((std::string("sigmoid").compare(layer->p_act->name) == 0) &&
		        (std::string("cross_entropy").compare(_cost_function->name) == 0)) {

			cost_activation_type = 1;

		} else if ((std::string("softmax").compare(layer->p_act->name) == 0) &&
		           (std::string("cross_entropy").compare(_cost_function->name) == 0)) {
			cost_activation_type = 1;

		} else if ((std::string("tanh").compare(layer->p_act->name) == 0) &&
		           (std::string("cross_entropy").compare(_cost_function->name) == 0)) {

			cost_activation_type = 4;

		} else if ((std::string("logsoftmax").compare(layer->p_act->name) == 0) &&
		           (std::string("distillation").compare(_cost_function->name) == 0)) {
			cost_activation_type = -1;
		}
	
		// for (int j = 0; j < layer_node_size; j++)
		// {
		// 	if(cost_activation_type>0)
		// 		layer->delta.x[j] = cost_activation_type*(layer->node.x[j]- target[j]);
		// 	else
		// 		layer->delta.x[j] = _cost_function->d_cost(layer->node.x[j], target[j])*layer->df(layer->node.x, j, layer_node_size);

		// 	// pick best response
		// 	if (layer->node.x[max_j_out] < layer->node.x[j]) max_j_out = j;
		// 	// for better E maybe just look at 2 highest scores so zeros don't dominate 

		// 	float f= mse::cost(layer->node.x[j], target[j]);
		// 	E += f;//mse::cost(layer->node.x[j], target[j]);
		// }


		for (int j = 0; j < layer_node_size; j++) {

			if (cost_activation_type > 0) {
				//std::cout << "Node: " << j << " value: " << layer->node.x[j] << " corresponding target: " << target[j] << std::endl;
				layer->delta.x[j] = cost_activation_type * (layer->node.x[j] - target[j]);
				//layer->delta.x[j] = _cost_function->d_cost(layer->node.x[j], target[j],0,0) * layer->df(layer->node.x, j, layer_node_size);
				//std::cout << "Delta for node: " << j << " value: " << layer->delta.x[j] << std::endl;
			} else if (cost_activation_type < 0) {
				//Distillation loss
				//std::cout << "Node: " << j << " value: " << layer->node.x[j] << " corresponding target: " << target[j] <<" teacher_val: "<< teacher_layer->node.x[j] << std::endl;
				layer->delta.x[j] = _cost_function->d_cost(layer->node.x[j], target[j], (*teacher_prob)[j], TEMPERATURE) ; //* layer->df(layer->node.x, j, layer_node_size,TEMPERATURE);
				//std::cout << "Delta for node: " << j << " value: " << layer->delta.x[j] << std::endl;
			} else {
				//std::cout << "Node: " << j << " value: " << layer->node.x[j] << " corresponding target: " << target[j] << std::endl;
				layer->delta.x[j] = _cost_function->d_cost (layer->node.x[j], target[j], (*teacher_prob)[j], TEMPERATURE) * layer->df(layer->node.x, j, layer_node_size, TEMPERATURE);
				//std::cout << "Delta for node: " << j << " value: " << layer->delta.x[j] << std::endl;
			}

			//count_prob += layer->node.x[j];
			//Pick best response
			if (layer->node.x[max_j_out] < layer->node.x[j]) {
				max_j_out = j;
			}

			// for better E maybe just look at 2 highest scores so zeros don't dominate
			float f;
			//if(teacher_model != NULL)
			//	f = distillation::cost(layer->node.x[j], target[j],teacher_layer->node.x[j],TEMPERATURE);
			//else
			f = mse::cost(layer->node.x[j], target[j], 0, 0);

			E += f;//mse::cost(layer->node.x[j], target[j]);
		}



	
		E /= (float)layer_node_size;
		// check for NAN
		if (E != E) bail("network blew up - try lowering learning rate\n");
		
		// critical section in here, blocking update
		bool match = false;
		if (max_j_target == max_j_out) match = true;
		update_smart_train(E, match);

		if (E>0 && E<_skip_energy_level && _smart_train && match)
		{
//			lock_batch();
//			batch_open[my_batch_index] = BATCH_FREE;
//			unlock_batch();
			return false;  // return without doing training
		}

		if (teacher_prob != NULL) {
			backward_hidden(my_batch_index, thread_number, TEMPERATURE);
		} else {
			backward_hidden(my_batch_index, thread_number, 1);
		}


		//backward_hidden(my_batch_index, thread_number);
		return true;
	}
	
	//----------------------------------------------------------------------------------------------------------
	// T R A I N   T A R G E T 
	//
	// after starting epoch, call this to train against a target vector
	// for thread safety, you must pass in the thread_index if calling from different threads
	// if positive=1, goal is to minimize the distance between in and target
	bool train_target(float *in, float *target, int positive=1, int _thread_number = -1)
	{
		if (_solver == NULL) bail("set solver");
		if (_thread_number < 0) _thread_number = get_thread_num();
		if (_thread_number > _thread_count)  bail("need to enable OMP");

		const int thread_number = _thread_number;

		mojo::matrix augmented_input = make_input(in, thread_number);

		float *input = augmented_input.x;

		// get next free mini_batch slot
		// this is tied to the current state of the model
		int my_batch_index = reserve_next_batch();
		// out of data or an error if index is negative
		if (my_batch_index < 0) return false;
		// run through forward to get nodes activated
		float *out=forward(in, 1.f,thread_number, 1);

		// set all deltas to zero
		__for__(auto layer __in__ layer_sets[thread_number]) layer->delta.fill(0.f);

		int layer_cnt = (int)layer_sets[thread_number].size();

		// calc delta for last layer to prop back up through network
		// d = (target-out)* grad_activiation(out)
		const int last_layer_index = layer_cnt - 1;
		base_layer *layer = layer_sets[thread_number][last_layer_index];
		const int layer_node_size = layer->node.size();

		if (dynamic_cast<dropout_layer*> (layer) != NULL) bail("can't have dropout on last layer");

		float E = 0;
		int max_j_out = 0;
		//int max_j_target = label_index;

		// was passing this in, but may as well just create it on the fly
		// a vector mapping the label index to the desired target output node values
		// all -1 except target node 1
//		std::vector<float> target;
		//if ((std::string("sigmoid").compare(layer->p_act->name) == 0) || (std::string("softmax").compare(layer->p_act->name) == 0))
//			target = std::vector<float>(layer_node_size, 0);
//		else
//			target = std::vector<float>(layer_node_size, -1);
//		if (label_index >= 0 && label_index<layer_node_size) target[label_index] = 1;

		const float grad_fudge = 1.0f;
		// because of numerator/demoninator cancellations which prevent a divide by zero issue, 
		// we need to handle some things special on output layer
		float cost_activation_type = 0;
		if ((std::string("sigmoid").compare(layer->p_act->name) == 0) &&
			(std::string("cross_entropy").compare(_cost_function->name) == 0))
			cost_activation_type = 1;
		else if ((std::string("softmax").compare(layer->p_act->name) == 0) &&
			(std::string("cross_entropy").compare(_cost_function->name) == 0))
			cost_activation_type = 1;
		else if ((std::string("tanh").compare(layer->p_act->name) == 0) &&
			(std::string("cross_entropy").compare(_cost_function->name) == 0))
			cost_activation_type = 4;

		for (int j = 0; j < layer_node_size; j++) {
			if (positive) { // want to minimize distance
				if (cost_activation_type > 0)
					layer->delta.x[j] = grad_fudge * cost_activation_type * (layer->node.x[j] - target[j]);
				else
					layer->delta.x[j] = grad_fudge * _cost_function->d_cost(layer->node.x[j], target[j], 0, 0) * layer->df(layer->node.x, j, layer_node_size, TEMPERATURE);
			} else {
				if (cost_activation_type > 0)
					layer->delta.x[j] = grad_fudge * cost_activation_type * (1.f - abs(layer->node.x[j] - target[j]));
				else
					layer->delta.x[j] = grad_fudge * (1.f - abs(_cost_function->d_cost(layer->node.x[j], target[j], 0, 0))) * layer->df(layer->node.x, j, layer_node_size, TEMPERATURE);
			}
			// pick best response
			if (layer->node.x[max_j_out] < layer->node.x[j]) max_j_out = j;
			// for better E maybe just look at 2 highest scores so zeros don't dominate

			// L2 distance x 2
			E += mse::cost(layer->node.x[j], target[j], 0, 0);
		}

		E /= (float)layer_node_size;
		// check for NAN
		if (E != E) bail("network blew up - try lowering learning rate\n");

		// critical section in here, blocking update
		bool match = false;
// FIxME		if ((max_j_target == max_j_out)) match = true;
		if (E < 0.01 && positive) match = true;
		else if (E > 0.1 && !positive) match = true;
		update_smart_train(E, match);

		if (E>0 && E<_skip_energy_level && _smart_train && match)
		{
			lock_batch();
			batch_open[my_batch_index] = BATCH_FREE;
			unlock_batch();
			return false;  // return without doing training
		}
		//backward_hidden(my_batch_index, thread_number);   !!!CAREFUL HERE
		return true;
	}

#else

	float get_learning_rate() {return 0;}
	void set_learning_rate(float alpha) {}
	void train(float *in, float *target){}
	void reset() {}
	float get_smart_train_level() {return 0;}
	void set_smart_train_level(float _level) {}
	bool get_smart_train() { return false; }
	void set_smart_train(bool _use) {}

#endif

};

}
