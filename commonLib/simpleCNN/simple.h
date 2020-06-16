/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <sstream>
#include <time.h>
#include "CNN/cnn.h"

using namespace std;

void forward(vector<layer_t *> &layers, tensor_t<float> &data) {
    activate(layers[0], data);
    for (int i = 1; i < layers.size(); i++) {
        activate(layers[i], layers[i - 1]->out);
    }
}


float train_SGD_step(vector<layer_t *> &layers, tensor_t<float> &data, tensor_t<float> &expected) {
    forward(layers, data);

    tensor_t<float> grads = layers.back()->out - expected;

    for (int i = layers.size() - 1; i >= 0; i--) {
        if (i == layers.size() - 1)
            calc_grads(layers[i], grads);
        else
            calc_grads(layers[i], layers[i + 1]->grads_in);
    }

    for (int i = 0; i < layers.size(); i++) {
        fix_weights(layers[i]);
    }

    float err = 0;
    for (int i = 0; i < grads.size.x * grads.size.y * grads.size.z; i++) {
        float f = expected.data[i];
        if (f > 0.5)
            err += abs(grads.data[i]);
    }
    return err * 100;
}


vector<float> grad_to_floats(vector<layer_t *> &layers) {
    vector<float> floats;
    for (int i = 0; i < layers.size(); i++) {
        auto layer_floats = grads_to_floats(layers[i]);
        floats.insert(floats.end(), layer_floats.begin(), layer_floats.end());
    }

    return floats;
}

void floats_to_grad(vector<layer_t *> &layers, vector<float> &floats) {
    for (int i = 0; i < layers.size(); i++) {
        floats_to_grads(layers[i], floats);
    }
}

void train_SGD_minibatch(vector<layer_t *> &layers, vector<tensor_t<float>> &x,
                         vector<tensor_t<float>> &y, int batch_size) {

    vector<tensor_t<float>> x_mini_batch;
    vector<tensor_t<float>> y_mini_batch;

	// Select a random mini batch
    for (int j = 0; j < batch_size; ++j) {
        int max = x.size() - 1;
        int min = 0;
        int index = min + (rand() % (max - min + 1));

        x_mini_batch.push_back(x[index]);
        y_mini_batch.push_back(y[index]);
    }


    init_acc_grads(layers[0]);
    for (int i = 1; i < layers.size(); i++) {
        init_acc_grads(layers[i]);
    }

    // Accumulating the gradient
    for (int i = 0; i < x_mini_batch.size(); i++) {
        auto xi = x_mini_batch[i];
        auto yi = y_mini_batch[i];

        forward(layers, xi);

        auto grads = (layers.back()->out - yi);

        for (int j = layers.size() - 1; j >= 0; j--) {
            if (j == layers.size() - 1) {
                calc_grads(layers[j], grads);
            } else {
                calc_grads(layers[j], layers[j + 1]->grads_in);
            }
        }
    }

    vector<float> floats = grad_to_floats(layers);
//    cout << floats.size() << endl;
    floats_to_grad(layers, floats);

    // Averaging the gradient, and update weights
    for (int i = 0; i < layers.size(); i++) {
        fix_weights(layers[i]);
    }
}


uint8_t *read_file(const char *szFile) {
    ifstream file(szFile, ios::binary | ios::ate);
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);

    if (size == -1)
        return nullptr;

    uint8_t *buffer = new uint8_t[size];
    file.read((char *) buffer, size);
    return buffer;
}

vector<vector<float>> load_csv(const char *csv_path) {
    vector<vector<float>> values;
    vector<float> valueline;
    ifstream fin(csv_path);
    if (fin.fail()) {
        cout << csv_path << " not found !" << endl;
        exit(1);
    }
    string item;
    for (string line; getline(fin, line);) {
        istringstream in(line);

        while (getline(in, item, ',')) {
            valueline.push_back(atof(item.c_str()));
        }

        values.push_back(valueline);
        valueline.clear();
    }

    cout << "Shape: (" << values.size() << "," << values[0].size() << ")" << endl;

    return values;
}


vector<tensor_t<float>>
csv_to_tensor(vector<vector<float>> &csv_y, const int size_x, const int size_y, const int size_z) {
    vector<tensor_t<float>> tensors_y;

    for (auto &yi : csv_y) {
        tensor_t<float> tensor_y(size_x, size_y, size_z);

        int i = 0;
        int j = 0;
        int k = 0;
        for (auto &yi_col : yi) {
            tensor_y(i, j, k) = yi_col;
            i++;
            if (i % size_x == 0) {
                i = 0;
                j++;

                if (j % size_y == 0) {
                    j = 0;
                    k++;
                }
            }
        }

        tensors_y.push_back(tensor_y);
    }

    return tensors_y;
}

vector<tensor_t<float>> load_csv_data(const char *csv_path, const int size_x, const int size_y, const int size_z) {
    auto csv_x = load_csv(csv_path);
    return csv_to_tensor(csv_x, size_x, size_y, size_z);
}

float compute_accuracy(vector<layer_t *> &layers, vector<tensor_t<float>> &x, vector<tensor_t<float>> &y) {

    float correct_count = 0;

    for (int i = 0; i < x.size(); ++i) {

        auto xi = x[i];
        auto yi = y[i];

        float expected = -1;
        for (int i = 0; i < yi.size.x; i++) {
            if (yi(i, 0, 0) == 1) {
                expected = i;
            }
        }

        forward(layers, xi);
        auto probs = layers.back()->out;
        float predicted = -1;
        float max_prob = -1;
        for (int i = 0; i < probs.size.x; i++) {
            if (probs(i, 0, 0) > max_prob) {
                max_prob = probs(i, 0, 0);
                predicted = i;
            }
        }

        if (predicted == expected) {
            correct_count++;
        }
    }

    return correct_count / x.size();
}

float compute_mae_loss(vector<layer_t *> &layers, vector<tensor_t<float>> &x, vector<tensor_t<float>> &y) {
    float sum = 0;

    for (int i = 0; i < x.size(); ++i) {

        auto xi = x[i];
        auto yi = y[i];

        forward(layers, xi);
        auto predicted_yi = layers.back()->out;

        auto diff = yi - predicted_yi;

        for (int i = 0; i < diff.size.x; i++) {
            sum += abs(diff(i, 0, 0));
        }
    }

    return sum / x.size();
}

int main() {
    srand(1);
	cout << "Learning rate: " << LEARNING_RATE << endl;
	cout << "Momentum: " << MOMENTUM << endl;
    cout << "Loading training set" << endl;
    auto train_x = load_csv_data("datasets/mnist/mnist_training_features.csv", 28, 28, 1);
    auto train_y = load_csv_data("datasets/mnist/mnist_training_labels.csv", 10, 1, 1);

    cout << "Loading validation set" << endl;
    auto val_x = load_csv_data("datasets/mnist/mnist_validation_features.csv", 28, 28, 1);
    auto val_y = load_csv_data("datasets/mnist/mnist_validation_labels.csv", 10, 1, 1);

    vector<layer_t *> layers;

    conv_layer_t *layer1 = new conv_layer_t(1, 5, 8, train_x[0].size);        // 28 * 28 * 1 -> 24 * 24 * 8
    relu_layer_t *layer2 = new relu_layer_t(layer1->out.size);
    pool_layer_t *layer3 = new pool_layer_t(2, 2, layer2->out.size);                // 24 * 24 * 8 -> 12 * 12 * 8
    fc_layer_t *layer4 = new fc_layer_t(layer3->out.size, 10);                    // 4 * 4 * 16 -> 10

    layers.push_back((layer_t *) layer1);
    layers.push_back((layer_t *) layer2);
    layers.push_back((layer_t *) layer3);
    layers.push_back((layer_t *) layer4);

    int ic = 0;

    clock_t begin = clock();
	vector<layer_t *> layers2;
	conv_layer_t *temp1;
	relu_layer_t *temp2;
	pool_layer_t *temp3;
	fc_layer_t *temp4;
	float acc;
    for (long ep = 0; ep < 100000;) {

        train_SGD_minibatch(layers, train_x, train_y, 1);
//        train_SGD_step(layers, x_mini_batch[0], y_mini_batch[0]);

        ep++;
        ic++;

        if (ep % 1000 == 0) {
            acc = compute_accuracy(layers, val_x, val_y);
			cout << acc << endl;	
			vector<float> netParams;
			vector<float> tempP1 = layer1->flatParams();
			vector<float> tempP2 = layer4->flatParams();
			netParams.insert(netParams.end(), tempP1.begin(), tempP1.end());
			netParams.insert(netParams.end(), tempP2.begin(), tempP2.end());

			temp1 = new conv_layer_t(1, 5, 8, train_x[0].size);
			int s1 = temp1->setParams(netParams);
   			netParams.erase(netParams.begin(), netParams.begin() + s1); 

			temp2 = new relu_layer_t(layer1->out.size);
		    temp3 = new pool_layer_t(2, 2, layer2->out.size);
			temp4 = new fc_layer_t(layer3->out.size, 10);
			temp4->setParams(netParams);

			free(layer1);
			free(layer2);
			free(layer3);
			free(layer4);
			layers2.push_back((layer_t *) temp1);
    		layers2.push_back((layer_t *) temp2);
    		layers2.push_back((layer_t *) temp3);
    		layers2.push_back((layer_t *) temp4);
            acc = compute_accuracy(layers2, val_x, val_y);
			layers2.clear();
			cout << acc << endl;	
			exit(0);

            auto val_acc = compute_accuracy(layers, val_x, val_y);
            auto val_loss = compute_mae_loss(layers, val_x, val_y);
            clock_t end = clock();
            double elapsed_time = double(end - begin) / CLOCKS_PER_SEC * 1000;
            cout << "eval:" << ep << "," << ep << ",0,0," << val_loss << "," << val_acc << "," << elapsed_time << endl;
        }
    }


#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-noreturn"
    while (true) {
        uint8_t *data = read_file("test.ppm");

        if (data) {
            uint8_t *usable = data;

            while (*(uint32_t *) usable != 0x0A353532)
                usable++;

#pragma pack(push, 1)
            struct RGB {
                uint8_t r, g, b;
            };
#pragma pack(pop)

            RGB *rgb = (RGB *) usable;

            tensor_t<float> image(28, 28, 1);
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    RGB rgb_ij = rgb[i * 28 + j];
                    image(j, i, 0) = (((float) rgb_ij.r
                                       + rgb_ij.g
                                       + rgb_ij.b)
                                      / (3.0f * 255.f));
                }
            }

            forward(layers, image);
            tensor_t<float> &out = layers.back()->out;
            for (int i = 0; i < 10; i++) {
                printf("[%i] %f\n", i, out(i, 0, 0) * 100.0f);
            }

            delete[] data;
        }

        struct timespec wait;
        wait.tv_sec = 1;
        wait.tv_nsec = 0;
        nanosleep(&wait, nullptr);
    }
#pragma clang diagnostic pop
    return 0;
}
