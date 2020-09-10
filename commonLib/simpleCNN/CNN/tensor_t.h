/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#pragma once

#include "point_t.h"
#include <cassert>
#include <vector>
#include <string.h>

template<typename T>
struct tensor_t {
    T *data;

    tdsize size;

    tensor_t(int _x, int _y, int _z) {
        data = new T[_x * _y * _z];
        size.x = _x;
        size.y = _y;
        size.z = _z;
    }

    tensor_t(const tensor_t &other) {
        data = new T[other.size.x * other.size.y * other.size.z];
        memcpy(
                this->data,
                other.data,
                other.size.x * other.size.y * other.size.z * sizeof(T)
        );
        this->size = other.size;
    }

    tensor_t<T> operator+(tensor_t<T> &other) {
        tensor_t<T> clone(*this);
        for (int i = 0; i < other.size.x * other.size.y * other.size.z; i++)
            clone.data[i] += other.data[i];
        return clone;
    }

    tensor_t<T> operator+=(tensor_t<T> &other) {
        for (int i = 0; i < other.size.x * other.size.y * other.size.z; i++)
            this->data[i] += other.data[i];
        return *this;
    }

    void add(tensor_t<T> &other) {
        for (int i = 0; i < this->size.x * this->size.y * this->size.z; i++)
            this->data[i] += other.data[i];
    }

    tensor_t<T> operator-(tensor_t<T> &other) {
        tensor_t<T> clone(*this);
        for (int i = 0; i < other.size.x * other.size.y * other.size.z; i++)
            clone.data[i] -= other.data[i];
        return clone;
    }

    tensor_t<T> operator/(float operand) {
        tensor_t<T> clone(*this);
        for (int i = 0; i < clone.size.x * clone.size.y * clone.size.z; i++)
            clone.data[i] /= operand;
        return clone;
    }

    T &operator()(int _x, int _y, int _z) {
        return this->get(_x, _y, _z);
    }

    T &get(int _x, int _y, int _z) {
        assert(_x >= 0 && _y >= 0 && _z >= 0);
        assert(_x < size.x && _y < size.y && _z < size.z);

        return data[
                _z * (size.x * size.y) +
                _y * (size.x) +
                _x
        ];
    }
    
    void copy_from(std::vector<std::vector<std::vector<T>>> data) {
        int z = data.size();
        int y = data[0].size();
        int x = data[0][0].size();

        for (int i = 0; i < x; i++)
            for (int j = 0; j < y; j++)
                for (int k = 0; k < z; k++)
                    get(i, j, k) = data[k][j][i];
    }
   
	std::vector<T> flatData() {
		std::vector<T> ret;
		ret.assign(data, data + this->getSize());
		return ret;
	}

	int getSize() {
		return size.x*size.y*size.z;
	}

    ~tensor_t() {
        delete[] data;
    }

	
};

static void print_tensor(tensor_t<float> &data, int limx, int limy, int limz) {
    int mx = std::min(limx, data.size.x);
    int my = std::min(limy, data.size.y);
    int mz = std::min(limz, data.size.z);

    for (int z = 0; z < mz; z++) {
        printf("[Dim%d]\n", z);
        for (int y = 0; y < my; y++) {
            for (int x = 0; x < mx; x++) {
                printf("%.5f \t", (float) data.get(x, y, z));
            }
            printf("\n");
        }
    }
}

static void print_tensor(tensor_t<float> &data) {
    int mx = data.size.x;
    int my = data.size.y;
    int mz = data.size.z;

    for (int z = 0; z < mz; z++) {
        printf("[Dim%d]\n", z);
        for (int y = 0; y < my; y++) {
            for (int x = 0; x < mx; x++) {
                printf("%.5f \t", (float) data.get(x, y, z));
            }
            printf("\n");
        }
    }
}

static tensor_t<float> to_tensor(std::vector<std::vector<std::vector<float>>> data) {
    int z = data.size();
    int y = data[0].size();
    int x = data[0][0].size();


    tensor_t<float> t(x, y, z);

    for (int i = 0; i < x; i++)
        for (int j = 0; j < y; j++)
            for (int k = 0; k < z; k++)
                t(i, j, k) = data[k][j][i];
    return t;
}
