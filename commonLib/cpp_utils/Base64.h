/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#ifndef _BASE64_H_
#define _BASE64_H_

#include <vector>
#include <string>

typedef unsigned char BYTE;

/* Class implementing encoding and decoding operations for int and float vectors
 * Precision is bounded by maxInt and max(input_vector)*/
class Base64
{
public:
    static std::string encode(std::vector<float> v);
    static std::string encode(std::vector<int> v);
    static std::vector<int> decodeInt(std::string encoded_string);
    static std::vector<float> decodeFloat(std::string encoded_string);
private:
    static int numDigits(int number);
    static std::vector<BYTE> decode(std::string encoded_string);
    static std::string encode(const std::vector<BYTE>& buf);
    static std::string encode(const BYTE* buf, unsigned int bufLen);
    static std::vector<int> float2int(std::vector<float> v);
    static std::vector<float> int2float(std::vector<int> v);
};
#endif

