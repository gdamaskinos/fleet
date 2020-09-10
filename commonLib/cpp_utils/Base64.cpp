/*
 * Copyright (c) 2020 Georgios Damaskinos
 * All rights reserved.
 * @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */


#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <jni.h>
#include <vector>
#include <iostream>
#include <sstream>
#include "Base64.h"
#include <math.h>

static const BYTE from_base64[] = { 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                                    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
                                    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,  62, 255,  62, 255,  63,
                                    52,  53,  54,  55,  56,  57,  58,  59,  60,  61, 255, 255, 255, 255, 255, 255,
                                    255,   0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,
                                    15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25, 255, 255, 255, 255,  63,
                                    255,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,
                                    41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51, 255, 255, 255, 255, 255};

static const char to_base64[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                "abcdefghijklmnopqrstuvwxyz"
                "0123456789+/";

static const int precision = 9; // maxInt digits = 10
static const int intNum = 1; // 1 or 2

int Base64::numDigits(int number)
{
    int digits = 0;
    if (number < 0) digits = 1; // remove this line if '-' counts as a digit
    while (number) {
        number /= 10;
        digits++;
    }
    return digits;
}

std::vector<int> Base64::float2int(std::vector<float> v) {

    // TODO precision depends on maxInt and max(v)

    std::vector<int> ret;
    for (int i=0; i<v.size(); i++) {
		if (intNum == 2) {
	    	double fractpart, intpart;
		    fractpart = modf(v[i], &intpart);
	        ret.push_back((int) intpart);
	        ret.push_back(static_cast<int>(floor(fractpart * pow(10, precision) + 0.5)));
		}
		else if (intNum == 1) {
			int digits = Base64::numDigits((int) v[i]);
			for (int j=0; j<precision-digits; j++)
				v[i] *= 10;
			int temp = (int) v[i];
			// replace last digit with numDigits
			int LSB = temp % 10;
			if (temp >= 0)
				temp = temp - LSB + digits;
			else
				temp = temp - LSB - digits;

			ret.push_back(temp);
		}

    }

    return ret;
}

std::vector<float> Base64::int2float(std::vector<int> v) {

    std::vector<float> ret;
    double fractpart, intpart;
    for (int i=0; i<v.size(); i++) {
		if (intNum == 2) {
	        intpart = v[i];
	        fractpart = v[i+1] / pow(10, precision);
	        i++;
	        ret.push_back(intpart + fractpart);
		}
		else if (intNum == 1) {
			int decodeDigits = abs(v[i] % 10); // get the numDigits
			float temp = (float) v[i];
			for (int j=0; j<precision-decodeDigits; j++) {
				temp /= 10;
			}
			ret.push_back(temp);
		}

    }

    return ret;
}
std::string Base64::encode(std::vector<float> v) {
    return Base64::encode(Base64::float2int(v));
}


std::string Base64::encode(std::vector<int> v) {
    const char* bytes = reinterpret_cast<const char*>(&v[0]);

    std::vector<unsigned char> byteVec(bytes, bytes + sizeof(int) * v.size());

    return Base64::encode(&byteVec[0], byteVec.size());
}

std::string Base64::encode(const std::vector<BYTE>& buf)
{
    if (buf.empty())
        return ""; // Avoid dereferencing buf if it's empty
    return encode(&buf[0], (unsigned int)buf.size());
}

std::string Base64::encode(const BYTE* buf, unsigned int bufLen)
{
    // Calculate how many bytes that needs to be added to get a multiple of 3
    size_t missing = 0;
    size_t ret_size = bufLen;
    while ((ret_size % 3) != 0)
    {
        ++ret_size;
        ++missing;
    }

    // Expand the return string size to a multiple of 4
    ret_size = 4*ret_size/3;

    std::string ret;
    ret.reserve(ret_size);

    for (unsigned int i=0; i<ret_size/4; ++i)
    {
        // Read a group of three bytes (avoid buffer overrun by replacing with 0)
        size_t index = i*3;
        BYTE b3[3];
        b3[0] = (index+0 < bufLen) ? buf[index+0] : 0;
        b3[1] = (index+1 < bufLen) ? buf[index+1] : 0;
        b3[2] = (index+2 < bufLen) ? buf[index+2] : 0;

        // Transform into four base 64 characters
        BYTE b4[4];
        b4[0] =                         ((b3[0] & 0xfc) >> 2);
        b4[1] = ((b3[0] & 0x03) << 4) + ((b3[1] & 0xf0) >> 4);
        b4[2] = ((b3[1] & 0x0f) << 2) + ((b3[2] & 0xc0) >> 6);
        b4[3] = ((b3[2] & 0x3f) << 0);

        // Add the base 64 characters to the return value
        ret.push_back(to_base64[b4[0]]);
        ret.push_back(to_base64[b4[1]]);
        ret.push_back(to_base64[b4[2]]);
        ret.push_back(to_base64[b4[3]]);
    }

    // Replace data that is invalid (always as many as there are missing bytes)
    for (size_t i=0; i<missing; ++i)
        ret[ret_size - i - 1] = '=';

    return ret;
}

std::vector<float> Base64::decodeFloat(std::string encoded_string) {
    return Base64::int2float(Base64::decodeInt(encoded_string));
}

std::vector<int> Base64::decodeInt(std::string encoded_string) {

    std::vector<unsigned char> decodedData = Base64::decode(encoded_string);
    unsigned char* bytes = &(decodedData[0]);    // point to beginning of memory

    int* floatArray = reinterpret_cast<int*>(bytes);
    std::vector<int> ret(floatArray, floatArray + decodedData.size() / sizeof(int));
    return ret;
}

std::vector<BYTE> Base64::decode(std::string encoded_string)
{
    // Make sure string length is a multiple of 4
    while ((encoded_string.size() % 4) != 0)
        encoded_string.push_back('=');

    size_t encoded_size = encoded_string.size();
    std::vector<BYTE> ret;
    ret.reserve(3*encoded_size/4);

    for (size_t i=0; i<encoded_size; i += 4)
    {
        // Get values for each group of four base 64 characters
        BYTE b4[4];
        b4[0] = (encoded_string[i+0] <= 'z') ? from_base64[encoded_string[i+0]] : 0xff;
        b4[1] = (encoded_string[i+1] <= 'z') ? from_base64[encoded_string[i+1]] : 0xff;
        b4[2] = (encoded_string[i+2] <= 'z') ? from_base64[encoded_string[i+2]] : 0xff;
        b4[3] = (encoded_string[i+3] <= 'z') ? from_base64[encoded_string[i+3]] : 0xff;

        // Transform into a group of three bytes
        BYTE b3[3];
        b3[0] = ((b4[0] & 0x3f) << 2) + ((b4[1] & 0x30) >> 4);
        b3[1] = ((b4[1] & 0x0f) << 4) + ((b4[2] & 0x3c) >> 2);
        b3[2] = ((b4[2] & 0x03) << 6) + ((b4[3] & 0x3f) >> 0);

        // Add the byte to the return value if it isn't part of an '=' character (indicated by 0xff)
        if (b4[1] != 0xff) ret.push_back(b3[0]);
        if (b4[2] != 0xff) ret.push_back(b3[1]);
        if (b4[3] != 0xff) ret.push_back(b3[2]);
    }

    return ret;
}
