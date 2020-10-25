"""
  Copyright (c) 2020 Georgios Damaskinos
  All rights reserved.
  @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
  This source code is licensed under the MIT license found in the
  LICENSE file in the root directory of this source tree.
"""

import numpy as np
import scipy.io
import argparse
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def oneHot(vector):
  zeros = np.zeros(len(np.unique(vector)), dtype=int)
  out = []
  for x in vector:
    temp = list(zeros)
    temp[int(x)] = 1
    out += [temp]

  return np.array(out)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='EMNIST dataset parser to csv files')

  parser.add_argument("input",help="/path/to/emnist-byclass.mat that can be downloaded from http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip")
  parser.add_argument("output_prefix",help="path prefix for the output .csv files")
  parser.add_argument("--validation_ratio",
                      type=float,
                      default=0.2,
                      help="len(validation_set) = validation_ratio * len(training_set)")

  args = parser.parse_args()

  ## load and convert to numpy
  mat = scipy.io.loadmat(args.input)

  trainX = mat['dataset']['train'][0][0]['images'][0][0]
  trainY = mat['dataset']['train'][0][0]['labels'][0][0]

  testX = mat['dataset']['test'][0][0]['images'][0][0]
  testY = mat['dataset']['test'][0][0]['labels'][0][0]

  # convert labels into one-hot vectors
  trainY = oneHot(trainY)
  testY = oneHot(testY)

  ## preprocessing
  trainX = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(trainX)
  testX = preprocessing.MinMaxScaler(feature_range=(0,1)).fit_transform(testX)

  # validation split
  trainX, valX, trainY, valY = train_test_split(trainX,
                                                trainY,
                                                train_size=1-args.validation_ratio,
                                                random_state=42,
                                                shuffle=True)

  ## save to csv files
  np.savetxt(args.output_prefix + 'training_features.csv', trainX, fmt='%.10f', delimiter=",")
  np.savetxt(args.output_prefix + 'validation_features.csv', valX, fmt='%.10f', delimiter=",")
  np.savetxt(args.output_prefix + 'test_features.csv', testX, fmt='%.10f', delimiter=",")

  np.savetxt(args.output_prefix + 'training_labels.csv', trainY, fmt='%d', delimiter=",")
  np.savetxt(args.output_prefix + 'validation_labels.csv', valY, fmt='%d', delimiter=",")
  np.savetxt(args.output_prefix + 'test_labels.csv', testY, fmt='%d', delimiter=",")

