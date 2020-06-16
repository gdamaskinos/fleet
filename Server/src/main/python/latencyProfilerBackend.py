"""
  Copyright (c) 2020 Georgios Damaskinos
  All rights reserved.
  @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
  This source code is licensed under the MIT license found in the
  LICENSE file in the root directory of this source tree.
"""


"""Template for python profiler communication with the server"""
import argparse
import socket
import numpy as np
import sys
import json
import string
import sklearn
import sklearn.linear_model
import pandas as pd
from threading import Lock

def openServerConn(port):
  """Waits for client to open connection and returns the connection"""
  sock = socket.socket()
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  sock.bind(("localhost", port))

  sock.listen(5)

  # Wait for a connection
  print('waiting for a connection...')
  connection, client_address = sock.accept()
  #print('connection from', client_address)

  return connection

def openClientConn(port, hostname):
  """Connects to server and returns socket"""
  while True:
    try:
      sock = socket.socket()
      sock.connect((hostname, port))
      return sock
    except Exception as e:
      print(type(e).__name__, e)
      print("Retrying to connect...")
      time.sleep(1)

def closeConnection(sock):
  #sock.shutdown(1)
  sock.close()

def sendMessage(sock, message="ACK"):
  """Sends a message to the connection (non-blocking)"""
  try:
    message = json.dumps(list(map(lambda x: str(x), message)))
#   message = json.dumps(["1","2","3","4"]).encode()
  except json.JSONEncodeError:
    print("Input message cannot be parsed into array")

  message += "\n"
  #print("SEND: ", message)
  message = message.encode()
  sock.sendall(message)

def loadTrainingDataset(trainDataset):
    df = pd.read_csv(trainDataset, index_col=None, header=0)
    assert(set(df.columns.values) == set(old_features))
    #pd.set_option('display.max_columns', 500)
    #pd.set_option('mode.chained_assignment', None)

    #convert everything to numeric except...
    cols = df.columns.drop(['clientID', 'afterPush', 'profilerEpoch', 'android_model', 'android_version', 'android_serialNumber'])
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    #df.replace([np.inf, -np.inf], np.nan)
    #print(df[df.isnull().any(axis=1)])

    #df.dropna(inplace=True)
    df = df[df['afterPush'] == 1]
    df = df[(df['deviceCpuUsage'] >= 0) & (df['deviceCpuUsage'] <= 1)]
    df = df[df['runningProcesses'] > 0]
    df['totalFreq'] =  df['cores'] * df['cpuMaxFreqMean']
    return df

def lm_fit(X,y):
    mod_lm = sklearn.linear_model.LinearRegression()
    mod_lm.fit(X, y)
    return mod_lm

def pa_fit(X, y, partial):
    mod_pa = sklearn.linear_model.PassiveAggressiveRegressor(C=0.1, epsilon=0.1, warm_start=False)
    if (partial == 1):
        mod_pa.partial_fit(X, y)
    else:
        mod_pa.fit(X, y)
    return mod_pa
    #mod_pa.fit(X_train.iloc[0].values.reshape(1,-1), y_train.iloc[0])

#features used for the old file already collected
old_features = ['clientID','afterPush','profilerEpoch','android_model','android_version',
                'android_serialNumber','availableMemory(MB)','runningProcesses','cores','threads',
                'bogoMips','networkLatency(ms)','sizeLatency(ms)','meanSizeLatency(ms)',
                'deviceLatency(ms)','heapSize(MB)','ramSize(MB)','bandwidth(Kbps)','batchSize',
                'sizeEnergy','deviceEnergy','meanSizeEnergy','deviceTotalRam','deviceAvailableRam',
                'deviceCpuUsage','temperature','batteryLevel','volt','cpuMaxFrequency[0]',
                'cpuMaxFrequency[1]','cpuMaxFrequency[2]','cpuMaxFrequency[3]','cpuMaxFrequency[4]',
                'cpuMaxFrequency[5]','cpuMaxFrequency[6]','cpuMaxFrequency[7]','cpuCurFrequency[0]',
                'cpuCurFrequency[1]','cpuCurFrequency[2]','cpuCurFrequency[3]','cpuCurFrequency[4]',
                'cpuCurFrequency[5]','cpuCurFrequency[6]','cpuCurFrequency[7]','cpuMaxFreqMean']

new_features = ['clientID','afterPush','profilerEpoch','android_model','android_version',
				'android_serialNumber','availableMemory(MB)','runningProcesses','cores',
				'littleThreads','bigThreads','bogoMips','networkLatency(ms)','sizeLatency(ms)',
				'meanSizeLatency(ms)','deviceLatency(ms)','heapSize(MB)','ramSize(MB)','bandwidth(Kbps)',
				'batchSize','sizeEnergy(mAh)','deviceEnergy(mAh)','idleEnergy(mAh)','deviceTotalRam',
				'deviceAvailableRam','deviceCpuUsage','temperature','batteryLevel','volt(mV)',
				'cpuMaxFrequency[0]','cpuMaxFrequency[1]','cpuMaxFrequency[2]','cpuMaxFrequency[3]',
				'cpuMaxFrequency[4]','cpuMaxFrequency[5]','cpuMaxFrequency[6]','cpuMaxFrequency[7]',
				'cpuCurFrequency[0]','cpuCurFrequency[1]','cpuCurFrequency[2]','cpuCurFrequency[3]',
				'cpuCurFrequency[4]','cpuCurFrequency[5]','cpuCurFrequency[6]','cpuCurFrequency[7]',
				'cpuMaxFreqMean']

ml_features = ['deviceTotalRam','deviceAvailableRam','temperature','totalFreq']

def build_train_set(df):
    keep_features = ['android_model', 'runningProcesses','cores','sizeLatency(ms)','batchSize','deviceTotalRam','deviceAvailableRam','deviceCpuUsage','temperature','cpuMaxFreqMean','totalFreq']
    train_set = df.loc[:, keep_features]
    train_set.dropna(inplace=True)
    train_set.reset_index(drop=True, inplace=True)
    train_phones = train_set.android_model.unique()

    X_train = pd.DataFrame()
    y_train = pd.DataFrame()
    train_set_phone_lats_0 = []
    train_set_phone_batchs_0 = []
    for phone in train_phones:
       train_set_phone = train_set[train_set['android_model'] == phone]
       train_set_phone.reset_index(drop=True, inplace=True)
       train_set_phone_1 = train_set_phone.ix[1:]

       train_set_phone_lat_0 = train_set_phone['sizeLatency(ms)'].iloc[0]
       train_set_phone_batch_0 = train_set_phone['batchSize'].iloc[0]
       if (train_set_phone_batch_0 < 10):
           train_set_phone_lats_0.append(train_set_phone_lat_0)
           train_set_phone_batchs_0.append(train_set_phone_batch_0)
       train_set_phone_lat_0_avg = sum(train_set_phone_lats_0) / len(train_set_phone_lats_0)
       train_set_phone_batch_0_avg = sum(train_set_phone_batchs_0) / len(train_set_phone_batchs_0)

       y_train_phone = (train_set_phone_1['sizeLatency(ms)'] - train_set_phone_lat_0_avg) / (train_set_phone_1['batchSize'] - train_set_phone_batch_0_avg)
       y_train_phone = pd.DataFrame(y_train_phone.values, columns=['slope'])
       y_train = y_train.append(y_train_phone, ignore_index=True)

       X_train = X_train.append(train_set_phone_1.loc[:, ml_features], ignore_index=True)
       #print(X_train, y_train)

    return (X_train, y_train)

def train(df, pa):
    featuresPlusLabel = build_train_set(df)
    #print(featuresPlusLabel[1])
    if (pa == 1):
        mod = pa_fit(*featuresPlusLabel, 0)
    else:
        mod = lm_fit(*featuresPlusLabel)
    return mod

def pa_retrain(df, phone):
    last_entry = df[df['android_model'] == phone].tail(1)
    #print("Cores and frequency: ", last_entry['cores'], last_entry['cpuMaxFreqMean'])
    last_entry['totalFreq'] = computeTotalFreq(last_entry)
    #last_entry['totalFreq'] =  last_entry['cores'] * last_entry['cpuMaxFreqMean']
    X = last_entry.loc[:, ml_features]

    #assume that the the first batchSize == number of threads and its computation latency is 100ms
    y = (last_entry['sizeLatency(ms)'] - 100) / last_entry['batchSize']

    print("Learning features:")
    print(X, y)
    mod = pa_fit(X, y, 1)
    return mod

def computeTotalFreq(pred_df):
    nb_cores = float(pred_df['cores'].values[0])

    littleCores = pred_df.loc[:, ['cpuMaxFrequency[0]','cpuMaxFrequency[1]','cpuMaxFrequency[2]',
        'cpuMaxFrequency[3]']].values[0]
    littleCores = littleCores.astype(np.float)
    bigCores = pred_df.loc[:, ['cpuMaxFrequency[4]','cpuMaxFrequency[5]','cpuMaxFrequency[6]',
        'cpuMaxFrequency[7]']].values[0]
    bigCores = bigCores.astype(np.float)

    if(nb_cores <= 4):
        meanFreq = max(littleCores)
    elif(nb_cores == 8):
        meanFreq = (max(littleCores) + max(bigCores)) / 2
    else:
        raise ValueError('The number of CPUs is larger than 8.')
    return (meanFreq * nb_cores)

def makePrediction(mod_lm, pred_df):
    pred_df['totalFreq'] = computeTotalFreq(pred_df)

    print("Prediction features:")
    print(pred_df.loc[:, ml_features])

    pred_slope = mod_lm.predict(pred_df.loc[:, ml_features])
    batch_size = int((10000 - 100) / pred_slope[0]);

    print("Predicted slope: ", pred_slope[0])
    print("Predicted batchSize for " + stats[3] + " : ", batch_size)

    if (batch_size > 10000):
        return 10000
    elif (batch_size < 104):
        return 104
    else:
        return batch_size + 8 - (batch_size % 8);

def storeStats(df, learn_df):
    learn_df['totalFreq'] = computeTotalFreq(learn_df)
    df = df.append(learn_df)
    #convert everything to numeric except...
    cols = df.columns.drop(['clientID', 'afterPush', 'profilerEpoch', 'android_model',
                            'android_version', 'android_serialNumber'])
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    return df

def getMessage(connection):
  """Blocks until receiving a message from the connection
  Returns:
    (np.array): message or None if exception
  """

  result = None
  try:
    data = b''
    BUFF_SIZE = 1024
    while True:
      part = connection.recv(BUFF_SIZE)
      data += part
      #print("NEXT", len(part), part.decode()[-1])
      if part.decode()[-1] == "\n":
       #if len(part) == 0: # < BUFF_SIZE:
        break

    result = data.decode()[:-1].encode()

    try:
      result = np.array(json.loads(data))
      #print("GOT (size=%d): %s" % (len(result), result))
    except json.JSONDecodeError:
      print("Input message cannot be parsed into array")
      result = result.decode()
      print(result)

  except Exception as e:
      print(type(e).__name__, e)
  finally:
      return result

if __name__ == "__main__":

  parser = argparse.ArgumentParser()
  parser.add_argument("--clientPort", type=int, default=9995,
      help='Port for communicating with the client; used only by the servers')

  parser.add_argument("--trainDataset",
      help='Path to the pretraining dataset')

  args = sys.argv[1:]
  args = parser.parse_args(args)

  df = loadTrainingDataset(args.trainDataset)
  global_mod = train(df, pa=0)
  per_device_models = {}

  mutex = Lock()

  conn = openServerConn(port=args.clientPort)
  while True:
   with mutex:
    stats = getMessage(conn)
    temp_df = pd.DataFrame([stats], columns=new_features)
    #prediction or learn stats?
    if (temp_df['afterPush'].values[0] == "0"):
        if temp_df['android_model'].values[0] in per_device_models:
            batchSize = makePrediction(per_device_models[temp_df['android_model'].values[0]], temp_df)
            sendMessage(conn, [batchSize])
        else:
            batchSize = makePrediction(global_mod, temp_df)
            print("new device prediction ", batchSize)
            sendMessage(conn, [batchSize])
    else:
        df = storeStats(df, temp_df)
        per_device_models[temp_df['android_model'].values[0]] = pa_retrain(df, temp_df['android_model'].values[0])
        global_mod = train(df, pa=0)
        #print("update from " + temp_df['android_model'].values[0] + ", " + temp_df['sizeLatency(ms)'].values[0])
  closeConnection(conn)
