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
  #print('waiting for a connection...')
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
    pd.set_option('display.max_columns', 500)
    pd.set_option('mode.chained_assignment', None)

    #convert everything to numeric except...
    cols = df.columns.drop(['clientID', 'afterPush', 'profilerEpoch', 'android_model', 'android_version', 'android_serialNumber'])
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

    return df

def lm_fit(X,y):
    mod_lm = sklearn.linear_model.LinearRegression()
    mod_lm.fit(X, y)
    return mod_lm

def pa_fit(X, y, partial):
    mod_pa = sklearn.linear_model.PassiveAggressiveRegressor(C=0.1, epsilon=0.00006, warm_start=False)
    if (partial == 1):
        mod_pa.partial_fit(X, y)
    else:
        mod_pa.fit(X, y)
    return mod_pa
    #mod_pa.fit(X_train.iloc[0].values.reshape(1,-1), y_train.iloc[0])

all_features = ['clientID','afterPush','profilerEpoch','android_model','android_version',
                'android_serialNumber','availableMemory(MB)','runningProcesses','cores','threads',
                'bogoMips','networkLatency(ms)','sizeLatency(ms)','meanSizeLatency(ms)',
                'deviceLatency(ms)','heapSize(MB)','ramSize(MB)','bandwidth(Kbps)','batchSize',
                'sizeEnergy(mAh)','deviceEnergy','idleEnergy(mAh)','deviceTotalRam','deviceAvailableRam',
                'deviceCpuUsage','temperature','batteryLevel','volt(mV)','cpuMaxFrequency[0]',
                'cpuMaxFrequency[1]','cpuMaxFrequency[2]','cpuMaxFrequency[3]','cpuMaxFrequency[4]',
                'cpuMaxFrequency[5]','cpuMaxFrequency[6]','cpuMaxFrequency[7]','cpuCurFrequency[0]',
                'cpuCurFrequency[1]','cpuCurFrequency[2]','cpuCurFrequency[3]','cpuCurFrequency[4]',
                'cpuCurFrequency[5]','cpuCurFrequency[6]','cpuCurFrequency[7]','cpuMaxFreqMean']
ml_features = ['deviceTotalRam','deviceAvailableRam','temperature','totalFreq', 'energPerJiffies']
keep_features = ['android_model', 'afterPush', 'cores', 'sizeEnergy(mAh)', 'idleEnergy(mAh)',
                  'volt(mV)', 'batchSize','deviceTotalRam','deviceAvailableRam','deviceCpuUsage',
                  'temperature','cpuMaxFreqMean']

def build_train_set(df):
    train_set = df.loc[:, keep_features]
    train_set.dropna(inplace=True)
    train_set.reset_index(drop=True, inplace=True)
    #train_phones = train_set.android_model.unique()
    #train_phones = ['Galaxy S7', 'COL-L29', 'Xperia E3', 'Galaxy S4 Mini','STF-L09-4threads', 'STF-L09']
    train_phones = ['COL-L29', 'Xperia E3', 'Galaxy S4 Mini']

    X_train = pd.DataFrame()
    y_train = pd.DataFrame()
    for phone in train_phones:
       train_set_phone = train_set[train_set['android_model'] == phone].sort_values('batchSize')
       train_set_phone.reset_index(drop=True, inplace=True)
       train_set_phone_after = train_set_phone[train_set_phone['afterPush'] == 1]
       train_set_phone_after.reset_index(drop=True, inplace=True)
       train_set_phone_before = train_set_phone[train_set_phone['afterPush'] == 0]
       train_set_phone_before.reset_index(drop=True, inplace=True)
       #print(phone)

       train_set_phone_0 = train_set_phone_after.iloc[0]
       train_set_phone_energy_0 = train_set_phone_0['sizeEnergy(mAh)'] #* train_set_phone_0['volt(mV)'] / 1000
       train_set_phone_batch_0 = train_set_phone_0['batchSize']

       train_set_phone_1_after = train_set_phone_after.ix[1:]
       train_set_phone_energ_1 = train_set_phone_1_after['sizeEnergy(mAh)'] #* train_set_phone_1_after['volt(mV)'] / 1000
       y_train_phone = (train_set_phone_energ_1 - train_set_phone_energy_0) / (train_set_phone_1_after['batchSize'] - train_set_phone_batch_0)
       y_train_phone = pd.DataFrame(y_train_phone.values, columns=['slope'])
       y_train = y_train.append(y_train_phone, ignore_index=True)

       train_set_phone_1_before = train_set_phone_before.ix[1:]
       train_set_phone_1_after['totalFreq'] =  train_set_phone_1_after['cores'] * train_set_phone_1_after['cpuMaxFreqMean']
       train_set_phone_1_after['energPerJiffies'] = (train_set_phone_1_before['idleEnergy(mAh)'] * 1000) / train_set_phone_1_before['deviceCpuUsage']
       X_train = X_train.append(train_set_phone_1_after.loc[:, ml_features], ignore_index=True)

    return (X_train, y_train)

def train(df, pa):
    featuresPlusLabel = build_train_set(df)
    #print(featuresPlusLabel[1])
    if (pa == 1):
        mod = pa_fit(*featuresPlusLabel, 0)
    else:
        mod = lm_fit(*featuresPlusLabel)
    return mod

battery_sizes = {
    'COL-L29': 3400,
    'STF-L09': 3200,
    'Galaxy S7': 3000,
    'Galaxy S4 Mini': 1900,
    'Xperia E3': 2330,
}

ENERGY0 = {
    'COL-L29': 0.6,
    'STF-L09': 0.12,
    'Galaxy S7': 0.1,
    'Galaxy S4 Mini': 0.05,
    'Xperia E3': 0.14,
}

SLO = 0.075 # % of the battery capacity

def pa_retrain(df, stats):
    last_entry_after = df[(df['android_model'] == stats[3]) & (df['afterPush'] == "1")].tail(1)
    last_entry_before = df[(df['android_model'] == stats[3]) & (df['afterPush'] == "0")].tail(1)

    last_entry_after['energPerJiffies'] = (last_entry_before['idleEnergy(mAh)'].values[0] * 1000) / last_entry_before['deviceCpuUsage'].values[0]
    last_entry_after['totalFreq'] =  last_entry_after['cores'] * last_entry_after['cpuMaxFreqMean']
    X = last_entry_after.loc[:, ml_features]

    last_entry_energy = last_entry_after['sizeEnergy(mAh)'] #* last_entry_after['volt(mV)'] / 1000
    y = (last_entry_energy - ENERGY0[stats[3]]) / (last_entry_after['batchSize'] - last_entry_after['threads'])

    #print("Learning features:")
    #print(X, y)
    mod = pa_fit(X, y, 1)
    return mod

def computeMaxFreqMean(stats):
    nb_cores = float(stats[8])

    littleCores = np.array(stats[28:32])
    littleCores = littleCores.astype(np.float)
    bigCores = np.array(stats[32:36])
    bigCores = bigCores.astype(np.float)

    if(nb_cores <= 4):
        totalFreq = max(littleCores)
    elif(nb_cores == 8):
        totalFreq = (max(littleCores) + max(bigCores)) / 2
    else:
        raise ValueError('The number of CPUs is larger than 8.')
    return totalFreq

def makePrediction(mod_lm, stats):
    stats[44] = computeMaxFreqMean(stats)

    pred_df = pd.DataFrame([stats], columns=all_features)
    skip = pred_df.columns.drop(['clientID', 'afterPush', 'profilerEpoch', 'android_model', 'android_version', 'android_serialNumber'])
    pred_df[skip] = pred_df[skip].apply(pd.to_numeric, errors='coerce')

    pred_df['energPerJiffies'] = (pred_df['idleEnergy(mAh)'] * 1000) / pred_df['deviceCpuUsage']
    pred_df['totalFreq'] = pred_df['cores'] * pred_df['cpuMaxFreqMean']


    #print("Prediction features:")
    #print(pred_df.loc[:, ml_features])
    pred_slope = mod_lm.predict(pred_df.loc[:, ml_features])
    #print("Predicted slope: ", pred_slope[0])
    if (pred_slope[0] == 0):
        return 56

    RELATIVE_SLO = SLO * battery_sizes[stats[3]] / 100
    #print("Relative SLO for " + stats[3] + ": ", RELATIVE_SLO)
    batch_size = int((RELATIVE_SLO - ENERGY0[stats[3]]) / pred_slope[0]);
    print("Predicted batchSize for " + stats[3] + ": ", batch_size)
    if (batch_size > 5000):
        return 5000
    elif (batch_size < 56):
        return 56
    else:
        return batch_size + 8 - (batch_size % 8);

def storeStats(df, stats):
    stats[44] = computeMaxFreqMean(stats)
    df = df.append(dict(zip(all_features, stats)), ignore_index=True)
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
    #print(stats)
    #prediction or learn stats?
    if (stats[1] == "0"):
        if stats[3] in per_device_models:
            batchSize = makePrediction(per_device_models[stats[3]], stats)
            #print("PA batch size: ", batchSize);
            sendMessage(conn, [batchSize])
        else:
            batchSize = makePrediction(global_mod, stats)
            #print("new device prediction; batch_size = ", batchSize)
            sendMessage(conn, [batchSize])
        df = storeStats(df, stats)
    else:
        df = storeStats(df, stats)
        per_device_models[stats[3]] = pa_retrain(df, stats)
        global_mod = train(df, pa=0)
        print(stats[3] + ", ", (float(stats[19]) * 100) / battery_sizes[stats[3]])
  closeConnection(conn)
