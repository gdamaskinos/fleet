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
    pd.set_option('display.max_columns', 500)
    pd.set_option('mode.chained_assignment', None)

    #convert everything to numeric except...
    cols = df.columns.drop(['clientID', 'afterPush', 'profilerEpoch', 'android_model', 'android_version', 'android_serialNumber'])
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

    df = df[df['afterPush'] == 1]
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
ml_features = ['batchSize']
keep_features = ['batchSize', 'sizeEnergy(mAh)']

def build_train_set(df):
    df = df[df['android_model'] == 'Galaxy S4 Mini']
    train_set = df.loc[:, keep_features]
    train_set.dropna(inplace=True)
    train_set.reset_index(drop=True, inplace=True)

    return (train_set.loc[:, ml_features], train_set['sizeEnergy(mAh)'])

def train(df):
    featuresPlusLabel = build_train_set(df)
    #print(featuresPlusLabel[1])
    mod = lm_fit(*featuresPlusLabel)
    return mod

battery_sizes = {
    'COL-L29': 3400,
    'STF-L09': 3200,
    'Galaxy S7': 3000,
    'Galaxy S4 Mini': 1900,
    'Xperia E3': 2330,
}
SLO = 0.075 # % of the battery capacity

def makePrediction(mod_lm, stats):
    pred_df = pd.DataFrame([stats], columns=all_features)
    cols = pred_df.columns.drop(['clientID', 'afterPush', 'profilerEpoch', 'android_model', 'android_version', 'android_serialNumber'])
    pred_df[cols] = pred_df[cols].apply(pd.to_numeric, errors='coerce')

    coefs = mod_lm.coef_
    intercept = mod_lm.intercept_
    print(intercept, coefs)

    RELATIVE_SLO = SLO * battery_sizes[stats[3]] / 100
    #print("Relative SLO for " + stats[3] + ": ", RELATIVE_SLO)
    batch_size = int((RELATIVE_SLO - intercept) / coefs[0])
    #print("Predicted batchSize: ", batch_size)

    if (batch_size > 2000):
        return 2000
    elif (batch_size < 56):
        return 56
    else:
        return batch_size + 8 - (batch_size % 8);

def storeStats(df, stats):
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

  parser.add_argument("--trainDataset", default='energy.csv',
      help='Path to the pretraining dataset')

  args = sys.argv[1:]
  args = parser.parse_args(args)

  df = loadTrainingDataset(args.trainDataset)
  global_mod = train(df)
  per_device_models = {}

  mutex = Lock()

  conn = openServerConn(port=args.clientPort)
  while True:
   with mutex:
    stats = getMessage(conn)
    #prediction or learn stats?
    if (stats[1] == "0"):
        batchSize = makePrediction(global_mod, stats)
        print("new device prediction for " + stats[3] + " : ", batchSize)
        sendMessage(conn, [batchSize])
    else:
        df = storeStats(df, stats)
        global_mod = train(df)
        print(stats[3] + ", ", (float(stats[19]) * 100) / battery_sizes[stats[3]])
  closeConnection(conn)
