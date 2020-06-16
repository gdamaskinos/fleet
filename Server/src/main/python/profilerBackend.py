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

  args = sys.argv[1:]
  args = parser.parse_args(args)

  mutex = Lock()

  per_device_index = {}

  conn = openServerConn(port=args.clientPort)
  while True:
   with mutex:
    stats = getMessage(conn)
    #print(stats)
    if (stats[1] == "0"):
        if stats[3] in per_device_index:
            per_device_index[stats[3]] += 8
            batchSize = per_device_index[stats[3]]
            print("old key: ", stats[3], batchSize)
            sendMessage(conn, [batchSize])
        else:
            per_device_index[stats[3]] = 8
            print("new key: ", stats[3], 8)
            sendMessage(conn, [8])
    else:
        print("got updates for ", stats[3])
  closeConnection(conn)

