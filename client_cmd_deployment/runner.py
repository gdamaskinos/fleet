"""
  Copyright (c) 2020 Georgios Damaskinos
  All rights reserved.
  @author Georgios Damaskinos <georgios.damaskinos@gmail.com>
  This source code is licensed under the MIT license found in the
  LICENSE file in the root directory of this source tree.
"""


from com.android.monkeyrunner import MonkeyRunner, MonkeyDevice
import socket
import os
import time
import sys

# definitions
package  = "com.androidsrc.client"
activity = "coreComponents.MainActivity"

runComponent = package + "/" + activity

emuSerial = int(sys.argv[1])
serverIP = sys.argv[2]
serverPort = int(sys.argv[3])

start_time = time.time()
device = "emulator-" + str(emuSerial)

device1 = MonkeyRunner.waitForConnection(deviceId=device)
device1.startActivity(component=runComponent)

device1.touch(0, 0, MonkeyDevice.DOWN_AND_UP)
MonkeyRunner.sleep(5) # DON'T MODIFY

# Text Box 1 (IP)
# avoid MonkeyRunner issues when having sleep time after each button press
# otherwise only a few first characters are input
for c in serverIP:
  device1.type(c)
device1.press("KEYCODE_TAB", MonkeyDevice.DOWN_AND_UP)

# Text Box 2 (port)
device1.type(str(serverPort))
device1.press("KEYCODE_TAB", MonkeyDevice.DOWN_AND_UP)

# Text Box 3 (clientId)
device1.type(socket.gethostname() + ":" + device)
device1.press("KEYCODE_TAB", MonkeyDevice.DOWN_AND_UP)

# Text Box 4 (sleep time)
device1.type("0")
device1.press("KEYCODE_TAB", MonkeyDevice.DOWN_AND_UP)

# Button 1 (start)
device1.press("KEYCODE_ENTER", MonkeyDevice.DOWN_AND_UP)

print("Total runner.py time: %g" % (time.time() - start_time))

