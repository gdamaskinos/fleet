# Deploy the Client from command line

## Install SDK/NDK

* Export Root Variables
  ```bash
   export ANDROID_SDK_HOME=/path/to/where/avds/will/be/saved  # .android parent folder !! DON'T PUT IN NFS (issues with died services)
   export ANDROID_SDK_ROOT=/path/to/android_sdk
   export ANDROID_HOME=/path/to/android_sdk
  ```
* [Download](https://developer.android.com/studio/index.html) latest sdk command line tools (checked with tools 28.0.3 for linux)
* Unzip to $ANDROID_SDK_ROOT 
* Export Path Variables
  ```bash
   # the following paths may alter depending on the sdk tools
   export PATH=$ANDROID_SDK_ROOT/tools/bin:$PATH
   export PATH=$ANDROID_SDK_ROOT/emulator:$PATH
   export PATH=$ANDROID_SDK_ROOT/platform-tools:$PATH
  ```
* Execute:
  ```bash
    sdkmanager --list
  ```
* Find the package_ids corresponding to the latest SDK version (or the chosen one) for:
  * Android SDK tools
  * Android SDK platform-tools
  * Android SDK Build-tools
  * SDK Platform Android
* Find the package_id for the ABI (used to create avd emulator):
  * For API-25: Google APIs Intel x86 Atom_64 System Image (or without "Atom_64" for x86 machines) 
  ```bash
    sdkmanager "build-tools;28.0.3" "emulator" "platform-tools" "platforms;android-25" "system-images;android-25;google_apis;x86"
  ```
* Install NDK
  ```bash
  sdkmanager ndk-bundle
  ```

## Common commands:
(see android studio app log when running adb emulator locally for example commands)

### create avd
* Make sure the folder: $ANDROID_SDK_ROOT/system-images/android-25/google_apis/x86_64 exists
```bash
avdmanager create avd -n emu1 -k "system-images;android-25;google_apis;x86"
rm $ANDROID_SDK_HOME/.android/avd/emu1.avd/multiinstance.lock # enable multiple emulators on same machine
```
* Modify the -t, --abi, --tag values accordingly for a different API and/or architecture (x86). These values depend on the downloaded packages
* Each avd takes around 2-4GB

### start emulator 
```bash
emulator -avd [emulator name] -no-window
```
* In case of hardware missing features (e.g. emulator: WARNING: Host CPU is missing the following feature(s) required for x86_64 emulation: SSSE3 SSE4.1 SSE4.2 POPCNT) the emulator will probably not start
* The emulator might not start in some machines (e.g. some of the graphene nodes)
* Troubleshoot: Modify `$ANDROID_SDK_HOME/.android/avd/[emulator name].avd/config.ini`
Tested with (RAM and heap size in MB) on grid5000 parasilo:
  * hw.ramSize=3000M
  * vm.heapSize=512M
  * disk.dataPartition.size=2000M


### launch emulator shell
```bash
adb -s [emulator-serial] shell
```
* Check /proc/meminfo for the allocated RAM
* default [emulator-serial]: emulator-5554

### logcat output
```bash
adb logcat
```

### install apk
```bash
adb -s [emulator-serial] install -r [path to apk]
```
* default [emulator-serial]: emulator-5554

### run the android app with monkeyrunner
* Execute [runner.py](runner.py) inside node:
```bash
monkeyrunner runner.py [number_of_emus] [server_ip]
```
* IMPORTANT: make sure the Client input fields are empty (i.e., comment out any setText commands from the Client)

### run app in shell
```bash
shell# am start [package]/[activity]
```
* Example:
```bash
shell# am start com.androidsrc.client/coreComponents.MainActivity
```

### stop app
```bash
shell# am force-stop [package]
```

### stop emulator
```bash
adb -s [emulator-serial] emu kill
```

### delete emulator
```bash
android delete avd -n [emulator name]
```

### push file to emulator
```bash
adb root
adb shell "mount -o rw,remount /system"
adb push /path/to/file /path/to/emulator/
```

### run other applications in emulator
* list packages in emulator
  ```bash
  shell# pm list packages
  ```

* use monkeytest to test an application 
(some applications will be running after the monkeytest is finished)
  ```bash
  shell# monkey -p [package name] -v [the number of action taken by monkeytest, like 500] 
  ```

* run tensorflow classification app with monkeytest
  ```bash
  adb install -r tensorflow_demo.apk
  adb shell am start -a android.intent.action.MAIN org.tensorflow.demo/.ClassifierActivity # for testing the app
  adb shell monkey -p org.tensorflow.demo -v 1000
  ```

### energy consumption measurement
* Grant the necessary perimissions
  ```bash
  adb shell pm grant com.[package] android.permission.DUMP
  adb shell pm grant com.[package] android.permission.PACKAGE_USAGE_STATS
  ```

* Mock battery unplugged
  ```bash
  adb shell dumpsys battery unplug
  ```
* Reset mock
  ```bash
  adb shell dumpsys battery reset
  ```

## Deploy scripts

Must comment out all the .setText() commands from the [MainActivity.java](../Client/app/src/main/java/coreComponents/MainActivity.java)

### [start_emu_single_machine.sh](start_emu_single_machine.sh)
Starts emulators on this node with name emu1, emu2 etc
* these emulator must exist

### [stop_emu_single_machine.sh](stop_emu_single_machine.sh)
Stops the emulators on this node

### [run_on_single_machine.sh](run_on_single_machine.sh)
Runs python script with monkeyrunner on this node

### [start_emu_g5k.sh](start_emu_g5k.sh)
Starts one emulator on each node with name emu1, emu2 etc
* Execute after connecting to reserved job
* these emulator must exist

### [stop_emu_g5k.sh](stop_emu_g5k.sh)
Stops the emulator on each node
* Execute after connecting to reserved job

### [run_on_g5k.sh](run_on_g5k.sh)
Runs python script with monkeyrunner on each node
* Execute after connecting to reserved job


# Setup for automated tests
## Code modifications
  * Set _testMode = true_ in [MainActivity.java](../Client/app/src/main/java/coreComponents/MainActivity.java).

      If testMode is enabled the client application runs with the default values which can be modified in MainActivity.java. The code   inside _buttonStop.setOnClickListener_ transforms Stop Button into a "dummy" button to avoid an unintended termination due to the random actions in automated tests. 
  * Assign the values which define the communication with the server and the client name.
      ```
      serverIP.setText("10.0.2.2");
      serverPort.setText("9999");
      clientID.setText("Client1");
      sleepTime.setText("0");
      ```
  * Build one APK per client running in the experiment setting respectively the clientID in the TextView.
* [ProfilerDispatcher.java](../Server/src/main/java/coreComponents/ProfilerDispatcher.java)
    Create the PAProfiler and/or Profiler instances that will run in the experiment setting _alignedLogins_ field to true.
* Define in PAProfiler and Profiler constructors the hardcoded logins setting the login epoch for each client and the total queries they send during an experiment. The structures defining hardcoded logins must be exactly the same in both classes.

## Deployment
* Firebase - Robo test
  * Create a _Robo Script_ following the instructions [Robo Test](https://firebase.google.com/docs/test-lab/android/robo-ux-test). The current [script](robo_script.json) can be used for the experiments.
  * Go to the _Test Lab_ service, select to run a _Robo Test_ and upload one APK and one Robo Script for each client.
  * Select one physical device and set the _Test timeout_ to the maximum value (currently 30 minutes) in the _Advanced options_ tab.
  * Optional: Firebase offers the option to run user-defined tests. In this way the fields assignment could be done through these tests, without setting default hardcoded values, and a pool of devices could run with a single script instead of running multiple different experiments. Info about how to create such tests [here](https://firebase.google.com/docs/test-lab/).
  
* Amazon device farm
  * Upload one APK per client.
  * In the _Configure_ tab select _Built-in: Fuzz_ and set to the fields _Event count_ and _Event throttle_ the maximum values to assure that the tests will not finish prematurely.
  * Create a _Device pool_ with one device, set the _Execution timeout_ and run. Set the execution timeout concerning the needs of the experiment because the pricing policy is based on the running time of each experiment.
  * Optional: Device Farm currently provides support for other test types. More info [here](https://docs.aws.amazon.com/devicefarm/latest/developerguide/welcome.html).
  
## Known issues - tips
* For each client included in an experiment a separate run must be initialized, instead of a single run with a pool of some devices, because no arguments can be given dynamically in the current automated tests. For more automation, focus must be given on the user-defined test options.
* Caution: some runs might crash for not clear reasons. Doing regular checks during experiment execution is recommended in order to deal with such cases on time.
