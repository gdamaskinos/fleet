#!/bin/bash

## Deploys FLeet locally 

handler()
{
  echo "Killing FLeet modules ..."
  kill $serverPID
  kill $driverPID
  gradle --stop
  pkill -f "run_on_single_machine.sh $emuSerial"
  pkill -f "local_deploy.sh $emuSerial"
  exit
}

trap handler SIGINT


if [ "$#" -ne 16 ]; then
  echo "usage: \nfrontend:$ $0 <emu_serial (e.g., 5554)> <server_port> <log_file> <dataset_dir> <batch_size_threshold> <similarity_threshold> <number of client requests> <eval rounds> <lrate> <M> <E> <sigma> <C> <staleness_size> <policy> <alpha>"
  exit 1
fi

cmd="$0 $@"
emuSerial=$1
serverPort=$2
log_file=$(readlink -f $3)
dataset_dir=$(readlink -f $4)
batch_size_threshold=$5
similarity_threshold=$6
clientRequestsNum=$7
evalRounds=$8
lrate=$9
M=${10}
E=${11}
sigma=${12}
C=${13}
staleness_size=${14}
policy=${15}
alpha=${16}

SCRIPT=`realpath -s $0`
SCRIPTPATH=`dirname $SCRIPT`

# check if emulator is running
{
  adb -s emulator-$emuSerial shell getprop init.svc.bootanim > /dev/null
} || {
  echo "Must first start Android emulator. e.g., cd $SCRIPTPATH/client_cmd_deployment && ./start_emu_single_machine.sh $number_of_emus 1 1000"
  exit
}

echo "Reproduce with: $cmd" > $log_file
cd $SCRIPTPATH
echo "Git commit: $(git rev-parse HEAD)" >> $log_file
echo "Git status: $(git status)" >> $log_file
echo "Emulator serial: $emuSerial" >> $log_file

# Compile Client
echo "Compiling Client..."
cd $SCRIPTPATH/Client
gradle assembleDebug >> $log_file 2>&1

# Compile and launch Server
echo "Compiling and launching Server..."
cd $SCRIPTPATH/Server
mvn clean
mvn -Dport=$serverPort tomcat7:run >> $log_file 2>&1 &
serverPID=$!

# Compile and launch Driver
echo "Compiling and launching Driver..."
cd $SCRIPTPATH/Driver
mvn clean install >> $log_file 2>&1  
mvn exec:java -Dexec.mainClass="coreComponents.Driver" -Dexec.args="http://localhost:${serverPort}/Server/Server $dataset_dir $dataset_dir $batch_size_threshold $similarity_threshold $clientRequestsNum $evalRounds $lrate $M $E $sigma $C $staleness_size $policy $alpha" >> $log_file 2>&1 &
driverPID=$!

echo "Server PID: "$serverPID
echo "Driver PID: "$driverPID

# Launch Client
echo "Launching Client..."
cd $SCRIPTPATH/client_cmd_deployment
./run_on_single_machine.sh $emuSerial runner.py $SCRIPTPATH/Client/app/build/outputs/apk/debug/app-x86-debug.apk $(hostname -I | awk '{print $1}') $serverPort 120

