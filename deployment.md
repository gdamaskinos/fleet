# Deployment

## IDE Setup
  * Eclipse EE
    * Java 8 support eclipse kepler sr2,
    * Java 8 Facets for web tools eclipse kepler sr2
    * Java 8 support for m2e for Eclipse Kepler SR2
  * Optional: [external tomcat 7](#tomcat) (better use the maven integrated tomcat)
  * Android Studio 3.5.1

## Server
* Compile + Deploy
  * Export JAVA_HOME environmental variable (vital for JNI support)
  * Set the preferred the port [here](Server/pom.xml#L212)
  * Eclipse
    * Import [Server](Server/) to eclipse as project
    * Project Properties > Run as > Maven build... > Goals: tomcat7:run
  * Command line
    ```bash
    cd /path/to/Server
    mvn clean # only for the first time
    export MAVEN_OPTS=-Xmx2g # constrain the memory consumption 
    mvn tomcat7:run
    ```
    
## Driver
* Compile
  * Export JAVA_HOME environmental variable (vital for JNI support)
  * Eclipse
    * Import [Driver](Driver/) to eclipse as project
    * Project Properties > Run as > Maven build... > Goals: clean install
    * environmental variables not defined
      * Define them in Run Configurations > Environment of the maven build
  * Command line
    ```bash
    cd /path/to/Driver
    mvn clean install
    ```

* Run
  * Eclipse
    * Project Properties > Run As > Java Application
    * Add VM argument: -Djava.library.path=/path/to/Driver/target/classes
  * Command line
    ```bash
    cd /path/to/Driver
    export MAVEN_OPTS="-Xmx100g"
    mvn exec:java -Dexec.mainClass="coreComponents.Driver" -Djava.library.path=/path/to/Driver/target/classes -Dexec.args="arg0 arg1"
    ```
  * Follow instructions for adding run arguments and VM arguments


## Client
* Import [Client](Client/) as android project to android studio (tested on version 3.0)
* Run
  * Android studio
    * Local android emulator (set [abi](Client/app/build.gradle#L42) to 'x86')
    * Mobile device (set [abi](Client/app/build.gradle#L42) to 'armeabi')
  * APK
    * Export: Build > Build APK (s)
    * [Deploy on android emulators with command line](client_cmd_deployment/) (compiled with 'x86')
    * Deploy on a mobile phone (compiled with 'armeabi')
      * Energy measurement:
        * Grant permissions
          ```bash
          adb -s [device] shell pm grant com.androidsrc.client android.permission.PACKAGE_USAGE_STATS
          adb -s [device] shell pm grant com.androidsrc.client android.permission.DUMP
          ```
        * For more accurate measurements, reduce backlight to the minimum
    * Deploy on [ADF](https://aws.amazon.com/device-farm/)
      * Remove dl4j dependencies from the Client ([issue with ADF](../../../issues/31))
      * Build apk (with 'armeabi')
      * Upload apk to ADF
      * Use Built-in: Fuzz for the configure test
    * Deploy on [Firebase](https://firebase.google.com/docs/test-lab/)
      * Build apk (with 'armeabi')
      * Upload apk to Firebase 
      * Use Robo test with a bound in the runtime as this test starts exploring the app UI
    * Deploy on Google Play
      * Build > Generate Signed APK...
      * Choose existing... > Select your keystore hynet1234.jks
      * Enter the keystore password
      * Press Next button
      * Build type : release
      * Check all boxes : V1 and V2
      * Press Finish button


## Troubleshoot

### Tomcat
* sftp/scp the __tomcat7__ folder to the target server (hosting the tomcat server).
* Modify conf/tomcat-users.xml for username/password for tomcat manager-gui
  * Do not use a locked username (e.g. "admin")
* Modify _pathname_ in conf/server.xml
* Modify permission of conf/web.xml (chmod 644) to access manager-gui across any browser at any ip address.
* Modify /tomcat7/webapps/manager/WEB-INF/web.xml:
  * Increase the max-file-size and max-request-size:
    ```xml
    <multipart-config>
    <!– 50MB max –>
    <max-file-size>52428800</max-file-size>
    <max-request-size>52428800</max-request-size>
    <file-size-threshold>0</file-size-threshold>
    </multipart-config>
    ```
* Modify /tomcat/conf/server.xml:
  * Set maxPostSize (unlimited):
    ```xml
    <Connector port="8080" protocol="HTTP/1.1"
      connectionTimeout="20000"
      redirectPort="8443"
      maxPostSize="-1" />
    ```
* (Optional) In case jdk is referenced:
  ```bash
  echo "export JAVA_HOME=/path/to/nonRoot/jdk" > $TOMCAT_HOME/bin/setenv.sh
  chmod +x $TOMCAT_HOME/bin/setenv.sh
  ```
* Set heap size:
  ```bash
  export CATALINA_OPTS="-Xms512M -Xmx1024M"
  ```

* Deploy Server 
  * Export TOMCAT_HOME and JAVA_HOME
  * Edit the corresponding lines on the Server [Makefile](Server/Makefile)
  * Eclipse
    * Tomcat (right click) > Add and remove... > add Module (Server)
    * Tomcat (double click) > Open launch configuration > Add to VM arguments: -Djava.library.path="/path/to/tomcat_home/lib"
    * Project Properties > Run as > Maven build... > Goals: clean install
    * Restart tomcat 
  * Command line: Upload generated war file to tomcat (inside webapps folder) and restart tomcat
    ```bash
    export JAVA_OPTS="-Djava.library.path=$TOMCAT_HOME/lib"
    $TOMCAT_HOME/bin/shutdown.sh # shutdown
    sleep 2 # wait for shutdown to complete
    rm -rf $TOMCAT_HOME/webapps/Server" # undeploy
    cp Server/target/Server.war $TOMCAT_HOME/webapps/ # copy
    $TOMCAT_HOME/bin/startup.sh # start
    ```

* NullPointerException when loading jni native files:
  * Make sure the $TOMCAT_HOME is set to the same directory as the running tomcat instance

#### Eclipse Troubleshoot
* build stuck (sleeping)
  * Migrate to a new workspace
* Cannot resolve (e.g., utils.dl4j)
  * Delete project and import again

#### Android Troubleshoot
* Tested on MAC with HAXM (3GB RAM) with AVD (3GB RAM, 64MB VM, 2GB storage)
* Insufficient AVD storage | AVD not starting
  * First start the emulator > then build and push the app
  * Increase AVD internal storage
* AVD crashes the OS
  * Reduce the AVD RAM
  * [Increase the HAXM RAM](http://stackoverflow.com/a/29288735)
  * Set Graphics from "Automatic" to "Software"
* In case of errors for gradle file
  * check android version
  * android tool version
  * jdk
  * version used in build.gradle files
* Client: Caused by: java.lang.IndexOutOfBoundsException: Index: 58, Size: 0
  * Potential cause: Some exception on the Server causes serialization issues
  * Check Server locally as most exceptions are not shown in catalina.out
* Buffer overflow, no available space
  * Increase emulator's heap size and RAM
* Caused by: java.lang.OutOfMemoryError: Cannot allocate new FloatPointer(3801600): totalBytes = 242M, physicalBytes = 338M
  * Increase emulator's heap size
* android studio clang++: error: no such file or directory: libc++.so
  ```bash
  cd /path/to/Android/Sdk/ndk-bundle/sources/cxx-stl/llvm-libc++/libs/x86
  ln -s libc++_shared.so libc++.so
  ```
* No toolchains found in the NDK toolchains folder for ABI with prefix: mips64el-linux-android
  ```bash
  cd /path/to/Android/sdk/ndk-bundle/toolchains/
  ln -s aarch64-linux-android-4.9/ mips64el-linux-android
  ln -s arm-linux-androideabi-4.9/ mipsel-linux-android
  ```


