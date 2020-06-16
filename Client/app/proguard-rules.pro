# Add project specific ProGuard rules here.
# By default, the flags in this file are appended to flags specified
# in /home/aubian/Android/Sdk/tools/proguard/proguard-android.txt
# You can edit the include path and order by changing the proguardFiles
# directive in build.gradle.
#
# For more details, see
#   http://developer.android.com/guide/developing/tools/proguard.html

# Add any project specific keep options here:

# If your project uses WebView with JS, uncomment the following
# and specify the fully qualified class name to the JavaScript interface
# class:
#-keepclassmembers class fqcn.of.javascript.interface.for.webview {
#   public *;
#}
#-ignorewarnings
#-keep class android.support.v7.widget.SearchView { *; }
#-dontwarn com.github.jaiimageio.impl.plugins.tiff.TIFFJPEGCompressor
#-dontwarn com.google.common.util.concurrent.MoreExecutors
#-dontwarn com.twelvemonkeys.imageio.plugins.tiff.*
#-dontwarn javassist.bytecode.ClassFile
#-dontwarn lombok.delombok.ant.Tasks$Delombok
#-dontwarn lombok.launch.PatchFixesHider$Util
#-dontwarn org.bytedeco.javacpp.indexer.UnsafeRaw
#-keep org.deeplearning4j.*
#-dontwarn org.joda.time.DateTimeZone
#-keep org.nd4j.shade.jackson.dataformat.xml.XmlFactory
#-dontwarn org.objenesis.instantiator.*
#-dontwarn org.reflections.util.Utils
#-dontwarn com.google.common.cache.Striped64
#-keep class android.support.v4.app.** { *; }
#-keep interface android.support.v4.app.** { *; }
#-keep class android.support.v7.app.** { *; }
#-keep interface android.support.v7.app.** { *; }