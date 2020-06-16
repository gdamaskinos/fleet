cmake_minimum_required(VERSION 3.4.1)
set(compiler_c_flags "-Wno-error=format-security")
ADD_DEFINITIONS(
    -std=c++11
    -mfpu=neon
    -O1
    -mtune=cortex-a57.cortex-a53
    # Other flags
)
add_library( # Specifies the name of the library.
             native-lib

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             native-lib.cpp)

add_library( # Specifies the name of the library.
             simpleCNN-lib

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             simpleCNN-lib.cpp)

add_library( # Specifies the name of the library.
             cppNN-lib

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             cppNN-lib.cpp)

add_library( # Specifies the name of the library.
             caloree

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             caloree.cpp)

add_library( # Specifies the name of the library.
             Base64

             # Sets the library as a shared library.
             SHARED

             # Provides a relative path to your source file(s).
             ../../../../../commonLib/cpp_utils/Base64.cpp)


# Include libraries needed for native-lib
target_link_libraries(native-lib
                      android
                      log)

# Include libraries needed for caloree
target_link_libraries(caloree
                      log)

# Include libraries needed for cppNN-lib
target_link_libraries(cppNN-lib
                      caloree
                      Base64
                      android
                      log)

# Include libraries needed for simpleCNN-lib
target_link_libraries(simpleCNN-lib
                      Base64
                      android
                      log)
