CC=g++
CFLAGS=-c -std=c++11 -fPIC -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux -I$(JAVA_HOME)/include/darwin 
LDFLAGS=-fPIC -shared

SOURCES_DIR=src/main/c++
OBJECTS_DIR=target/c++

LIB_SOURCES_DIR=../commonLib/cpp_utils
LIB_OBJECTS_DIR=target/c++/lib

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
	EXECUTABLE=target/classes/libnative.so
endif
ifeq ($(UNAME_S),Darwin) # Mac specific
#	EXECUTABLE=target/classes/libnative.jnilib
	EXECUTABLE=target/classes/libnative.so
endif

SOURCES=$(shell find '$(SOURCES_DIR)' -type f -name '*.cpp')
OBJECTS=$(SOURCES:$(SOURCES_DIR)/%.cpp=$(OBJECTS_DIR)/%.o)

LIB_SOURCES=$(shell find $(LIB_SOURCES_DIR) -type f -name '*.cpp')
LIB_OBJECTS=$(LIB_SOURCES:$(LIB_SOURCES_DIR)/%.cpp=$(LIB_OBJECTS_DIR)/%.o)

all: $(EXECUTABLE)

$(EXECUTABLE): $(LIB_OBJECTS) $(OBJECTS)
	$(CC) $(LDFLAGS) $(OBJECTS) $(LIB_OBJECTS) -o $@

#$(OBJECTS): $(SOURCES)
$(OBJECTS_DIR)/%.o: $(SOURCES_DIR)/%.cpp
	mkdir -p $(OBJECTS_DIR)
	$(CC) $(CFLAGS) $< -o $@

$(LIB_OBJECTS_DIR)/%.o: $(LIB_SOURCES_DIR)/%.cpp
	mkdir -p $(LIB_OBJECTS_DIR)
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf $(OBJECTS_DIR) $(EXECUTABLE)
