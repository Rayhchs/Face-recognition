CC = g++
CFLAGS = -c -Wall
LDFLAGS = -L./lib -L./lib -L./lib -L./lib -lsqlite3 -lyaml-cpp -ltensorflow-lite -ldl -lpthread -lpthread_nonshared `pkg-config --libs opencv`
INCLUDE += -I./include/yaml-cpp/
INCLUDE += -I./include/sqlite3/
INCLUDE += -I./include/yaml-cpp/
INCLUDE += -I./include/tensorflow/
INCLUDE += -I./include/flatbuffers/
INCLUDE += -I./include/opencv4/
INCLUDE += -I./include/eigen3
SOURCES = main.cpp common/utils.cpp common/face_AI.cpp

OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE=main

# run:
# 	export PKG_CONFIG_PATH=./lib/pkgconfig:$PKG_CONFIG_PATH && ./$(EXECUTABLE)

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) -o $@ $(OBJECTS) $(LDFLAGS) 

.cpp.o:
	$(CC) $< -o $@ $(CFLAGS) $(INCLUDE)

create_folder:
	mkdir database

clean:
	rm -rf *o $(EXECUTABLE)


