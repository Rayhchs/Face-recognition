CC = g++
CFLAGS = -c -Wall
LDFLAGS = -L./lib -ltensorflow-lite -L./lib -ldl -L./lib -lpthread -L./lib -lsqlite3 -L./lib `pkg-config --libs opencv`
LDFLAGS += -lyaml-cpp
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

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) -o $@ $(OBJECTS) $(LDFLAGS) 

.cpp.o:
	$(CC) $< -o $@ $(CFLAGS) $(INCLUDE)

create_folder:
	mkdir database

clean:
	rm -rf *o $(EXECUTABLE)


