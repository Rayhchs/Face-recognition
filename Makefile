CC = g++
CFLAGS = -c -Wall
LDFLAGS = -ltensorflow-lite -ldl -lpthread -lsqlite3 -L/usr/local/lib `pkg-config --libs opencv`
LDFLAGS += -lyaml-cpp
INCLUDE += -I/home/ray/tensorflow_src/
INCLUDE += -I/home/ray/tflite_build/flatbuffers/include
INCLUDE += -I/usr/local/include/opencv4
INCLUDE += -I /usr/include/eigen3
SOURCES = main.cpp lib/utils.cpp lib/face_AI.cpp

OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE=test

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) -o $@ $(OBJECTS) $(LDFLAGS) 

.cpp.o:
	$(CC) $< -o $@ $(CFLAGS) $(INCLUDE)

create_folder:
	mkdir database

clean:
	rm -rf *o $(EXECUTABLE)


