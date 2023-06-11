CC = g++
CFLAGS = -c -Wall
LDFLAGS = -ltensorflow-lite -ldl -lpthread -lsqlite3 -L/usr/local/lib `pkg-config --libs opencv`
LDFLAGS += -lyaml-cpp
INCLUDE += -I./include/tensorflow_src/
INCLUDE += -I./include/flatbuffers/include
INCLUDE += -I./include/opencv4
INCLUDE += -I./include/eigen3
SOURCES = main.cpp lib/utils.cpp lib/face_AI.cpp

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


