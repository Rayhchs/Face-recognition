CC = g++
CFLAGS = -c -Wall
LDFLAGS += -L./lib ./lib/libpthread_nonshared.a -ltensorflow-lite -ldl -lpthread -lsqlite3 -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio
LDFLAGS += -lyaml-cpp
INCLUDE += -I./include/yaml-cpp/
INCLUDE += -I./include/sqlite3/
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


