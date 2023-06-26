CC = g++
CFLAGS = -c -Wall
LDFLAGS = -ltensorflow-lite -ldl -lpthread -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -lsqlite3
INCLUDE += -I/home/ray/tensorflow_src/
INCLUDE += -I/home/ray/tflite_build/flatbuffers/include
INCLUDE += -I/usr/local/include/opencv4
SOURCES = test.cpp lib/face_AI.cpp
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE=test

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) -o $@ $(OBJECTS) $(LDFLAGS) 

.cpp.o:
	$(CC) $< -o $@ $(CFLAGS) $(INCLUDE)

clean:
	rm -rf *o $(EXECUTABLE)