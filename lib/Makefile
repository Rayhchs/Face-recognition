
CC := g++
CFLAGS := -shared -fPIC -O3

CFLAGS=-c -Wall
LDFLAGS= -ltensorflow-lite -ldl -lpthread -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

INCLUDE += -I/home/ray/tensorflow_src/
INCLUDE += -I/home/ray/tflite_build/flatbuffers/include
INCLUDE += -I/usr/local/include/opencv4


TARGET := libface_AI.so
SRCS := face_AI.cpp

TARGET := libface_AI.so
SRCS := face_AI.cpp

$(TARGET): $(SRCS) $(TFLITE)
	$(CC) $(CFLAGS) -o $(TARGET) $(INCLUDE) $(LDFLAGS) $(SRCS)

.PHONY: clean
clean:

	rm -f $(TARGET)