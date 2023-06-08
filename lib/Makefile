
CC := g++
CFLAGS := -shared -fPIC -O3

CFLAGS=-c -Wall
LDFLAGS= -ltensorflow-lite -ldl -lpthread -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs

INCLUDE += -I/home/ray/tensorflow_src/
INCLUDE += -I/home/ray/tflite_build/flatbuffers/include
INCLUDE += -I/usr/local/include/opencv4

# 定义目标文件和源文件
TARGET := libface_AI.so
SRCS := face_AI.cpp

# 生成目标文件
$(TARGET): $(SRCS) $(TFLITE)
	$(CC) $(CFLAGS) -o $(TARGET) $(INCLUDE) $(LDFLAGS) $(SRCS)

.PHONY: clean
clean:
	rm -f $(TARGET)


# # 模型檔案名稱
# MODEL_FILE = BlazeFace.tflite
# # 轉換後的 C++ 檔案名稱
# MODEL_CPP = BlazeFace.cpp
# # 轉換後的 C++ 函式名稱
# MODEL_FUNC = BlazeFace

# all: $(MODEL_CPP)

# $(MODEL_CPP): $(MODEL_FILE)
# 	xxd -i $< > $@

# clean:
# 	rm -f $(MODEL_CPP)

# .PHONY: all clean


# CC = g++
# CFLAGS = -Wall -shared -fPIC
# MODEL_SRC = BlazeFace.cpp
# MODEL_OBJ = BlazeFace.o
# MODEL_DATA = BlazeFace.h
# TARGET = BlazeFace.so

# all: $(TARGET)

# # 生成目标文件
# $(TARGET): $(MODEL_SRC) $(MODEL_DATA)
# 	$(CC) $(CFLAGS) -o $(TARGET) $(MODEL_SRC)


# clean:
# 	rm -f $(TARGET) $(MODEL_DATA)
