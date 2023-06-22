# Face Recognition

This repository contains code and resources for performing face recognition tasks using various algorithms and models.

## Features

* Face detection: Detecting faces in images or video streams.
* Face alignment: Aligning faces to a standardized pose for accurate recognition.
* Face recognition: Recognizing and identifying individuals based on their faces.


## Installation

1. Clone the repository:

   ```bash
   git https://github.com/Rayhchs/Face-recognition.git
   cd Face-recognition
   ```

2. Download rest libraries

- sqlite3

   ```bash
   sudo apt install sqlite3
   sudo apt install libsqlite3-dev
   ```

- yaml-cpp

see https://github.com/jbeder/yaml-cpp for more

## Usage

   ```bash
   make create_folder
   make
   ./main
   ```
   - press 'r' to register a face
   - press 'i' to identify a face from database
   - press 'q' to quit

## Models
* Detection model, $./models/Det.tflite$ is [Blazeface](https://arxiv.org/abs/1907.05047) which is converted from pretrain model of [blazeface](https://github.com/zineos/blazeface).
* Recognition model, $./models/Rec.tflite$ is [MobileFaceNet](https://arxiv.org/abs/1804.07573) converted from pretrain model of [FaceX-zoo](https://github.com/JDAI-CV/FaceX-Zoo/tree/main/training_mode).

Recognition model can be changed to any kind of tflite model. However, Detection model cannot.

## Acknowledge
Code and model heavily borrows from [zineos_blazeface](https://github.com/zineos/blazeface) and [FaceX-zoo](https://github.com/JDAI-CV/FaceX-Zoo/tree/main/training_mode). Thanks for the excellent work!