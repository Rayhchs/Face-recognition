# Face Recognition

This repository contains code and resources for performing face recognition tasks using various algorithms and models.

## Features

* Face detection: Detecting faces in images or video streams.
* Face recognition: Recognizing and identifying individuals based on their faces.


## Installation

1. Clone the repository:

   ```bash
   git https://github.com/Rayhchs/Face-recognition.git
   cd Face-recognition
   ```

2. Usage:

   ```bash
   make create_folder
   make all
   ./main
   ```
   - press 'r' to register a face
   - press 'i' to identify a face from database
   - press 'q' to quit

## Models
* Detection model, $Det.tflite$ is blazeface.
* Recognition model, $Rec.tflite$ is MobileFaceNet converted from pretrain model of [FaceX-zoo](https://github.com/JDAI-CV/FaceX-Zoo/tree/main/training_mode).

Recognition model can be changed to any kind of tflite model. However, Detection model cannot.