#include "lib/face_AI.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace face_AI;
using namespace cv;

int main()
{
    string det_model_dir = "./models/Det.tflite";
    string rec_model_dir = "./models/Rec.tflite";
    Mat img = cv::imread("noface.jpg");

    static Face_Det face_det;
    static Face_Rec face_rec;

    // Initialize
    Detection_session session_Detect;
    Recognition_session session_Recognize;
    face_det.init_det(&session_Detect, det_model_dir);
    face_rec.init_rec(&session_Recognize, rec_model_dir);

    // Inference Detection
    vector<PredBox> face_list;
    auto temp_img = face_det.inference_det(img, &session_Detect, face_list);
    if (int(face_list.size()) <= 0){
        return -1;
    }

    // Inference Recognition
    vector<float> feature;
    int result = face_rec.inference_rec(img, feature, &session_Recognize);

    return 0;
}
