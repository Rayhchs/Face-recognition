#ifndef FACE_AI_H
#define FACE_AI_H

#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/string_util.h>
#include <tensorflow/lite/optional_debug_tools.h>
#include <tensorflow/lite/kernels/register.h>

using namespace std;

namespace face_AI{

struct Detection_session
{

    std::unique_ptr<tflite::FlatBufferModel> model;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;

    int i_height;
    int i_width;
    int i_channels;
    int o_size;

    //tflite detection
    int score_size;
    int landmark_size;
    int bbox_size;
    int mask_size;

};

struct Recognition_session
{

    std::unique_ptr<tflite::FlatBufferModel> model;
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;

    int i_height;
    int i_width;
    int i_channels;
    int o_size;

};

struct PredBox
{

    float x1 = 0;
    float y1 = 0;
    float x2 = 0;
    float y2 = 0;

    //key_points <x y>
    std::vector<std::pair<float, float>> key_points = {};

    float score = 0;

};

class Face_Det{

public:
    // detection
	void init_det(Detection_session *session_detect, string model_dir);
	std::vector<cv::Mat> inference_det(cv::Mat &input, Detection_session *session_detect, std::vector<PredBox> &face_list);

private:

	// Declear variables of detection
    const int in_h = 128;
    const int in_w = 128;
	const int num_featuremap = 2;
	const float center_variance = 0.1;
	const float size_variance = 0.2;
	const std::vector<float> strides = {8.0, 16.0};
    const std::vector<std::vector<float>> min_boxes = {
            {8.0f,  11.0f},
            {14.0f,  19.0f, 26.0f, 38.0f, 64.0f, 149.0f}};
	const int num_keypoints = 5;
	const int num_priors = 896;
    const float score_threshold = 0.7;
    const float iou_threshold = 0.4;
	std::vector<std::vector<float>> priors = {};

	// Variables for face alignment
    float points_dst[5][2] = {
        { 38.2946f, 51.6963f },
        { 73.5318f, 51.5014f },
        { 56.0252f, 71.7366f },
        { 41.5493f, 92.3655f },
        { 70.7299f , 92.2041f }
    };

	// Computation for detection
	void CalBoxes(std::vector<PredBox> &bbox_collection,
                        	float * score_data,
	                        float * landmark_data,
	                        float * bbox_data);
	void GenPriorBox();
	void NMS(std::vector<PredBox> &input, std::vector<PredBox> &output);
	cv::Mat face_Alignment(cv::Mat Img, float landmarks[5][2]);
	cv::Mat similar_Transform(cv::Mat src,cv::Mat dst);
    cv::Mat mean_Axis(const cv::Mat &src);
    cv::Mat element_Wise_Minus(const cv::Mat &A,const cv::Mat &B);
    cv::Mat varAxis0(const cv::Mat &src);
    int matrix_Rank(cv::Mat M);
};


class Face_Rec{

public: 
    // recognition
    void init_rec(Recognition_session *session, string model_dir);
    int inference_rec(cv::Mat &face_in, std::vector<float> &face_feature,  Recognition_session *session);

private:
    const int in_h = 112;
    const int in_w = 112;
};
}
#endif