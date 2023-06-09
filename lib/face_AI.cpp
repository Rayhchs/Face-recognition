#include "face_AI.h"
#include <iostream>
#include <numeric>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#define clip(x, y) (x < 0 ? 0 : (x > y ? y : x))

using namespace std;
using namespace cv;
using namespace face_AI;

// Face Detection
void Face_Det::init_det(Detection_session *session_detect, std::string model_dir) {
    
    // Load model
    session_detect->model =
        tflite::FlatBufferModel::BuildFromFile(model_dir.c_str());

    if (!session_detect->model)
    {

        cout << "Invalid to load detecion model" << endl;
    }

    // Create interpreter
    tflite::InterpreterBuilder(*session_detect->model, session_detect->resolver)(&session_detect->interpreter);
    if (!session_detect->interpreter)
    {

        cout << "Invalid to construct interpreter" << endl;
    }

    // Allocate Tensors
    if(session_detect->interpreter->AllocateTensors() != kTfLiteOk)
    {
        cout << "Invalid to allocate" << endl;
    }

    // Get input 
    int input_det = session_detect->interpreter->inputs()[0];
    TfLiteIntArray* input_Dims_Det = session_detect->interpreter->tensor(input_det)->dims;

    session_detect->i_height = input_Dims_Det->data[3];  // height
    session_detect->i_width = input_Dims_Det->data[2];   // width
    session_detect->i_channels = input_Dims_Det->data[1]; // channels

    // Get output(0) size
    int output_det = session_detect->interpreter->outputs()[0];
    TfLiteIntArray* output_Dims_Det = session_detect->interpreter->tensor(output_det)->dims;
    session_detect->o_size = output_Dims_Det->data[1];

    int score_ = session_detect->interpreter->outputs()[0];    //0
    int landmark_ = session_detect->interpreter->outputs()[1]; //1
    int bbox_ = session_detect->interpreter->outputs()[2];     //2

    TfLiteIntArray* score_Dims = session_detect->interpreter->tensor(score_)->dims;
    TfLiteIntArray* landmak_Dims = session_detect->interpreter->tensor(landmark_)->dims;
    TfLiteIntArray* bbox_Dims = session_detect->interpreter->tensor(bbox_)->dims;

    session_detect->score_size = score_Dims->data[2];
    session_detect->landmark_size = landmak_Dims->data[2];
    session_detect->bbox_size = bbox_Dims->data[2];
    Face_Det::GenPriorBox();
}


std::vector<cv::Mat> Face_Det::inference_det(Mat &input, Detection_session *session_detect, vector<PredBox> &face_list)
{
    int ret = 0;
    vector<Mat> faces;

    // check image source
    if (input.empty()){
        std::cout<<"Invalid to input Mat!"<<std::endl;
        return faces;
    }

    int max_edge_size;
    int min_edge_size;
    if (input.cols >input.rows)
    {
        max_edge_size = input.cols;
        min_edge_size = input.rows;
    }else
    {
        max_edge_size = input.rows;
        min_edge_size = input.cols;
    }

    // Put image into square
    Mat scale_img = Mat(max_edge_size,max_edge_size,CV_8UC3, Scalar(104,117,123));
    Rect Roi = Rect(0,0,input.cols,input.rows);
    input.copyTo(scale_img(Roi));
    Mat img = scale_img.clone();

    // Fill data to tensor
    Mat face_resize;
    resize(img, face_resize, Size(128, 128), 0, 0, INTER_LINEAR); // original method, Haar Cascade

    if (face_resize.channels() != 3){
        cout << "Invalid Channels" << endl;
        return faces; 
    }

    std::vector<float> in;
    in.resize(face_resize.rows * face_resize.cols * face_resize.channels());
    std::vector<float> dst_data;
    std::vector<Mat> bgrChannels(3);
    split(face_resize, bgrChannels);

    float mean_vals[3] = {104., 117., 123.};
    for(auto i = 0; i <  bgrChannels.size();i++){
        Mat img2;
        bgrChannels[i].convertTo(img2, CV_32F);
        Mat meanMat = Mat(1,face_resize.rows * face_resize.cols, CV_32F, mean_vals[i]);
        Mat channel = img2.reshape(1,1) - mean_vals[i] ;
        std::vector<float> data_ = std::vector<float>(channel);
        dst_data.insert(dst_data.end(),data_.begin(),data_.end());
    }


    for(size_t i = 0; i < in.size(); i++)
    {
        session_detect->interpreter->typed_input_tensor<float>(0)[i] = (dst_data[i]);
    }


    // Invoke
    ret = session_detect->interpreter->Invoke();
    if(ret)
    {   
        cout << "Cannot Invoke!" << endl;
        return faces;   
    }

    std::vector<PredBox> bbox_collection;
    float *score    =  (float*)session_detect->interpreter->typed_output_tensor<float>(0);//0
    float *landmark =  (float*)session_detect->interpreter->typed_output_tensor<float>(1);//1
    float *bbox     =  (float*)session_detect->interpreter->typed_output_tensor<float>(2);//2

    CalBoxes(bbox_collection, score, landmark, bbox);
    NMS(bbox_collection, face_list);

    // Face alignment process
    const int image_orig_height = int(img.rows);
    const int image_orig_width = int(img.cols);
    float scale_x = image_orig_width / 128.0;
    float scale_y = image_orig_height / 128.0;
    float landmarks[5][2];

    Mat temp_img;

    for (int i = 0; i < int(face_list.size()); i++) {
        temp_img = img.clone();
        auto face = face_list[i];
        int x_min = int(scale_x * std::min(face.x1, face.x2));
        int y_min = int(scale_y * std::min(face.y1, face.y2));
        int x_max = int(scale_x * std::max(face.x1, face.x2));
        int y_max = int(scale_y * std::max(face.y1, face.y2));
        x_min = (std::min)((std::max)(x_min, 0), image_orig_width - 1);
        x_max = (std::min)((std::max)(x_max, 0), image_orig_width - 1);
        y_min = (std::min)((std::max)(y_min, 0), image_orig_height - 1);
        y_max = (std::min)((std::max)(y_max, 0), image_orig_height - 1);

        //test align
        for (int j = 0; j<5; j++){
            landmarks[j][0]= scale_x * face.key_points[j].first;
            landmarks[j][1]= scale_y * face.key_points[j].second;

        }
        auto align_mat = face_Alignment(temp_img, landmarks);
        faces.push_back(align_mat);
    }
    return faces;
}


// Face Recognition
void Face_Rec::init_rec(Recognition_session *session, string model_dir)
{

    // Load model
    session->model = tflite::FlatBufferModel::BuildFromFile(model_dir.c_str());


    if (!session->model)
    {

        std::cout << "Invalid to load recognition model!" << std::endl;
    }

    tflite::InterpreterBuilder(*session->model, session->resolver)(&session->interpreter);
    if (!session->interpreter)
    {
        std::cout << "Invalid to construct interpreter! (Recognition model)" << std::endl;
    }

    if(true)
    {
      session->interpreter->SetNumThreads(6);
    }

    // resize tensors
    if(session->interpreter->AllocateTensors() != kTfLiteOk)
    {
        std::cout << "Invalid to allocate tensors! (Recognition model)" << std::endl;
    }

    // get input(0) size
    int input = session->interpreter->inputs()[0];
    TfLiteIntArray* input_dims = session->interpreter->tensor(input)->dims;

    session->i_height = input_dims->data[1];  // height
    session->i_width = input_dims->data[2];   // width
    session->i_channels = input_dims->data[3]; // channels ?
    // get output(0) size
    int output = session->interpreter->outputs()[0];
    TfLiteIntArray* output_dims = session->interpreter->tensor(output)->dims;
    session->o_size = output_dims->data[1];
}


int Face_Rec::inference_rec(Mat &face_in, std::vector<float> &face_feature, Recognition_session *session)
{
    int ret = 0;

    // check image source
    if (face_in.empty()){
        cout << "Failed to input Mat!" << endl;
        return -1;
    }

    // resize the face_in size according to NN input size.
    Mat face_resize;
    resize(face_in, face_resize, Size(in_h, in_w), 0, 0, INTER_LINEAR);

    // check if input is rgb
    if(face_resize.channels() != 3)
    {
        cout << "Invalid channels!" << endl;
        return -1;
    }

    // Input tensor
    std::vector<float> in;
    in.resize(face_resize.rows * face_resize.cols * face_resize.channels());
    std::vector<float> dst_data;
    std::vector<Mat> bgrChannels(3);
    split(face_resize, bgrChannels);
    for(auto i = 0; i < bgrChannels.size();i++){
        std::vector<float> data_ = std::vector<float>(bgrChannels[i].reshape(1,1));
        dst_data.insert(dst_data.end(),data_.begin(),data_.end());
    }

    // fill data to tensor input(0)
    for(size_t i = 0; i < in.size(); i++)
    {
        session->interpreter->typed_input_tensor<float>(0)[i] = (dst_data[i]-127.5)/127.5;
    }

    // Invoke
    ret = session->interpreter->Invoke();

    if(ret)
    {
        return -1;
    }

    // read out and fill data to output vector
    face_feature.clear();
    face_feature.resize(session->o_size);
    for(int o_cnt = 0; o_cnt < session->o_size; o_cnt++)
    {
        face_feature[o_cnt] = session->interpreter->typed_output_tensor<float>(0)[o_cnt];
    }

    float norm = std::sqrt(std::accumulate(face_feature.begin(), face_feature.end(), 0.0f, [](float sum, float val) {
        return sum + val * val;
    }));
    
    for (float& value : face_feature) {
    value /= norm;
    }

    return ret;
}



// Function for face detection
void Face_Det::CalBoxes(std::vector<PredBox> &bbox_collection,
                        float *score_data,
                        float *landmark_data,
                        float *bbox_data) {

    for (int i = 0; i < num_priors; i++) {
        if (score_data[i * 2 +1] > score_threshold) {

            PredBox info;

            float x_center = bbox_data[i * 4] * center_variance * priors[i][2] + priors[i][0];
            float y_center = bbox_data[i * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
            float w        = exp(bbox_data[i * 4 + 2] * size_variance) * priors[i][2];
            float h        = exp(bbox_data[i * 4 + 3] * size_variance) * priors[i][3];
            info.x1    = clip(x_center - w / 2.0, 1) * 128;
            info.y1    = clip(y_center - h / 2.0, 1) * 128;
            info.x2    = clip(x_center + w / 2.0, 1) * 128;
            info.y2    = clip(y_center + h / 2.0, 1) * 128;
            info.score = clip(score_data[i * 2 + 1], 1);
            // key points
            int offset_keypoints = 5*2*i;
            for(int j=0; j<num_keypoints; j++) {
                float kp_x = (landmark_data[offset_keypoints+ j * 2 + 0]  * center_variance * priors[i][2] + priors[i][0])* 128;
                float kp_y = (landmark_data[offset_keypoints + j * 2 + 1] * center_variance * priors[i][3] + priors[i][1])* 128;

                info.key_points.push_back(std::make_pair(kp_x, kp_y));
            }
            bbox_collection.push_back(info);
        }
    }
}

void Face_Det::GenPriorBox(){

    std::vector<std::vector<float>> featuremap_size;
    std::vector<std::vector<float>> shrinkage_size;
    auto w_h_list = {in_w, in_h};

    for (auto size : w_h_list) {
        std::vector<float> fm_item;
        for (float stride : strides) {
            fm_item.push_back(ceil(size / stride));
        }
        featuremap_size.push_back(fm_item);
    }

    for (auto size : w_h_list) {
        shrinkage_size.push_back(strides);
    }

    /* generate prior anchors */
    for (int index = 0; index < num_featuremap; index++) {
        float scale_w = in_w / shrinkage_size[0][index];
        float scale_h = in_h / shrinkage_size[1][index];
        for (int j = 0; j < featuremap_size[1][index]; j++) {
            for (int i = 0; i < featuremap_size[0][index]; i++) {
                float x_center = (i + 0.5) / scale_w;
                float y_center = (j + 0.5) / scale_h;

                for (float k : min_boxes[index]) {
                    float w = k / in_w;
                    float h = k / in_h;
                    priors.push_back({clip(x_center, 1), clip(y_center, 1), clip(w, 1), clip(h, 1)});
                }
            }
        }
    }
}

void Face_Det::NMS(std::vector<PredBox> &input, std::vector<PredBox> &output) {
    std::sort(input.begin(), input.end(), [](const PredBox &a, const PredBox &b) { return a.score > b.score; });
    output.clear();

    int box_num = input.size();

    std::vector<int> merged(box_num, 0);

    for (int i = 0; i < box_num; i++) {
        if (merged[i])
            continue;
        std::vector<PredBox> buf;

        buf.push_back(input[i]);
        merged[i] = 1;

        float h0 = input[i].y2 - input[i].y1 + 1;
        float w0 = input[i].x2 - input[i].x1 + 1;

        float area0 = h0 * w0;

        for (int j = i + 1; j < box_num; j++) {
            if (merged[j])
                continue;

            float inner_x0 = input[i].x1 > input[j].x1 ? input[i].x1 : input[j].x1;
            float inner_y0 = input[i].y1 > input[j].y1 ? input[i].y1 : input[j].y1;

            float inner_x1 = input[i].x2 < input[j].x2 ? input[i].x2 : input[j].x2;
            float inner_y1 = input[i].y2 < input[j].y2 ? input[i].y2 : input[j].y2;

            float inner_h = inner_y1 - inner_y0 + 1;
            float inner_w = inner_x1 - inner_x0 + 1;

            if (inner_h <= 0 || inner_w <= 0)
                continue;

            float inner_area = inner_h * inner_w;

            float h1 = input[j].y2 - input[j].y1 + 1;
            float w1 = input[j].x2 - input[j].x1 + 1;

            float area1 = h1 * w1;

            float score;

            score = inner_area / (area0 + area1 - inner_area);

            if (score > iou_threshold) {
                merged[j] = 1;
                buf.push_back(input[j]);
            }
        }
        output.push_back(buf[0]);
    }
}

Mat Face_Det::face_Alignment(Mat Img, float landmarks[5][2]){
    Mat result;
    Mat src(5,2,CV_32FC1, points_dst);
    memcpy(src.data, points_dst, 2 * 5 * sizeof(float));
    Mat dst(5,2,CV_32FC1, landmarks);
    memcpy(dst.data, landmarks, 2 * 5 * sizeof(float));

    Mat M = similar_Transform(dst, src);  // skimage.transform.SimilarityTransform
    warpPerspective(Img, result, M, Size(112, 112));

    return result;
}

Mat Face_Det::similar_Transform(Mat src, Mat dst) {
    int num = src.rows;
    int dim = src.cols;
    Mat src_mean = mean_Axis(src);
    Mat dst_mean = mean_Axis(dst);
    Mat src_demean = element_Wise_Minus(src, src_mean);
    Mat dst_demean = element_Wise_Minus(dst, dst_mean);
    Mat A = (dst_demean.t() * src_demean) / static_cast<float>(num);
    Mat d(dim, 1, CV_32F);
    d.setTo(1.0f);
    if (determinant(A) < 0) {
        d.at<float>(dim - 1, 0) = -1;

    }
    Mat T = Mat::eye(dim + 1, dim + 1, CV_32F);
    Mat U, S, V;
    SVD::compute(A, S,U, V);

    // the SVD function in opencv differ from scipy .
    int rank = matrix_Rank(A);
    if (rank == 0) {
        assert(rank == 0);

    } else if (rank == dim - 1) {
        if (determinant(U) * determinant(V) > 0) {
            T.rowRange(0, dim).colRange(0, dim) = U * V;
        } else {
            int s = d.at<float>(dim - 1, 0) = -1;
            d.at<float>(dim - 1, 0) = -1;

            T.rowRange(0, dim).colRange(0, dim) = U * V;
            Mat diag_ = Mat::diag(d);
            Mat twp = diag_*V; //np.dot(np.diag(d), V.T)
    Mat B = Mat::zeros(3, 3, CV_8UC1);
    Mat C = B.diag(0);
            T.rowRange(0, dim).colRange(0, dim) = U* twp;
            d.at<float>(dim - 1, 0) = s;
        }
    }
    else{
        Mat diag_ = Mat::diag(d);
        Mat twp = diag_*V.t(); //np.dot(np.diag(d), V.T)
        Mat res = U* twp; // U
        T.rowRange(0, dim).colRange(0, dim) = -U.t()* twp;
    }
    Mat var_ = varAxis0(src_demean);
    float val = sum(var_).val[0];
    Mat res;
    multiply(d,S,res);
    float scale =  1.0 / val * sum(res).val[0];
    T.rowRange(0, dim).colRange(0, dim) = - T.rowRange(0, dim).colRange(0, dim).t();
    Mat  temp1 = T.rowRange(0, dim).colRange(0, dim); // T[:dim, :dim]
    Mat  temp2 = src_mean.t(); //src_mean.T
    Mat  temp3 = temp1*temp2; // np.dot(T[:dim, :dim], src_mean.T)
    Mat temp4 = scale*temp3;
    T.rowRange(0, dim).colRange(dim, dim+1)=  -(temp4 - dst_mean.t()) ;
    T.rowRange(0, dim).colRange(0, dim) *= scale;
    return T;
}

Mat Face_Det::element_Wise_Minus(const Mat &A,const Mat &B)
{
    Mat output(A.rows,A.cols,A.type());

    assert(B.cols == A.cols);
    if(B.cols == A.cols)
    {
        for(int i = 0 ; i <  A.rows; i ++)
        {
            for(int j = 0 ; j < B.cols; j++)
            {
                output.at<float>(i,j) = A.at<float>(i,j) - B.at<float>(0,j);
            }
        }
    }
    return output;
}
Mat Face_Det::mean_Axis(const Mat &src)
{
    int num = src.rows;
    int dim = src.cols;

    // x1 y1
    // x2 y2

    Mat output(1,dim,CV_32F);
    for(int i = 0 ; i <  dim; i ++)
    {
        float sum = 0 ;
        for(int j = 0 ; j < num ; j++)
        {
            sum+=src.at<float>(j,i);
        }
        output.at<float>(0,i) = sum/num;
    }

    return output;
}
Mat Face_Det::varAxis0(const Mat &src)
{
    auto tmp1 = mean_Axis(src);
    Mat temp_ = element_Wise_Minus(src,tmp1);
    multiply(temp_ ,temp_ ,temp_ );
    return mean_Axis(temp_);

}
int Face_Det::matrix_Rank(Mat M)
{
    Mat w, u, vt;
    SVD::compute(M, w, u, vt);
    Mat1b nonZeroSingularValues = w > 0.0001;
    int rank = countNonZero(nonZeroSingularValues);
    return rank;

}