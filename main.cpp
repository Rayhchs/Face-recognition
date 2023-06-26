#include "common/face_AI.h"
#include "common/utils.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp> 
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <sqlite3.h>
#include <iostream>
#include <sstream>

using namespace std;
using namespace face_AI;
using namespace cv;
using namespace Utils;


int main()
{
    // Utils
    static DB_tools DT;
    static Tools tools;

    // Model Initialization
    YAML::Node config = YAML::LoadFile("config.yaml");
    static Face_Det face_det;
    static Face_Rec face_rec;
    Detection_session session_Detect;
    Recognition_session session_Recognize;
    face_det.init_det(&session_Detect, config["detection_model"].as<std::string>());
    face_rec.init_rec(&session_Recognize, config["recognition_model"].as<std::string>());

    // Database Initialization
    sqlite3 *db;
    int result_open = sqlite3_open("./database/faces.db", &db);
    if (result_open != SQLITE_OK) {
        cout << "Something Wrong when opening database!" << endl;
        return 1;
    }
    DT.Create_Table(db);
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cout << "Cannot Open Camera" << std::endl;
        return -1;
    }

    int Width = 640;
    int Height = 480;
    cap.set(cv::CAP_PROP_FRAME_WIDTH, Width);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, Height);

    cv::Mat frame;

    while (true) {
        cap.read(frame);

        if (frame.empty()) {
            std::cout << "Cannot Read file" << std::endl;
            break;
        }

        cv::imshow("Camera", frame);

        char key = cv::waitKey(1);

        if (key == 'q' || key == 'Q')
        {
            break;
        }
        if (key == 'r' || key == 'R')
        {
            // Inference Detection
            vector<PredBox> face_list;
            auto temp_img = face_det.inference_det(frame, &session_Detect, face_list);
            if (int(face_list.size()) <= 0){
                cout << "No Face!" << endl;
            }

            else{
                int user_id = 0;
                cout << "Type Your ID" << endl;
                cin >> user_id;
                // Inference Recognition
                vector<float> feature;
                string feat_str;
                int result = face_rec.inference_rec(frame, feature, &session_Recognize);

                tools.Vec2Str(feature, feat_str);
                DT.Insert(db, user_id, feat_str);
            }
        }
        if (key == 'i' || key == 'I')
        {

            // Inference Detection
            vector<PredBox> face_list;
            auto temp_img = face_det.inference_det(frame, &session_Detect, face_list);
            // cv::imwrite("test.jpg",temp_img[0]);
            if (int(face_list.size()) <= 0){
                cout << "No Face!" << endl;
            }

            else{

                // Inference Recognition
                std::vector<std::vector<float>> features;
                vector<float> feature;
                face_rec.inference_rec(frame, feature, &session_Recognize);
                features.push_back(feature);
                Eigen::MatrixXf feature_matrix = tools.Vec2Eig(features).transpose();

                // Query database
                QueryData querydata;
                DT.Query(db, querydata);
                if (querydata.qdata.size() == 0){
                    cout << "Database is empty" << endl;
                    return 1;
                }
                Eigen::MatrixXf database_matrix = tools.Vec2Eig(querydata.qdata);

                // Calculate similarity
                Eigen::MatrixXf similarity = database_matrix * feature_matrix;

                Eigen::Index maxRow, maxCol;
                double max = similarity.maxCoeff(&maxRow, &maxCol);
                if (max >= 0.67){;
                    cout << "HI! " << querydata.q_id[maxRow][0] << endl;
                }
                else{
                    cout << "Not in Database" << endl;
                }
            }
        }

    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
