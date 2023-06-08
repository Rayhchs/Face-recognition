#include "lib/face_AI.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp> 
#include <yaml-cpp/yaml.h>
#include <sqlite3.h>
#include <iostream>
#include <sstream>

using namespace std;
using namespace face_AI;
using namespace cv;



int Create_Table(sqlite3* db){
    int result_create = sqlite3_exec(db, "CREATE TABLE USERS(id, feature)", NULL, NULL, NULL);
    if (result_create != SQLITE_OK) {
        cout << "Something Wrong when creating table!" << endl;
        return 1;
    }

    return 0;
}

int Insert(sqlite3* db, int id, string feat_str){
    sqlite3_stmt *stmt;
    const char *tail;
    string sql = "INSERT INTO USERS (id, feature) VALUES (?, ?)";
    int rc = sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, &tail);
    if (rc == SQLITE_OK) {
        sqlite3_bind_int(stmt, 1, 1);
        sqlite3_bind_text(stmt, 2, feat_str.c_str(), -1, SQLITE_TRANSIENT);
        rc = sqlite3_step(stmt);
        if (rc != SQLITE_DONE) {
            cout << "Cannot Insert Data" << endl;
            return 1;
        }
        sqlite3_finalize(stmt);
    }

    return 0;
}


void Vec2Str(std::vector<float>& vec, std::string& str) {
    
    std::stringstream ss;
    for (auto& f : vec) {
        ss << f << " ";
    }

    str = ss.str();
}



int main(){

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
    int result_open = sqlite3_open("faces.db", &db);
    if (result_open != SQLITE_OK) {
        cout << "Something Wrong when opening database!" << endl;
        return 1;
    }
    Create_Table(db);

    // char *sql = "INSERT INTO USERS VALUES (0, '0.1 0.999 0.888');";
    // Insert(db, sql);

    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cout << "Cannot Open Camera" << std::endl;
        return -1;
    }

    cv::Mat frame;

    while (true) {
        cap.read(frame);

        if (frame.empty()) {
            std::cout << "Cannot Read file" << std::endl;
            break;
        }

        cv::imshow("Camera", frame);

        char key = cv::waitKey(1);

        if (key == 'q' || key == 'Q') {
            break;
        }
        if (key == 'r' || key == 'R') {

            // Inference Detection
            vector<PredBox> face_list;
            auto temp_img = face_det.inference_det(frame, &session_Detect, face_list);
            if (int(face_list.size()) <= 0){
                cout << "No Face!" << endl;
            }

            // Inference Recognition
            vector<float> feature;
            string feat_str;
            int result = face_rec.inference_rec(frame, feature, &session_Recognize);
            Vec2Str(feature, feat_str);
            Insert(db, 0, feat_str);

            // Insert face feature to database
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
