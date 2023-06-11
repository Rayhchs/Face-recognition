#include "utils.h"
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <sqlite3.h>

using namespace std;
using namespace Utils;


int DB_tools::Create_Table(sqlite3* db){
    int result_create = sqlite3_exec(db, "CREATE TABLE USERS(id, feature)", NULL, NULL, NULL);
    if (result_create != SQLITE_OK) {
        cout << "Something Wrong when creating table!" << endl;
        return 1;
    }

    return 0;
}

int DB_tools::Insert(sqlite3* db, int id, string feat_str){
    sqlite3_stmt *stmt;
    const char *tail;
    string sql = "INSERT INTO USERS (id, feature) VALUES (?, ?)";
    int rc = sqlite3_prepare_v2(db, sql.c_str(), -1, &stmt, &tail);
    if (rc == SQLITE_OK) {
        sqlite3_bind_int(stmt, 1, id);
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


int DB_tools::callback(void* data, int argc, char** argv, char** azColName) {

    QueryData* querydata = static_cast<QueryData*>(data);

    // 處理每一行的結果
    std::vector<float> row0;
    std::vector<std::string> row1;
    for (int i = 0; i < argc; i++) {
        if (i==0)
        {
            row1.push_back(argv[i] ? argv[i] : "NULL");
        }
        else if (i==1)
        {
            std::string valueString = argv[i] ? argv[i] : "NULL";
            std::istringstream iss(valueString);
            std::string singleValue;
            while (std::getline(iss, singleValue, ' ')) {
                float floatValue = std::stof(singleValue);
                row0.push_back(floatValue);
            }
        }
    }
    querydata->qdata.push_back(row0);
    querydata->q_id.push_back(row1);

    return 0;
}


int DB_tools::Query(sqlite3* db, QueryData& querydata) {

    char* sql = "SELECT * FROM USERS";
    char* errMsg = nullptr;

    // 執行查詢語句
    int rc = sqlite3_exec(db, sql, callback, &querydata, &errMsg);
    if (rc != SQLITE_OK) {
        std::cerr << "查詢出錯: " << errMsg << std::endl;
        sqlite3_free(errMsg);
        return 1;
    }
    return 0;
}


void Tools::Vec2Str(std::vector<float>& vec, std::string& str) {
    
    std::stringstream ss;
    for (auto& f : vec) {
        ss << f << " ";
    }

    str = ss.str();
}


Eigen::MatrixXf Tools::Vec2Eig(std::vector<std::vector<float>>& data){
    int n = data.size();
    int m = data[0].size();
    Eigen::MatrixXf matrix(n, m);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            matrix(i, j) = data[i][j];
        }
    }
    return matrix;
}