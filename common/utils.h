#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <sqlite3.h>

using namespace std;


namespace Utils{
struct QueryData
{
    std::vector<std::vector<float>> qdata;
    std::vector<std::vector<std::string>> q_id;
};

class DB_tools
{
public:
    int Create_Table(sqlite3* db);
    int Insert(sqlite3* db, int id, string feat_str);
    int Query(sqlite3* db, QueryData& querydata);

private:
    static int callback(void* data, int argc, char** argv, char** azColName);
};

class Tools
{
public:
    void Vec2Str(std::vector<float>& vec, std::string& str);
    Eigen::MatrixXf Vec2Eig(std::vector<std::vector<float>>& data);

};
}
#endif