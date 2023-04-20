#include <vector>
#include <string>
#include <memory>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "cnrt.h"
#include<cmath>

class Data_obj{
    public:
        cv::Mat preProcess(int height , int width);
        void setFilename(char* in_filename, char* save_filename);
        void resultDeal(float*result);
    private:
        //图像路径名
       char *in_filename, *save_filename;
       //真实图像的高、宽
       int realImg_height, realImg_width;
};

class Runtime_obj{
    public:
        Runtime_obj(char* model_path);
        void Run(cv::Mat img_mat);
        void destroy_all();
        cnrtModel_t model;
        cnrtFunction_t function;
        cnrtRuntimeContext_t ctx;
        cnrtQueue_t queue;
        //输入、输出数量
        int inputNum, outputNum;
        //输入、输出内存大小
        int64_t *inputSizeS, *outputSizeS; 
        //CPU、MLU端输入、输出指针
        void **inputCpuPtrS,**outputCpuPtrS,**inputMluPtrS, **outputMluPtrS,**param;
};

