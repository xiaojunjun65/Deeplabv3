#include <fstream>
#include <iostream>
#include <gflags/gflags.h>
#include<glog/logging.h>
#include<vector>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cnrt.h"
#include "u2net_offline.h"
#include <dirent.h>
using namespace std;
using namespace cv;
void Data_obj::setFilename(char* input_filename, char* output_filename){
    in_filename = input_filename;
    save_filename = output_filename;
}
//softmax操作
void softmax(float * data,int h,int w,int num_classes)
{
   for(int y=0;y<h;++y)
   {
    for(int x=0;x<w;++x)
    {
      float sum = 0.0f;
      for (int c=0;c<num_classes;++c)
      {
        sum += exp(data[y*w*num_classes+x*num_classes+c]);

      }
      for(int c=0;c<num_classes;++c)
      {
        data[y*w*num_classes+x*num_classes+c]=exp(data[y*w*num_classes+x*num_classes+c]) /sum;
      }
    }
   }
}
//数据预处理函数
Mat Data_obj::preProcess(int height , int width) {
    //读取图片为矩阵
    cv::Mat image = cv::imread(in_filename, cv::IMREAD_COLOR);
    
    cv::cvtColor(image,image,cv::COLOR_BGR2RGB);
    realImg_width = image.cols;
    realImg_height = image.rows;
    //图像缩放
    cv::resize(image, image, cv::Size(width, height));
    //将图像转为浮点型
    cv::Mat normalized_image;
    image.convertTo(image, CV_32FC2);
    cv::Mat nomalimg;
    vector<cv::Mat>channels;
    cv::split(image,channels);
    for (int i =0;i<3;i++)
    {
      cv::Mat norch = channels[i] /255.0f;
      channels[i] = norch;
    }
    cv::merge(channels,normalized_image);
    // //标准化输入数据
    // cv::Mat subtract_image = cv::Mat(image.rows,image.cols,  CV_32FC3, cv::Scalar(0.485*255, 0.456*255, 0.406*255));
    // cv::subtract(image, subtract_image, subtract_image);
    // cv::Mat img_device = cv::Mat(image.rows, image.cols, CV_32FC3, cv::Scalar(0.229*255, 0.224*255, 0.225*255));
    // cv::divide(subtract_image , img_device, normalized_image);

    return normalized_image;
  }

//结果后处理函数
void Data_obj::resultDeal(float*result ){
 cv::Mat result_(cv::Size(512, 512), CV_8UC1);
  softmax(result,512,512,2);
int background =0;
int building =0;
for(int i =0;i<512*512;i++)
{
  float result0 = result[i*2];
  float result1 = result[i*2+1];
  if(result0>result1)
  {
    result_.at<uchar>(i/512,i%512)=0;
    background++;
  }
  else{
    result_.at<uchar>(i/512,i%512)=255;
    building++;
  }
}
cout<<"background:"<<background<<endl;
  cout<<"building:"<<building<<endl;
   cv::Mat image = cv::imread(in_filename, cv::IMREAD_COLOR);
   cv::resize(image,image,cv::Size(512,512));
  cv::Mat jiajie = result_.clone();
  cv::cvtColor(jiajie,jiajie,cv::COLOR_GRAY2BGR);
  cv::Mat result_1 = cv::Mat::zeros(cv::Size(512,512),image.type());
  image.copyTo(result_1,jiajie);
cv::imwrite(save_filename,result_1);
}

Runtime_obj::Runtime_obj(char* model_path){
  cnrtInit(0);
  // 加载模型
  cnrtLoadModel(&model, model_path);
  // 从模型中取得function
  cnrtCreateFunction(&function);
  cnrtExtractFunction(&function, model, "subnet0");
  // 获取输入输出参数
  cnrtGetInputDataSize(&inputSizeS, &inputNum, function);
  cnrtGetOutputDataSize(&outputSizeS, &outputNum, function);
  // 分配CPU端的内存指针
  inputCpuPtrS = (void **)malloc(inputNum * sizeof(void *));
  outputCpuPtrS = (void **)malloc(outputNum * sizeof(void *));
  // 分配MLU端的内存指针
  inputMluPtrS = (void **)malloc(inputNum * sizeof(void *));
  outputMluPtrS = (void **)malloc(outputNum * sizeof(void *));
  // 分配输入内存 inputNum=1
  for (int i = 0; i < inputNum; i++) {
    // 分配CPU端的输入内存
    inputCpuPtrS[i] = malloc(inputSizeS[i]);
    cnrtMalloc(&(inputMluPtrS[i]), inputSizeS[i]);
  }
  // 分配输出的内存  outputNum=7
  for (int i = 0; i < outputNum; i++) {   
    // 分配CPU端的输出内存
    outputCpuPtrS[i] = malloc(outputSizeS[i]);
    // 分配MLU端的输出内存
    cnrtMalloc(&(outputMluPtrS[i]), outputSizeS[i]);
  }
  // 创建RuntimeContext
  cnrtCreateRuntimeContext(&ctx, function, NULL);
  // 设置当前使用的设备
  cnrtSetRuntimeContextDeviceId(ctx, 0);
  // 初始化
  cnrtInitRuntimeContext(ctx, NULL);
  cnrtRuntimeContextCreateQueue(ctx, &queue);
  param = (void **)malloc(sizeof(void *) * (inputNum + outputNum));
}


//离线推理函数
void Runtime_obj::Run(cv::Mat img_mat){
    // 填充输入数据
	  memcpy((void *)inputCpuPtrS[0],img_mat.data, inputSizeS[0]);
	  // 从CPU端的内存复制到MLU端的内存
    cnrtMemcpy(inputMluPtrS[0], inputCpuPtrS[0], inputSizeS[0], CNRT_MEM_TRANS_DIR_HOST2DEV);
    // 准备调用cnrtInvokeRuntimeContext_V2时的param参数
    for (int i = 0; i < inputNum; ++i) {
      param[i] = inputMluPtrS[i];
    }
    for (int i = 0; i < outputNum; ++i) {
      param[inputNum + i] = outputMluPtrS[i];
    }
    // 进行计算
    cnrtInvokeRuntimeContext_V2(ctx, NULL, param, queue, NULL);
    // 等待执行完毕
    cnrtSyncQueue(queue);
    // 取回数据
    for (int i = 0; i < 1; i++) {
    cnrtMemcpy(outputCpuPtrS[i], outputMluPtrS[i], outputSizeS[i], CNRT_MEM_TRANS_DIR_DEV2HOST);
    }
}


//run函数作用：分配内存，调用数据预处理函数，进行离线推理
void  Runtime_obj::destroy_all(){
  // 释放内存
  for (int i = 0; i < inputNum; i++) {
    free(inputCpuPtrS[i]);
    cnrtFree(inputMluPtrS[i]);
  }
  for (int i = 0; i < outputNum; i++) {
    free(outputCpuPtrS[i]);
    cnrtFree(outputMluPtrS[i]);
  } 
  free(inputCpuPtrS);
  free(outputCpuPtrS);
  free(param);
  // 销毁资源
  cnrtDestroyQueue(queue);
  cnrtDestroyRuntimeContext(ctx);
  cnrtDestroyFunction(function);
  cnrtUnloadModel(model);
  cnrtDestroy();
}


//获取文件夹下图像文件的路径
bool get_filelist_from_dir(string _path, vector<string>& _files)
{

	DIR* dir;	
	dir = opendir(_path.c_str());
	struct dirent* ptr;
	vector<string> file;
	while((ptr = readdir(dir)) != NULL)
	{
		if(ptr->d_name[0] == '.') {continue;}
		file.push_back(ptr->d_name);
	}
	closedir(dir);
	sort(file.begin(), file.end());
	_files = file;
  return true;
}

//接收程序参数
DEFINE_string(offmodel,"../model/quant_model.cambricon","offline model path");
DEFINE_string(input_img_path,"../../test_data/test","input image path");
DEFINE_string(save_path,"../","save segment image path");
DEFINE_int32(width,512,"picture width");
DEFINE_int32(height,512,"picture height");
int main(int argc, char* argv[]) { 
    google::InitGoogleLogging(argv[0]);
    gflags::ParseCommandLineFlags(&argc,&argv,true);
    clock_t start,finish;   //记录开始时间和结束时间; 
    double totaltime;   
    start=clock();          //clock():确定处理器当前时间   
    vector<string> files; 
    get_filelist_from_dir(FLAGS_input_img_path, files);
    
    //网络输入的宽高
    int width = FLAGS_width;
    int height = FLAGS_height;
    Data_obj *data_obj = new Data_obj();
    Runtime_obj *run_obj = new Runtime_obj(const_cast<char *>(FLAGS_offmodel.c_str()));
    int size = files.size();
    
    for (int i = 0;i < size;i++)  
    {
      //输入图像文件路径
      string input_path = FLAGS_input_img_path + "/"+ files[i];
      //保存图像文件路径
      string save_path = FLAGS_save_path + "/"+ files[i];
      //设置输入图像路径以及保存图像路径
      data_obj->setFilename(const_cast<char *>(input_path.c_str()), const_cast<char *>(save_path.c_str()));
      cv::Mat img_processed = data_obj->preProcess(height,width);
      // cout<<cv::format(img_processed,cv::Formatter::FMT_DEFAULT)<<endl;
      // //运行代码
      run_obj->Run(img_processed);
      data_obj->resultDeal((float*)run_obj->outputCpuPtrS[0]);
    }
    run_obj->destroy_all();
    finish=clock();   
    totaltime=(double)(finish-start)/CLOCKS_PER_SEC; 
    cout<<"time:"<<totaltime<<endl;
    return 0;
}
