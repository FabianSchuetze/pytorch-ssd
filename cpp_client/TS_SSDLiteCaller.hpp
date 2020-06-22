#ifndef TS_SSDLiteCaller_hpp
#define TS_SSDLiteCaller_hpp
#include <ATen/core/ivalue.h>
#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>

//#include <chrono>
//#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "DataProcessing.hpp"

class TS_SSDLiteCaller {
   private:
    torch::jit::script::Module model;
    PreProcessing preprocess;
    PostProcessing detection;

   public:
    TS_SSDLiteCaller() = delete;
    TS_SSDLiteCaller(const std::string&, const std::string&);
    void predict(const cv::Mat&, std::vector<PostProcessing::Landmark>&);

   private:
    void derserialize_model(const std::string&, const std::string&);
};

#endif
