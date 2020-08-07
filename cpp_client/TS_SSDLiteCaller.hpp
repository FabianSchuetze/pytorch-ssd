#ifndef TS_SSDLiteCaller_hpp
#define TS_SSDLiteCaller_hpp
#define LIBRARY_EXPORTS

#include <ATen/core/ivalue.h>
#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>
//#ifdef _WIN32
//#    ifdef LIBRARY_EXPORTS
//#        define LIBRARY_API __declspec(dllexport)
//#    else
//#        define LIBRARY_API __declspec(dllimport)
//#    endif
//#elif
//#    define LIBRARY_API
//#endif
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Export.hpp"

#include "DataProcessing.hpp"

class TS_SSDLiteCaller {
   private:
    torch::jit::script::Module model;
    PreProcessing preprocess;
    PostProcessing detection;

   public:
    TS_SSDLiteCaller() = delete;
    LIBRARY_API TS_SSDLiteCaller(const std::string&, const std::string&);
    LIBRARY_API void predict(const cv::Mat&, std::vector<Landmark>&);

   private:
    void derserialize_model(const std::string&, const std::string&);
};

#endif
