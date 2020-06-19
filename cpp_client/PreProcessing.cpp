#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "DataProcessing.hpp"
#include "LoadConfig.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
PreProcessing::PreProcessing(const std::string& config) : _size(0) {
    std::ifstream paramFile{config};
    std::map<std::string, std::string> params{
        std::istream_iterator<kv_pair>{paramFile},
        std::istream_iterator<kv_pair>{}};
    _size = std::stoi(params.at("size"));
}

torch::Tensor PreProcessing::process(const cv::Mat& img) {
    cv::Mat transformed;
    cv::Size sz(_size, _size);
    cv::resize(img, transformed, sz);
    at::Tensor tensor_img =
        torch::from_blob(transformed.data, {1, _size, _size, 3}, at::kByte);
    tensor_img = tensor_img.to(at::kFloat);
    tensor_img = tensor_img.permute({0, 3, 1, 2});
    tensor_img = tensor_img / 255;
    return tensor_img;
}
