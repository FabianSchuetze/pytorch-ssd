#include "TS_SSDLiteCaller.hpp"

#include <exception>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
using namespace cv;  // ugly - how to get COLOR_BGR2RGB?

void TS_SSDLiteCaller::derserialize_model(const std::string& model_pth,
                                          const std::string& config) {
    try {
        model = torch::jit::load(model_pth);
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
    torch::Device device(torch::kCPU);
    model.to(device);  // put it on CPU
    std::cout << config << std::endl;
    preprocess = PreProcessing(config);
    detection = PostProcessing(config);
}

TS_SSDLiteCaller::TS_SSDLiteCaller(const std::string& model_pth,
                                   const std::string& config)
    : model(), preprocess(), detection() {
    derserialize_model(model_pth, config);
}

void TS_SSDLiteCaller::predict(const cv::Mat& input,
                               std::vector<Landmark>& results) {
    cv::Mat image;
    cv::cvtColor(input, image, COLOR_BGR2RGB);
    int height = image.size().height;
    int width = image.size().width;
    std::pair<float, float> size = std::make_pair(height, width);
    torch::Tensor tensor_image = preprocess.process(image);
    std::vector<torch::jit::IValue> inputs{tensor_image};
    auto outputs = model.forward(inputs).toTuple();
    results = detection.process(outputs->elements()[0].toTensor(),
                                outputs->elements()[1].toTensor(), size);
}
