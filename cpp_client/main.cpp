#include <ATen/core/ivalue.h>
#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "DataProcessing.hpp"

namespace fs = std::filesystem;
using namespace cv;
using namespace std::chrono;

std::vector<std::string> load_images(const std::string& path) {
    std::vector<std::string> files;
    for (const auto& img : fs::recursive_directory_iterator(path)) {
        files.push_back(img.path());
    }
    return files;
}

void serialize_results(const std::string& file,
                       const std::vector<PostProcessing::Landmark>& result) {
    std::ofstream myfile;
    size_t pos = file.find_last_of("/");
    std::string filename = file.substr(pos + 1);
    size_t pos_end = filename.find(".");
    std::string token = filename.substr(0, pos_end);
    std::string outfile = "results/" + token + ".result";
    myfile.open(outfile, std::ios::trunc);
    if (myfile.fail()) {
        std::cout << "couldnt open file: " << outfile << std::endl;
    } else {
        for (const PostProcessing::Landmark& res : result) {
            float xmin = res.xmin;
            float ymin = res.ymin;
            float xmax = res.xmax;
            float ymax = res.ymax;
            myfile << xmin << ", " << ymin << ", " << xmax << ", " << ymax
                   << ", " << res.confidence << ", " << res.label << std::endl;
        }
    }
    myfile.close();
}

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    } catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    std::string config =
        "/home/fabian/Documents/work/github/ssd.pytorch/cpp_client/params.txt";
    // std::string path =
    //"/home/fabian/data/TS/CrossCalibration/ImageTCL/test/";
     std::string path = "/home/fabian/data/grayscale/";
     std::vector<std::string> files = load_images(path);
    PostProcessing detection(config);
    PreProcessing preprocess(config);
    std::vector<torch::jit::IValue> inputs(1);
    //std::vector<std::string> files = {
        //"/home/fabian/data/TS/CrossCalibration/ImageTCL/greyscale/"
        //"2000-01-01_109790322_121528_PCL_003754.png"};
    for (const std::string& img : files) {
        cv::Mat image;
        try {
            cv::Mat tmp = cv::imread(img, cv::IMREAD_COLOR);
            cv::cvtColor(tmp, image, COLOR_BGR2RGB);
        } catch (...) {
            std::cout << "couldnt read img " << img << "; continue\n ";
            continue;
        }
        int height = image.size().height;
        int width = image.size().width;
        std::pair<float, float> size = std::make_pair(height, width);
        torch::Tensor tensor_image = preprocess.process(image);
        //std::cout << tensor_image.slice({1, 1, } << std::endl;

        torch::Device device(torch::kCPU);
        module.to(device);
        inputs[0] = tensor_image;
        auto start = high_resolution_clock::now();
        auto outputs = module.forward(inputs).toTuple();
        torch::Tensor prediction = outputs->elements()[1].toTensor();
        std::vector<PostProcessing::Landmark> result =
            detection.process(outputs->elements()[0].toTensor(),
                              outputs->elements()[1].toTensor(), size);
        // priors.attr("priors").toTensor(), size);
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        std::cout << "finished " << img << " in " << duration.count()
                  << std::endl;
        serialize_results(img, result);
    }
}
