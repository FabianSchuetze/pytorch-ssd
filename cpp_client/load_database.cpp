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
#include "Database.hpp"
#include "TS_SSDLiteCaller.hpp"

using namespace cv;
using namespace std::chrono;

int main(int argc, const char* argv[]) {
    if (argc != 4) {
        std::cerr << "usage: example-app <path-to-exported-script-module> "
                     "<path-to-config> < path-to-image-folder>\n";
        return -1;
    }
    TS_SSDLiteCaller SSDLite(argv[1], argv[2]);
    std::string path = argv[3];
    Database database(path);
    int count(0);
    float total_durations(0.);
    std::cout << "start things" << std::endl;
    int start = 0;
    std::vector<std::vector<PostProcessing::Landmark>> gts, predictions;
    //std::vector<std::vector<PostProcessing::Landmark>> predictions;
    while (start < database.length) {
        start++;
        auto img = database.get_element();
        cv::Mat tmp;
        std::cout << "loading image " << img.first << std::endl;
        try {
            tmp = cv::imread(img.first, cv::IMREAD_COLOR);
        } catch (...) {
            std::cout << "couldnt read img " << img.first
                      << "; continue\n ";
            continue;
        }
        if (!tmp.data)  {
            std::cout << "couldnt read img " << img.first
                      << "; continue\n ";
            continue;
        }
        std::vector<PostProcessing::Landmark> result;
        auto start = std::chrono::high_resolution_clock::now();
        SSDLite.predict(tmp, result);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        total_durations += duration.count();
        predictions.push_back(result);
        gts.push_back(img.second);
        count++;
    }
    std::cout << "finished " << count << " images in " << total_durations / 1000
              << " seconds; fps: " << count / (total_durations / 1000)
              << std::endl;
}
