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
    std::vector<Database> dataset = read_xml_file(path);
    // should have vector with class, contain, name results and gt
    int count(0);
    float total_durations(0.);
    std::cout << "start things" << std::endl;
    for (Database& data : dataset) {
        cv::Mat tmp;
        std::cout << "loading image " << data.filename << std::endl;
        try {
            tmp = cv::imread(data.filename, cv::IMREAD_COLOR);
        } catch (...) {
            std::cout << "couldnt read img " << data.filename
                      << "; continue\n ";
            continue;
        }
        if (!tmp.data)  {
            std::cout << "couldnt read img " << data.filename
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
        count++;
        data.predictions = result;
        // serialize_results(img, result);
    }
    std::cout << "finished " << count << " images in " << total_durations / 1000
              << " seconds; fps: " << count / (total_durations / 1000)
              << std::endl;
}
