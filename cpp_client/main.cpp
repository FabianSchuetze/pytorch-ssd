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
#include "TS_SSDLiteCaller.hpp"

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

//void serialize_results(const std::string& file,
                       //const std::vector<PostProcessing::Landmark>& result) {
    //std::ofstream myfile;
    //size_t pos = file.find_last_of("/");
    //std::string filename = file.substr(pos + 1);
    //size_t pos_end = filename.find(".");
    //std::string token = filename.substr(0, pos_end);
    //std::string outfile = "results/" + token + ".result";
    //myfile.open(outfile, std::ios::trunc);
    //if (myfile.fail()) {
        //std::cout << "couldnt open file: " << outfile << std::endl;
    //} else {
        //for (const PostProcessing::Landmark& res : result) {
            //float xmin = res.xmin;
            //float ymin = res.ymin;
            //float xmax = res.xmax;
            //float ymax = res.ymax;
            //myfile << xmin << ", " << ymin << ", " << xmax << ", " << ymax
                   //<< ", " << res.confidence << ", " << res.label << std::endl;
        //}
    //}
    //myfile.close();
//}

int main(int argc, const char* argv[]) {
    if (argc != 4) {
        std::cerr << "usage: example-app <path-to-exported-script-module> "
                     "<path-to-config> < path-to-image-folder>\n";
        return -1;
    }
    TS_SSDLiteCaller SSDLite(argv[1], argv[2]);
    std::string path = argv[3];
    std::vector<std::string> files = load_images(path);
    // should have vector with class, contain, name results and gt
    int count(0);
    float total_durations(0.);
    for (const std::string& img : files) {
        cv::Mat tmp;
        try {
            tmp = cv::imread(img, cv::IMREAD_COLOR);
        } catch (...) {
            std::cout << "couldnt read img " << img << "; continue\n ";
            continue;
        }
        std::vector<Landmark> result;
        auto start = std::chrono::high_resolution_clock::now();
        SSDLite.predict(tmp, result);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        total_durations += duration.count();
        count++;
        serialize_results(img, result);
    }
    std::cout << "finished " << count << " images in " << total_durations / 1000
              << " seconds; fps: " << count / (total_durations / 1000)
              << std::endl;
}
