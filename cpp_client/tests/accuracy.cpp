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
#include <gtest/gtest.h>

#include "../DataProcessing.hpp"
#include "../Database.hpp"
#include "../Evaluate.hpp"
#include "../TS_SSDLiteCaller.hpp"

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

using namespace cv;
using namespace std::chrono;

TEST(MYTEST, Accuracy) {
    std::cout << "begin test" << std::endl;
    std::string root = "/home/fabian/Documents/work/github/pytorch-ssd/";
    std::string model = root + "traced_quantized.pt";
    std::string params = root + "cpp_client/params.txt";
    std::string db =
        "/home/fabian/data/TS/CrossCalibration/TCLObjectDetectionDatabase/"
        "greyscale_test.xml";
    std::cout << "begin test1.5" << std::endl;
    TS_SSDLiteCaller SSDLite(model, params);
    Database database(db);
    std::cout << "begin test3" << std::endl;

    int count(0);
    float total_durations(0.);
    std::cout << "start things" << std::endl;
    int start = 0;
    std::vector<std::vector<PostProcessing::Landmark>> gts, predictions;
    std::vector<std::string> names;
    while (start < database.length) {
        start++;
        auto img = database.get_element();
        cv::Mat tmp;
        try {
            tmp = cv::imread(img.first, cv::IMREAD_COLOR);
        } catch (...) {
            std::cout << "couldnt read img " << img.first << "; continue\n ";
            continue;
        }
        if (!tmp.data) {
            std::cout << "couldnt read img " << img.first << "; continue\n ";
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
        serialize_results(img.first, result);
        names.push_back(img.first);
    }
    result res = eval_result(predictions, gts, names);
    ASSERT_GT(res.precision, 0.5);
    ASSERT_GT(res.recall, 0.5);

}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
};
