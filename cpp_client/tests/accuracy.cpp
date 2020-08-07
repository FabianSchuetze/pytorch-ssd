#include <ATen/core/ivalue.h>
#include <gtest/gtest.h>
#include <opencv2/core/types.hpp>
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
#include <vector>

#include "../DataProcessing.hpp"
#include "../Database.hpp"
#include "../Evaluate.hpp"
#include "../TS_SSDLiteCaller.hpp"

using namespace cv;
using namespace std::chrono;

cv::Mat load_image(const std::string file) {
    cv::Mat tmp;
    try {
        tmp = cv::imread(file, cv::IMREAD_COLOR);
    } catch (...) {
        std::cout << "couldnt read img " << file << "; continue\n ";
    }
    return tmp;
}


void insert_rectangle(cv::Mat& tmp, const std::vector<Landmark>& landmarks,
                      const cv::Scalar& color) {
    for (const Landmark& landmark: landmarks)
        rectangle(tmp, Point(landmark.xmin, landmark.ymin),
                       Point(landmark.xmax, landmark.ymax), color);
}
void plot_image(cv::Mat tmp, const std::vector<Landmark>& predictions,
                const std::vector<Landmark>& gts) {
    namedWindow( "Display window", WINDOW_AUTOSIZE );
    cv::Scalar green = cv::Scalar(0, 128, 0);
    cv::Scalar red = cv::Scalar(0, 0, 255);
    insert_rectangle(tmp, predictions, green);
    insert_rectangle(tmp, gts, red);
    imshow( "Display window", tmp);
    waitKey(0);
}

TEST(MYTEST, Accuracy) {
    std::string root = "C:\\Users\\fs\\pytorch-ssd\\";
    std::string model = root + "traced_ssdlite.pt";
    std::string params = root + "cpp_client/params.txt";
    std::string db = "C:\\Users\\fs\\Documents\\CrossCalibration\\TCLObjectDetectionDatabase\\greyscale.xml";
    TS_SSDLiteCaller SSDLite(model, params);
    Database database(db);
    int count(0);
    float total_durations(0.);
    std::vector<std::vector<Landmark>> gts, predictions;
    while (count < database.length) {
        auto img = database.get_element();
        cv::Mat tmp = load_image(img.first);
        if (!tmp.data) {
            std::cout << "couldnt read img " << img.first << "; continue\n ";
            continue;
        }
        std::vector<Landmark> landmarks;
        auto start = std::chrono::high_resolution_clock::now();
        SSDLite.predict(tmp, landmarks);
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        total_durations += duration.count();
        predictions.push_back(landmarks);
        gts.push_back(img.second);
        //plot_image(tmp, landmarks, img.second);
        count++;
    }
    float fps = (float) count / (total_durations / 1000);
    ASSERT_GT(fps, 30); // Conversative, should exceed 70 on most hardware
    result res = eval_result(predictions, gts);
    ASSERT_GT(res.precision, 0.8);
    ASSERT_GT(res.recall, 0.8);

}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
};
