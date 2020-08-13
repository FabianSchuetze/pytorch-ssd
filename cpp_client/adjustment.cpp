#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
cv::Mat load_image(const std::string file) {
    cv::Mat tmp;
    try {
        tmp = cv::imread(file, cv::IMREAD_COLOR);
    } catch (...) {
        std::cout << "couldnt read img " << file << "; continue\n ";
    }
    return tmp;
}

int8_t mean(const cv::Mat& img) {
    float sum = 0.;
    float n = 0;
    for (int r = 0; r < img.rows; ++r) {
        for (int c = 0; c < img.cols; ++c) {
            sum += img.at<int8_t>(r, c);
            n++;
        }
    }
    int8_t avg = static_cast<int8_t>(sum / n);
    return avg;
}

void replace_smaller(cv::Mat& v, int8_t avg) {
    avg = 58 - avg;
    int8_t lim = 255 - avg;
    for (int r = 0; r < v.rows; ++r) {
        for (int c = 0; c < v.cols; ++c) {
            const int8_t val = v.at<uint>(r, c);
            if (val > lim)
                v.at<int8_t>(r, c) = static_cast<int8_t>(255);
            else
                v.at<int8_t>(r, c) += avg;
        }
    }
}

void replace_larger(cv::Mat& v, int8_t avg) {
    avg = avg - 58;
    for (int r = 0; r < v.rows; ++r) {
        for (int c = 0; c < v.cols; ++c) {
            const int8_t val = v.at<uint>(r, c);
            if (val > avg)
                v.at<int8_t>(r, c) -= avg;
            else
                v.at<int8_t>(r, c) = 0;
        }
    }
}
cv::Mat adjust_brightness(const cv::Mat& img) {
    cv::Mat transformed, hsv[3], final_hsv, out;
    cv::cvtColor(img, transformed, cv::COLOR_BGR2HSV);
    cv::split(transformed, hsv);
    std::cout << transformed.type() << std::endl;
    std::cout << hsv[2].type() << std::endl;
    int8_t avg = mean(hsv[2]);
    if (avg < 58) {
        replace_smaller(hsv[2], avg);
    } else if (avg > 58) {
        replace_larger(hsv[2], avg);
    }
    std::vector<cv::Mat> channels = {hsv[0], hsv[1], hsv[2]};
    cv::merge(channels, final_hsv);
    cv::cvtColor(final_hsv, out, cv::COLOR_HSV2BGR);
    return out;
}

int main() {
    std::string path = "/home/fabian/data/TS/kids/01.png";
    cv::Mat img = load_image(path);
    namedWindow( "Display window", WINDOW_AUTOSIZE );
    cv::imshow("Display window", img);
    cv::waitKey(0);
    cv::Mat adjusted = adjust_brightness(img);
    //cv::imshow("Image", adjusted);
    //cv::waitKey(0);
}
