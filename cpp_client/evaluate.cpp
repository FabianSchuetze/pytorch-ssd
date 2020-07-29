#include "Evaluate.hpp"

#include <stdexcept>

using std::vector;
typedef PostProcessing::Landmark Landmark;

float area(const Landmark landmark) {
    float length = landmark.xmax - landmark.xmin;
    float height = landmark.ymax - landmark.ymin;
    return length * height;
}

void print_area(const Landmark landmark) {
    std::cout << "xmin, xmax, ymin, ymax: " << 
        landmark.xmin << ", " << landmark.xmax << ", " <<
        landmark.ymin << ", " << landmark.ymax << std::endl;
}
float caclulate_iou(const Landmark& prediction, const Landmark& gt) {
    std::cout << "area of prediction: "; print_area(prediction);
    std::cout << "area of gts: "; print_area(gt);
    float xA = std::max(prediction.xmin, gt.xmin);
    float xB = std::min(prediction.xmax, gt.xmax);
    float yA = std::max(prediction.ymin, gt.ymin);
    float yB = std::min(prediction.ymax, gt.ymax);
    float intersection = (xB - xA) * (yB - yA);
    float area_union = area(prediction) + area(gt) - intersection;
    return intersection / area_union;
}

void check_precision(const Landmarks& predictions, const Landmarks& gts,
                     int& correct, int& total) {
    for (const auto prediction : predictions) {
        auto res = std::find_if(gts.begin(), gts.end(),
                                [&prediction](const Landmark& x) {
                                    return x.label == prediction.label;
                                });
        if (res != gts.end()) {
            float iou = caclulate_iou(prediction, *res);
            std::cout << "The iou is " << iou << std::endl;
            if (iou > 0.5) correct++;
        }
        total++;
    }
}
float eval_result(const vector<Landmarks>& predictions,
                  const vector<Landmarks>& gts) {
    int correct(0), total(0);
    if (predictions.size() != gts.size()) {
        throw std::runtime_error("Predictions and Gts have different lenght");
    }
    for (size_t i = 0; i < predictions.size(); ++i) {
        check_precision(predictions[i], gts[i], correct, total);
    }
    return (float)correct / total;
}
