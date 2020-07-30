#include "Evaluate.hpp"

#include <algorithm>
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
float caclulate_iou(const Landmark& a, const Landmark& b) {
    float xA = std::max(a.xmin, b.xmin);
    float xB = std::min(a.xmax, b.xmax);
    float yA = std::max(a.ymin, b.ymin);
    float yB = std::min(a.ymax, b.ymax);
    float low = 0.;
    float intersection = std::max((xB - xA),low) * std::max((yB - yA), low);
    float area_union = area(a) + area(b) - intersection;
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
            if (iou > 0.5) correct++;
        }
        total++;
    }
}

void check_recall(const Landmarks& predictions, const Landmarks& gts,
                     int& correct, int& total) {
    for (const auto gt : gts) {
        auto res = std::find_if(predictions.begin(), predictions.end(),
                                [&gt](const Landmark& x) {
                                    return x.label == gt.label;
                                });
        if (res != predictions.end()) {
            float iou = caclulate_iou(*res, gt);
            if (iou > 0.5) correct++;
        }
        total++;
    }
}
result eval_result(const vector<Landmarks>& predictions,
                  const vector<Landmarks>& gts,
                  const vector<std::string>& names) {
    result res;
    int precision_cor(0), precision_tot(0);
    int recall_cor(0), recall_tot(0);
    if (predictions.size() != gts.size()) {
        throw std::runtime_error("Predictions and Gts have different lenght");
    }
    for (size_t i = 0; i < predictions.size(); ++i) {
        check_precision(predictions[i], gts[i], precision_cor, precision_tot);
        check_recall(predictions[i], gts[i], recall_cor, recall_tot);
    }
    res.precision = (float) precision_cor / precision_tot;
    res.recall = (float) recall_cor / recall_tot;
    return res;
}
