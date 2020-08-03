#ifndef evaluate_hpp
#define evaluate_hpp
#include <string>
#include <vector>

#include "DataProcessing.hpp"
typedef std::vector<Landmark> Landmarks;
typedef struct {
    float precision;
    float recall;
} result;
result eval_result(const std::vector<Landmarks>&, const std::vector<Landmarks>&);
float caclulate_iou(const Landmark&, const Landmark&);
#endif
