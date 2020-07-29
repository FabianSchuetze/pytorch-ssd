#ifndef evaluate_hpp
#define evaluate_hpp
#include <vector>
#include "DataProcessing.hpp"
typedef std::vector<PostProcessing::Landmark> Landmarks;
float eval_result(const std::vector<Landmarks>&,  const std::vector<Landmarks>&);
#endif
