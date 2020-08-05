#ifndef evaluate_hpp
#define evaluate_hpp

#ifdef _WIN32
#    ifdef LIBRARY_EXPORTS
#        define LIBRARY_API __declspec(dllexport)
#    else
#        define LIBRARY_API __declspec(dllimport)
#    endif
#elif
#    define LIBRARY_API
#endif

#include <string>
#include <vector>

#include "DataProcessing.hpp"
typedef std::vector<Landmark> Landmarks;
typedef struct {
    float precision;
    float recall;
} result;
LIBRARY_API result eval_result(const std::vector<Landmarks>&, const std::vector<Landmarks>&);
float caclulate_iou(const Landmark&, const Landmark&);
#endif
