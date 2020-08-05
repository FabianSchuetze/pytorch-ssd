#ifndef data_processing_hpp
#define data_processing_hpp

#ifdef _WIN32
#    ifdef LIBRARY_EXPORTS
#        define LIBRARY_API __declspec(dllexport)
#    else
#        define LIBRARY_API __declspec(dllimport)
#    endif
#elif
#    define LIBRARY_API
#endif

#include <torch/script.h>  // One-stop header.

#include <opencv2/core/core.hpp>

struct Landmark {
    float xmin, xmax, ymin, ymax, confidence;
    int label;
};

LIBRARY_API void serialize_results(const std::string&, const std::vector<Landmark>&);

struct kv_pair : public std::pair<std::string, std::string> {
    friend std::istream& operator>>(std::istream& in, kv_pair& p) {
        return in >> std::get<0>(p) >> std::get<1>(p);
    }
};

class PostProcessing {
   public:
    PostProcessing() = default;
    PostProcessing(const std::string&);
    std::vector<Landmark> process(const torch::Tensor& scores,
                                  const torch::Tensor& boxes,
                                  std::pair<float, float> const&);

   private:
    void convert(int, const torch::Tensor&, const torch::Tensor&,
                 const std::pair<float, float>&, std::vector<Landmark>&);
    void print_arguments();

    int _num_classes, _bkg_label;
    float _conf_thresh, _nms_thresh;
};

class PreProcessing {
   public:
    PreProcessing() = default;
    PreProcessing(const std::string&);
    torch::Tensor process(const cv::Mat&);

   private:
    int _size;
};
#endif
