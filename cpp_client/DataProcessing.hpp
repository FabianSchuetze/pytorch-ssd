#ifndef data_processing_hpp
#define data_processing_hpp
#include <torch/script.h>  // One-stop header.

#include <opencv2/core/core.hpp>

// struct Landmark{
// float x
struct Landmark {
    float xmin, xmax, ymin, ymax, confidence;
    int label;
};

void serialize_results(const std::string&, const std::vector<Landmark>&);

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
    std::vector<float> _variances;
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
