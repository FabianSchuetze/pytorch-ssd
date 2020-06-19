#ifndef data_processing_hpp
#define data_processing_hpp
#include <torch/script.h>  // One-stop header.
#include <opencv2/core/core.hpp>

class PostProcessing {
   public:
    struct Landmark {
        float xmin, xmax, ymin, ymax, confidence;
        int label;
    };
    PostProcessing(const std::string&);
    std::vector<Landmark> process(const torch::Tensor& scores,
                                  const torch::Tensor& boxes,
                                  std::pair<float, float> const&);;

   private:
    torch::Tensor decode(const torch::Tensor& localization,
                         const torch::Tensor& priors);
    void convert(int, const torch::Tensor&, const torch::Tensor&,
                 const std::pair<float, float>&, std::vector<Landmark>&);
    void print_arguments();

    int _num_classes, _top_k, _bkg_label;
    float _conf_thresh, _nms_thresh;
    std::vector<float> _variances;
};

class PreProcessing {
   public:
    PreProcessing(const std::string&);
    torch::Tensor process(const cv::Mat&);

   private:
    int _size;
};
#endif
