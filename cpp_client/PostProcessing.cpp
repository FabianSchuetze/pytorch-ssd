#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torchvision/nms.h>
#include <torchvision/vision.h>

#include <fstream>

#include "DataProcessing.hpp"
#include "LoadConfig.hpp"

using torch::Tensor;
typedef std::vector<PostProcessing::Landmark> landmarks;

PostProcessing::PostProcessing(const std::string& config)
    : _num_classes(0),
      _top_k(0),
      _bkg_label(0),
      _conf_thresh(0),
      _nms_thresh(0),
      _variances(2) {
    std::ifstream paramFile{config};
    std::map<std::string, std::string> params{
        std::istream_iterator<kv_pair>{paramFile},
        std::istream_iterator<kv_pair>{}};
    _num_classes = std::stoi(params["num_classes"]);
    _top_k = std::stoi(params.at("top_k"));
    _bkg_label = std::stoi(params.at("bkg_label"));
    _conf_thresh = std::stof(params.at("conf_thresh"));
    _nms_thresh = std::stof(params.at("nms_thresh"));
    _variances[0] = std::stof(params.at("variance_0"));
    _variances[1] = std::stof(params.at("variance_1"));
    print_arguments();

    ;
}

void PostProcessing::print_arguments() {
    std::cout << "The parameters for the algortihm are: "
              << "num_classes: " << _num_classes << std::endl
              << "bgk_label: " << _bkg_label << std::endl
              << "conf_thresh: " << _conf_thresh << std::endl
              << "nms_thresh: " << _nms_thresh << std::endl
              << " bgk_label: " << _bkg_label << std::endl
              << " variance: " << _variances[0] << ", " << _variances[1]
              << std::endl;
}

landmarks PostProcessing::process(const Tensor& confidence, 
                                  const Tensor& localization,
                                  const std::pair<float, float>& img_size) {
    std::vector<PostProcessing::Landmark> results;
    Tensor loc = localization.squeeze(0);
    Tensor conf = confidence.squeeze(0);
    for (int i = 1; i < _num_classes; ++i) {
        Tensor cur = conf.slice(1, i, i + 1);
        Tensor confident = cur.gt(_conf_thresh);
        Tensor scores = cur.masked_select(confident);
        if (scores.size(0) == 0) {
            continue;
        }
        Tensor l_mask = confident.expand_as(loc);
        Tensor boxes = loc.masked_select(l_mask).view({-1, 4});
        Tensor ids = nms_cpu(boxes, scores, _nms_thresh);
        Tensor selected_scores = scores.index_select(0, ids);
        Tensor selected_boxes = boxes.index_select(0, ids);
        convert(i, selected_scores, selected_boxes, img_size, results);
    }
    return results;
}

void PostProcessing::convert(int label, const Tensor& scores, const Tensor& boxes,
                             const std::pair<float, float>& img_size,
                             landmarks& results) {
    int height = img_size.first;
    int width = img_size.second;
    for (int i = 0; i < scores.size(0); ++i) {
        float xmin = boxes[i][0].item<float>() * 300;
        float ymin = boxes[i][1].item<float>() * 300;
        float xmax = boxes[i][2].item<float>() * 300;
        float ymax = boxes[i][3].item<float>() * 300;
        PostProcessing::Landmark l;
        l.xmin = xmin;
        l.ymin = ymin;
        l.xmax = xmax;
        l.ymax = ymax;
        l.confidence = scores[i].item<float>();
        l.label = label;
        results.push_back(l);
    }
}
