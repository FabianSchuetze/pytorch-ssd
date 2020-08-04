#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torchvision/nms.h>
#include <torchvision/vision.h>

#include <fstream>

#include "DataProcessing.hpp"
//#include "LoadConfig.hpp"

using torch::Tensor;
typedef std::vector<Landmark> Landmarks;

PostProcessing::PostProcessing(const std::string& config)
    : _num_classes(0), _bkg_label(0), _conf_thresh(0), _nms_thresh(0) {
    std::ifstream paramFile{config};
    if (paramFile.fail()) {
        std::string m("Cannot load config at: " + config + ", thrown from:\n");
#ifdef __linux__
        throw std::runtime_error(m + __PRETTY_FUNCTION__);
#elif _WIN32
        throw std::runtime_error(m + __FUNCSIG__);
#endif
    }
    std::map<std::string, std::string> params{
        std::istream_iterator<kv_pair>{paramFile},
        std::istream_iterator<kv_pair>{}};
    _num_classes = std::stoi(params["num_classes"]);
    _bkg_label = std::stoi(params.at("bkg_label"));
    _conf_thresh = std::stof(params.at("conf_thresh"));
    _nms_thresh = std::stof(params.at("nms_thresh"));
    print_arguments();

    ;
}

void PostProcessing::print_arguments() {
    std::cout << "The parameters for the algortihm are: "
              << "num_classes: " << _num_classes << std::endl
              << "bgk_label: " << _bkg_label << std::endl
              << "conf_thresh: " << _conf_thresh << std::endl
              << "nms_thresh: " << _nms_thresh << std::endl
              << " bgk_label: " << _bkg_label << std::endl;
}

Landmarks PostProcessing::process(const Tensor& confidence,
                                  const Tensor& localization,
                                  const std::pair<float, float>& img_size) {
    std::vector<Landmark> results;
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

void PostProcessing::convert(int label, const Tensor& scores,
                             const Tensor& boxes,
                             const std::pair<float, float>& img_size,
                             Landmarks& results) {
    int height = img_size.first;
    int width = img_size.second;
    for (int i = 0; i < scores.size(0); ++i) {
        float xmin = boxes[i][0].item<float>() * 300;
        float ymin = boxes[i][1].item<float>() * 300;
        float xmax = boxes[i][2].item<float>() * 300;
        float ymax = boxes[i][3].item<float>() * 300;
        Landmark l;
        l.xmin = xmin;
        l.ymin = ymin;
        l.xmax = xmax;
        l.ymax = ymax;
        l.confidence = scores[i].item<float>();
        l.label = label;
        results.push_back(l);
    }
}

void serialize_results(const std::string& file, const Landmarks& result) {
    std::ofstream myfile;
    size_t pos = file.find_last_of("/");
    std::string filename = file.substr(pos + 1);
    size_t pos_end = filename.find(".");
    std::string token = filename.substr(0, pos_end);
    std::string outfile = "results/" + token + ".result";
    myfile.open(outfile, std::ios::trunc);
    if (myfile.fail()) {
        std::cout << "couldnt open file: " << outfile << std::endl;
    } else {
        for (const Landmark& res : result) {
            float xmin = res.xmin;
            float ymin = res.ymin;
            float xmax = res.xmax;
            float ymax = res.ymax;
            myfile << xmin << ", " << ymin << ", " << xmax << ", " << ymax
                   << ", " << res.confidence << ", " << res.label << std::endl;
        }
    }
    myfile.close();
}
