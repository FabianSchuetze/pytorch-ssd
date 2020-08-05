#include "Database.hpp"

#include <map>
#include <memory>
#include <stdexcept>
#include <tinyxml2.h>

Database::Database(const std::string& location)
    : length(0), doc(), position(0) {
    init_database(location);
    init_transform();
}

void Database::init_database(const std::string& location) {
    tinyxml2::XMLError res = doc.LoadFile(location.c_str());
    if (res != tinyxml2::XML_SUCCESS) {
        std::string m("Cannot parse file, at " + location + ", thrown from:\n");
        throw std::runtime_error(m + __PRETTY_FUNCTION__);
    }
    tinyxml2::XMLElement* root =
        doc.FirstChildElement("dataset")->FirstChildElement("images");
    for (tinyxml2::XMLElement* tmp = root->FirstChildElement(); tmp != NULL;
         tmp = tmp->NextSiblingElement()) {
        length++;
    }
}

void Database::init_transform() {
    transform["glabella"] = 1;
    transform["left_eye"] = 2;
    transform["right_eye"] = 3;
    transform["nose_tip"] = 4;
};

Landmark Database::get_box(tinyxml2::XMLElement* box) {
    Landmark landmark;
    float xmin = std::stof(box->Attribute("left"));
    float xmax = xmin + std::stof(box->Attribute("width"));
    float ymin = std::stof(box->Attribute("top"));
    float ymax = ymin + std::stof(box->Attribute("height"));
    landmark.xmin = xmin * 300.0 / 320;
    landmark.xmax = xmax * 300.0 / 320;
    landmark.ymin = ymin * 300.0 / 256;
    landmark.ymax = ymax * 300.0 / 256;
    std::string label = box->FirstChildElement()->GetText();
    landmark.label = transform[label];
    return landmark;
}

Database::iterator Database::get_gts(tinyxml2::XMLElement* pos) {
    std::pair<std::string, std::vector<Landmark>> gts;
    for (tinyxml2::XMLElement* box = pos->FirstChildElement(); box != NULL;
         box = box->NextSiblingElement()) {
        gts.second.push_back(get_box(box));
    }
    gts.first = pos->Attribute("file");
    return gts;
}

Database::iterator Database::get_element() {
    tinyxml2::XMLElement* start = doc.FirstChildElement("dataset")
                                      ->FirstChildElement("images")
                                      ->FirstChildElement();
    int tmp = 0;
    while (tmp < position) {
        start = start->NextSiblingElement();
        tmp++;
    }
    std::string name = start->Attribute("file");
    iterator res = get_gts(start);
    position++;
    return res;
}

