#include "Database.hpp"

#include <map>
#include <stdexcept>

#include "/home/fabian/Documents/work/github/tinyxml2/tinyxml2.h"
#include "/home/fabian/Documents/work/github/tinyxml2/tinyxml2.h"

std::map<std::string, int> transform = {
    {"glabella", 1}, {"left_eye", 2}, {"right_eye", 3}, {"nose_tip", 4}};

PostProcessing::Landmark get_box(tinyxml2::XMLElement* box) {
    PostProcessing::Landmark landmark;
    landmark.xmax = std::stoi(box->Attribute("top"));
    landmark.xmax = std::stoi(box->Attribute("left"));
    landmark.ymin = std::stoi(box->Attribute("width"));
    landmark.ymax = std::stoi(box->Attribute("height"));
    std::string label = box->FirstChildElement()->GetText();
    landmark.label = transform[label];
    return landmark;
}

std::vector<Database> read_xml_file(const std::string& location) {
    std::vector<Database> database;
    tinyxml2::XMLDocument doc;
    tinyxml2::XMLError res = doc.LoadFile(location.c_str());
    if (res != tinyxml2::XML_SUCCESS) {
        throw std::runtime_error("Couldn't load file " + location);
    }
    tinyxml2::XMLElement* start =
        doc.FirstChildElement("dataset")->FirstChildElement("images");
    for (tinyxml2::XMLElement* test = start->FirstChildElement(); test != NULL;
         test = test->NextSiblingElement()) {
        Database data;
        data.filename = test->Attribute("file");
        for (tinyxml2::XMLElement* box = test->FirstChildElement(); box != NULL;
             box = box->NextSiblingElement()) {
            data.gts.push_back(get_box(box));
        }
        database.push_back(data);
    }
    return database;
}

