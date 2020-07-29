#ifndef database_hpp
#define database_hpp
#include "DataProcessing.hpp"
#include "tinyxml2.h"
#include <memory>
#include <string>

class Database{

    public:
        using iterator = std::pair<std::string, std::vector<PostProcessing::Landmark>>;
        //using const_iterator = const std::vector<PostProcessing::Landmark>&;
        Database(const std::string&);

        iterator get_element();
        int length;
        //iterator end();
        //iterator operator++(int);
        //iterator operator++();

    private:
        tinyxml2::XMLDocument doc;
        int position;
        //tinyxml2::XMLElement* begin_database;
        //tinyxml2::XMLElement* end_database;
        //tinyxml2::XMLElement* position;
        std::map<std::string, int> transform;
        PostProcessing::Landmark get_box(tinyxml2::XMLElement*);
        void init_database(const std::string&);
        void init_transform();
        iterator get_gts(tinyxml2::XMLElement*);
};
#endif
