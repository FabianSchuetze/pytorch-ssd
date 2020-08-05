#ifndef database_hpp
#define database_hpp

#ifdef _WIN32
#    ifdef LIBRARY_EXPORTS
#        define LIBRARY_API __declspec(dllexport)
#    else
#        define LIBRARY_API __declspec(dllimport)
#    endif
#elif
#    define LIBRARY_API
#endif


#include "DataProcessing.hpp"
#include "tinyxml2.h"
#include <memory>
#include <string>

class Database{

    public:
        using iterator = std::pair<std::string, std::vector<Landmark>>;
        //using const_iterator = const std::vector<PostProcessing::Landmark>&;
        LIBRARY_API Database(const std::string&);

        LIBRARY_API iterator get_element();
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
        Landmark get_box(tinyxml2::XMLElement*);
        void init_database(const std::string&);
        void init_transform();
        iterator get_gts(tinyxml2::XMLElement*);
};
#endif
