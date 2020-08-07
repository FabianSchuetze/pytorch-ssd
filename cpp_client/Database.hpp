#ifndef database_hpp
#define database_hpp
#define LIBRARY_EXPORTS

#include <tinyxml2.h>

#include <string>
#include <vector>

#include "DataProcessing.hpp"
#include "Export.hpp"

class Database {
   public:
    using iterator = std::pair<std::string, std::vector<Landmark>>;
    LIBRARY_API Database(const std::string&);
    LIBRARY_API iterator get_element();
    int length;

   private:
    tinyxml2::XMLDocument doc;
    int position;
    std::map<std::string, int> transform;
    Landmark get_box(tinyxml2::XMLElement*);
    void init_database(const std::string&);
    void init_transform();
    iterator get_gts(tinyxml2::XMLElement*);
};
#endif
