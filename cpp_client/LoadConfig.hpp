#include <fstream>
#include <istream>
#include <map>
#include <iostream>
#include <iterator>

struct kv_pair : public std::pair<std::string, std::string> {
    friend std::istream& operator>>(std::istream& in, kv_pair& p) {
        return in >> std::get<0>(p) >> std::get<1>(p);
    }
};
