#pragma once
#ifndef export_hpp
#define export_hpp
#ifdef _WIN32
#    ifdef LIBRARY_EXPORTS
#        define LIBRARY_API __declspec(dllexport)
#    else
#        define LIBRARY_API __declspec(dllimport)
#    endif
#elif __linux__
#    define LIBRARY_API
#endif
#endif