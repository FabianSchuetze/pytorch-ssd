set(${LIBRARY_TARGET_NAME}_TEST
    accuracy.cpp
    initialize.cpp
)


#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -fprofile-arcs -ftest-coverage -g -O0\
    #-coverage -fno-inline -fno-inline-small-functions -fno-default-inline")
enable_testing()
foreach( testsourcefile ${${LIBRARY_TARGET_NAME}_TEST} )
    string( REPLACE ".cpp" "" testname ${testsourcefile} )
    add_executable( ${testname} ${testsourcefile} )
    target_link_libraries(${testname} PUBLIC Faces  "${TORCH_LIBRARIES}" TorchVision::TorchVision ${OpenCV_LIBS} gtest gmock gtest_main)
    add_test(${testname} ${testname})
endforeach( testsourcefile ${${LIBRARY_TARGET_NAME}_TEST})
