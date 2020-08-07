set(${LIBRARY_TARGET_NAME}_TEST
    accuracy.cpp
    initialize.cpp
)

enable_testing()
foreach( testsourcefile ${${LIBRARY_TARGET_NAME}_TEST} )
    string( REPLACE ".cpp" "" testname ${testsourcefile} )
    add_executable( ${testname} ${testsourcefile})
    target_link_libraries(${testname} PUBLIC Faces  "${TORCH_LIBRARIES}" TorchVision::TorchVision ${OpenCV_LIBS} tinyxml2 gtest gmock gtest_main)
    add_test(${testname} ${testname})
endforeach( testsourcefile ${${LIBRARY_TARGET_NAME}_TEST})
