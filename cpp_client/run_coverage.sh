#sript to run coverage of tests and the library

./build/tests/accuracy
./build/tests/initialize

mkdir -p build/coverage


lcov --capture --directory build/tests/CMakeFiles/accuracy.dir \
    --output-file build/tests/CMakeFiles/accuracy.dir/coverage.info

lcov --capture --directory build/tests/CMakeFiles/initialize.dir \
    --output-file build/tests/CMakeFiles/initialize.dir/coverage.info

lcov --capture --directory build/CMakeFiles/Faces.dir \
    --output-file build/CMakeFiles/Faces.dir/coverage.info

lcov -a build/CMakeFiles/Faces.dir/coverage.info \
     -a build/tests/CMakeFiles/accuracy.dir/coverage.info \
     -a build/tests/CMakeFiles/initialize.dir/coverage.info \
     -o build/coverage/total.info

lcov --remove build/coverage/total.info "*torch/*" -o build/coverage/total.info
lcov --remove build/coverage/total.info "/usr*" -o build/coverage/total.info
lcov --remove build/coverage/total.info "*tinyxml2*" -o build/coverage/total.info

genhtml build/coverage/total.info --output-directory build/coverage/
