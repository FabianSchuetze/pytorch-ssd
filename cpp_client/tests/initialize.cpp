#include <ATen/core/ivalue.h>
#include <gtest/gtest.h>
#include <torch/script.h>  // One-stop header.
#include <torch/torch.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>

#include "../DataProcessing.hpp"
#include "../Database.hpp"
#include "../Evaluate.hpp"
#include "../TS_SSDLiteCaller.hpp"

TEST(INIT, LoadConfig) {
    std::string root = "/home/fabian/Documents/work/github/pytorch-ssd/";
    std::string model = root + "traced_quantized.pt";
    std::string wrong_location = root + "params.txt";
    ASSERT_THROW(TS_SSDLiteCaller SSDLite(model, wrong_location),
                 std::runtime_error);
    std::string correct_location = root + "cpp_client/params.txt";
    ASSERT_NO_THROW(TS_SSDLiteCaller SSDLite(model, correct_location));
}

TEST(INIT, LoadModel) {
    std::string root = "/home/fabian/Documents/work/github/pytorch-ssd/";
    std::string wrong_model = root + "does_not_exists.pt";
    std::string location = root + "cpp_client/params.txt";
    ASSERT_ANY_THROW(TS_SSDLiteCaller SSDLite(wrong_model, location));
                 //std::runtime_error);
    std::string model = root + "traced_quantized.pt";
    ASSERT_NO_THROW(TS_SSDLiteCaller SSDLite(model, location));
}

TEST(INIT, LoadDataBase) {
    std::string wrong_location = "/home/fabian/test";
    ASSERT_THROW(Database database(wrong_location), std::runtime_error);
    std::string correct_file =
        "/home/fabian/data/TS/CrossCalibration/TCLObjectDetectionDatabase/"
        "database.xml";
    ASSERT_NO_THROW(Database database(correct_file));
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
};
