/**
 * @file MeshTestMain.cpp
 * @brief Main entry point for the Mesh and Cell class test suite
 * 
 * This file provides the main function that runs all the tests for
 * the Mesh and Cell classes using Google Test framework.
 */

#include <gtest/gtest.h>
#include <iostream>

/**
 * @brief Main function to run all tests
 * 
 * @param argc Command line argument count
 * @param argv Command line arguments
 * @return int Program exit code (0 for success)
 */
int main(int argc, char **argv) {
    // Print test header
    std::cout << "=======================================" << std::endl;
    std::cout << "  Starting Mesh and Cell Class Tests" << std::endl;
    std::cout << "=======================================" << std::endl;

    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);
    
    // Run all tests and return the result
    int result = RUN_ALL_TESTS();
    
    // Print test footer
    std::cout << "=======================================" << std::endl;
    std::cout << "  Completed Mesh and Cell Class Tests" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    return result;
}
