#include "ConfigManager.h"
#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>

class ConfigManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set up a basic configuration for tests
        config.addGroup("TestGroup", "Group for testing");
        config.setParameter<int>("TestGroup", "intParam", 42, "Test integer parameter");
        config.setParameter<double>("TestGroup", "doubleParam", 3.14159, "Test double parameter");
        config.setParameter<bool>("TestGroup", "boolParam", true, "Test boolean parameter");
        config.setParameter<std::string>("TestGroup", "stringParam", "hello", "Test string parameter");
        config.setParameter<std::vector<int>>("TestGroup", "intVectorParam", {1, 2, 3}, "Test int vector");
        config.setParameter<std::vector<double>>("TestGroup", "doubleVectorParam", {1.1, 2.2, 3.3}, "Test double vector");
        config.setParameter<std::vector<std::string>>("TestGroup", "stringVectorParam", 
                                                    {"one", "two", "three"}, "Test string vector");
    }

    void TearDown() override {
        // Clean up any test files
        if (std::filesystem::exists("test_config.ini")) {
            std::filesystem::remove("test_config.ini");
        }
    }

    ConfigManager config;
};

// Test adding and retrieving parameters
TEST_F(ConfigManagerTest, BasicParameterOperations) {
    // Test parameter retrieval
    EXPECT_EQ(config.getParameter<int>("TestGroup", "intParam"), 42);
    EXPECT_DOUBLE_EQ(config.getParameter<double>("TestGroup", "doubleParam"), 3.14159);
    EXPECT_TRUE(config.getParameter<bool>("TestGroup", "boolParam"));
    EXPECT_EQ(config.getParameter<std::string>("TestGroup", "stringParam"), "hello");
    
    // Test vector parameters
    auto intVector = config.getParameter<std::vector<int>>("TestGroup", "intVectorParam");
    EXPECT_EQ(intVector.size(), 3);
    EXPECT_EQ(intVector[0], 1);
    EXPECT_EQ(intVector[1], 2);
    EXPECT_EQ(intVector[2], 3);
    
    auto doubleVector = config.getParameter<std::vector<double>>("TestGroup", "doubleVectorParam");
    EXPECT_EQ(doubleVector.size(), 3);
    EXPECT_DOUBLE_EQ(doubleVector[0], 1.1);
    EXPECT_DOUBLE_EQ(doubleVector[1], 2.2);
    EXPECT_DOUBLE_EQ(doubleVector[2], 3.3);
    
    auto stringVector = config.getParameter<std::vector<std::string>>("TestGroup", "stringVectorParam");
    EXPECT_EQ(stringVector.size(), 3);
    EXPECT_EQ(stringVector[0], "one");
    EXPECT_EQ(stringVector[1], "two");
    EXPECT_EQ(stringVector[2], "three");
}

// Test parameter modification
TEST_F(ConfigManagerTest, ParameterModification) {
    // Modify parameters
    config.setParameter<int>("TestGroup", "intParam", 100);
    EXPECT_EQ(config.getParameter<int>("TestGroup", "intParam"), 100);
    
    config.setParameter<double>("TestGroup", "doubleParam", 2.71828);
    EXPECT_DOUBLE_EQ(config.getParameter<double>("TestGroup", "doubleParam"), 2.71828);
    
    config.setParameter<bool>("TestGroup", "boolParam", false);
    EXPECT_FALSE(config.getParameter<bool>("TestGroup", "boolParam"));
    
    config.setParameter<std::string>("TestGroup", "stringParam", "world");
    EXPECT_EQ(config.getParameter<std::string>("TestGroup", "stringParam"), "world");
    
    // Modify vector parameters
    config.setParameter<std::vector<int>>("TestGroup", "intVectorParam", {4, 5, 6});
    auto newIntVector = config.getParameter<std::vector<int>>("TestGroup", "intVectorParam");
    EXPECT_EQ(newIntVector.size(), 3);
    EXPECT_EQ(newIntVector[0], 4);
    EXPECT_EQ(newIntVector[1], 5);
    EXPECT_EQ(newIntVector[2], 6);
}

// Test default parameter values
TEST_F(ConfigManagerTest, DefaultParameterValues) {
    // Test retrieving non-existent parameter with default
    EXPECT_EQ(config.getParameter<int>("TestGroup", "nonExistentParam", 123), 123);
    EXPECT_DOUBLE_EQ(config.getParameter<double>("TestGroup", "nonExistentParam", 2.5), 2.5);
    EXPECT_FALSE(config.getParameter<bool>("TestGroup", "nonExistentParam", false));
    EXPECT_EQ(config.getParameter<std::string>("TestGroup", "nonExistentParam", "default"), "default");
    
    // Test retrieving from non-existent group with default
    EXPECT_EQ(config.getParameter<int>("NonExistentGroup", "intParam", 456), 456);
}

// Test type safety
TEST_F(ConfigManagerTest, TypeSafety) {
    // Attempt to retrieve parameter with wrong type
    // This should return the default value and print an error
    EXPECT_EQ(config.getParameter<double>("TestGroup", "intParam", 0.0), 0.0);
    EXPECT_EQ(config.getParameter<int>("TestGroup", "doubleParam", 0), 0);
    EXPECT_EQ(config.getParameter<int>("TestGroup", "stringParam", 999), 999);
}

// Test adding new groups
TEST_F(ConfigManagerTest, GroupOperations) {
    // Add a new group
    config.addGroup("NewGroup", "A new test group");
    EXPECT_TRUE(config.hasGroup("NewGroup"));
    
    // Add parameters to the new group
    config.setParameter<int>("NewGroup", "count", 10);
    EXPECT_EQ(config.getParameter<int>("NewGroup", "count"), 10);
    
    // Check that existing group still exists
    EXPECT_TRUE(config.hasGroup("TestGroup"));
    
    // Check non-existent group
    EXPECT_FALSE(config.hasGroup("NonExistentGroup"));
}

// Test file I/O
TEST_F(ConfigManagerTest, FileIO) {
    // Save configuration to file
    EXPECT_TRUE(config.saveToFile("test_config.ini"));
    
    // Create a new config and load from file
    ConfigManager loadedConfig;
    EXPECT_TRUE(loadedConfig.loadFromFile("test_config.ini"));
    
    // Verify loaded parameters
    EXPECT_EQ(loadedConfig.getParameter<int>("TestGroup", "intParam"), 42);
    EXPECT_DOUBLE_EQ(loadedConfig.getParameter<double>("TestGroup", "doubleParam"), 3.14159);
    EXPECT_TRUE(loadedConfig.getParameter<bool>("TestGroup", "boolParam"));
    EXPECT_EQ(loadedConfig.getParameter<std::string>("TestGroup", "stringParam"), "hello");
    
    // Verify vector parameters
    auto intVector = loadedConfig.getParameter<std::vector<int>>("TestGroup", "intVectorParam");
    EXPECT_EQ(intVector.size(), 3);
    EXPECT_EQ(intVector[0], 1);
    EXPECT_EQ(intVector[1], 2);
    EXPECT_EQ(intVector[2], 3);
}

// Test failure handling
TEST_F(ConfigManagerTest, FailureHandling) {
    // Try to load from non-existent file
    ConfigManager invalidConfig;
    EXPECT_FALSE(invalidConfig.loadFromFile("non_existent_file.ini"));
    
    // Try to save to invalid location
    EXPECT_FALSE(config.saveToFile("/invalid/path/test.ini"));
}

// Test special cases for string serialization
TEST_F(ConfigManagerTest, StringSerialization) {
    // Test strings with special characters
    config.setParameter<std::string>("TestGroup", "specialString", "Quote \" and comma , and newline \n");
    config.saveToFile("test_config.ini");
    
    ConfigManager loadedConfig;
    loadedConfig.loadFromFile("test_config.ini");
    
    EXPECT_EQ(loadedConfig.getParameter<std::string>("TestGroup", "specialString"), 
             "Quote \" and comma , and newline \n");
    
    // Test empty strings
    config.setParameter<std::string>("TestGroup", "emptyString", "");
    config.saveToFile("test_config.ini");
    
    loadedConfig = ConfigManager();
    loadedConfig.loadFromFile("test_config.ini");
    
    EXPECT_EQ(loadedConfig.getParameter<std::string>("TestGroup", "emptyString"), "");
}

// Test comment preservation
TEST_F(ConfigManagerTest, CommentPreservation) {
    // Set parameters with descriptions
    config.setParameter<int>("TestGroup", "paramWithComment", 100, "This is a comment");
    config.saveToFile("test_config.ini");
    
    // Manually check if the file contains the comment
    std::ifstream file("test_config.ini");
    std::string fileContents((std::istreambuf_iterator<char>(file)), 
                             std::istreambuf_iterator<char>());
    
    EXPECT_TRUE(fileContents.find("This is a comment") != std::string::npos);
}

// Test boundary values
TEST_F(ConfigManagerTest, BoundaryValues) {
    // Test extreme values
    config.setParameter<int>("TestGroup", "maxInt", std::numeric_limits<int>::max());
    config.setParameter<int>("TestGroup", "minInt", std::numeric_limits<int>::min());
    config.setParameter<double>("TestGroup", "maxDouble", std::numeric_limits<double>::max());
    config.setParameter<double>("TestGroup", "minDouble", std::numeric_limits<double>::lowest());
    
    config.saveToFile("test_config.ini");
    
    ConfigManager loadedConfig;
    loadedConfig.loadFromFile("test_config.ini");
    
    EXPECT_EQ(loadedConfig.getParameter<int>("TestGroup", "maxInt"), std::numeric_limits<int>::max());
    EXPECT_EQ(loadedConfig.getParameter<int>("TestGroup", "minInt"), std::numeric_limits<int>::min());
    EXPECT_DOUBLE_EQ(loadedConfig.getParameter<double>("TestGroup", "maxDouble"), 
                   std::numeric_limits<double>::max());
    EXPECT_DOUBLE_EQ(loadedConfig.getParameter<double>("TestGroup", "minDouble"), 
                   std::numeric_limits<double>::lowest());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

