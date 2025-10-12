# tests/CMakeLists.txt

# Include the CMake module for fetching dependencies
include(FetchContent)

# Download and set up Google Test
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
FetchContent_MakeAvailable(googletest)

# Create our test executable
add_executable(pi-tests main_test.cpp)

# Link our library and Google Test to the test executable
target_link_libraries(pi-tests PRIVATE pi-format GTest::gtest_main)

# Register the test with the CTest testing system
include(GoogleTest)
gtest_discover_tests(pi-tests)
