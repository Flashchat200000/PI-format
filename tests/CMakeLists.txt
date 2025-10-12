include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/release-1.12.1.zip
)
FetchContent_MakeAvailable(googletest)

add_executable(pi-tests main_test.cpp)

target_link_libraries(pi-tests PRIVATE pi-format GTest::gtest_main)
