cmake_minimum_required(VERSION 3.10)
 project(PI-Format VERSION 4.2)

 set(CMAKE_CXX_STANDARD 17)
 set(CMAKE_CXX_STANDARD_REQUIRED ON)

 # Find dependencies
 find_package(Threads REQUIRED)
 find_package(Zstd REQUIRED)

 # library
 add_library(pi-format INTERFACE)
 target_include_directories(pi-format INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
 target_link_libraries(pi-format INTERFACE Threads::Threads Zstd::zstd)

 # Tool demo
 add_executable(create-demo  demo.cpp)
 target_link_libraries(create-demo PRIVATE pi-format)

 # Tool inspector
 add_executable(pix-inspect inspector.cpp)
 target_link_libraries(pix-inspect PRIVATE pi-format)
 Enable test support

enable_testing()

Add our test directory to the build

add_subdirectory(tests)

