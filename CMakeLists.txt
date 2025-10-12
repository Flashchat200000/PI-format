# CMakeLists.txt (root)

# CMake version and project name
cmake_minimum_required(VERSION 3.10)
project(PI-Format VERSION 4.2 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- PI-FORMAT LIBRARY ---
# Create our library. It’s header-only, so we use INTERFACE.
add_library(pi-format INTERFACE)

# Tell CMake that public headers for pi-format
# are located in the 'include' directory.
target_include_directories(pi-format INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Link dependencies to our library
find_package(Threads REQUIRED)
find_package(Zstd REQUIRED)
target_link_libraries(pi-format INTERFACE Threads::Threads Zstd::zstd)

# --- APPLICATIONS AND TESTS ---

# Include the examples folder in the build.
# CMake will go into 'examples' and look for another CMakeLists.txt there.
add_subdirectory(examples)

# Enable testing support
enable_testing()

# Include the tests folder in the build.
# CMake will go into 'tests' and look for yet another CMakeLists.txt there.
add_subdirectory(tests)
