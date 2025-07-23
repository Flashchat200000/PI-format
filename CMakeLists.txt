cmake_minimum_required(VERSION 3.15)

project(PixEngine VERSION 4.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add compiler flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3 -g")
endif()

# Find required packages
find_package(PkgConfig REQUIRED)

# Find threading library
find_package(Threads REQUIRED)

# Find zstd
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZSTD REQUIRED libzstd)

# Find math library
find_library(MATH_LIBRARY m)

# Try to find GLM (header-only)
find_path(GLM_INCLUDE_DIR glm/glm.hpp
    HINTS
    /usr/include
    /usr/local/include
    /opt/local/include
)

if(NOT GLM_INCLUDE_DIR)
    message(STATUS "GLM not found system-wide, using fallback math implementation")
    # Create a simple fallback for GLM if not available
    add_definitions(-DPIX_USE_SIMPLE_MATH)
endif()

# Create the main executable
add_executable(pix_engine pix_engine_v40.cpp)

# Link libraries
target_link_libraries(pix_engine 
    Threads::Threads
    ${ZSTD_LIBRARIES}
)

if(MATH_LIBRARY)
    target_link_libraries(pix_engine ${MATH_LIBRARY})
endif()

# Include directories
if(GLM_INCLUDE_DIR)
    target_include_directories(pix_engine PRIVATE ${GLM_INCLUDE_DIR})
endif()

target_include_directories(pix_engine PRIVATE ${ZSTD_INCLUDE_DIRS})

# Add compile definitions
target_compile_definitions(pix_engine PRIVATE ${ZSTD_CFLAGS_OTHER})

# Optional: Add graphics libraries if available
find_package(OpenGL QUIET)
find_package(glfw3 QUIET)
find_package(GLEW QUIET)
find_package(Vulkan QUIET)

if(OpenGL_FOUND)
    target_link_libraries(pix_engine OpenGL::GL)
    target_compile_definitions(pix_engine PRIVATE PIX_OPENGL_SUPPORT)
endif()

if(glfw3_FOUND)
    target_link_libraries(pix_engine glfw)
    target_compile_definitions(pix_engine PRIVATE PIX_GLFW_SUPPORT)
endif()

if(GLEW_FOUND)
    target_link_libraries(pix_engine GLEW::GLEW)
    target_compile_definitions(pix_engine PRIVATE PIX_GLEW_SUPPORT)
endif()

if(Vulkan_FOUND)
    target_link_libraries(pix_engine Vulkan::Vulkan)
    target_compile_definitions(pix_engine PRIVATE PIX_VULKAN_SUPPORT)
endif()

# Set output directory
set_target_properties(pix_engine PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Installation
install(TARGETS pix_engine
    RUNTIME DESTINATION bin
)

# Print configuration summary
message(STATUS "PIX Engine v4.0 Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Threading: ${CMAKE_THREAD_LIBS_INIT}")
message(STATUS "  ZSTD: ${ZSTD_FOUND}")
if(GLM_INCLUDE_DIR)
    message(STATUS "  GLM: Found at ${GLM_INCLUDE_DIR}")
else()
    message(STATUS "  GLM: Using fallback math")
endif()
if(OpenGL_FOUND)
    message(STATUS "  OpenGL: Found")
endif()
if(glfw3_FOUND)
    message(STATUS "  GLFW: Found")
endif()
if(GLEW_FOUND)
    message(STATUS "  GLEW: Found")
endif()
if(Vulkan_FOUND)
    message(STATUS "  Vulkan: Found")
endif()