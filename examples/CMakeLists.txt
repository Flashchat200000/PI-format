# examples/CMakeLists.txt

# Tell CMake that for ALL targets in this folder (demo and inspector)
# the root 'include' folder needs to be added to the header search path
target_include_directories(create-demo PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_include_directories(pix-inspect PRIVATE ${PROJECT_SOURCE_DIR}/include)

add_executable(create-demo demo.cpp)
target_link_libraries(create-demo PRIVATE pi-format)

add_executable(pix-inspect inspector.cpp)
target_link_libraries(pix-inspect PRIVATE pi-format)
#Update: A more correct and cleaner way is to add search paths for each target separately.

 #Fixing examples/CMakeLists.txt:
cmake
# examples/CMakeLists.txt

add_executable(create-demo demo.cpp)
# Declaring that create-demo depends on pi-format
target_link_libraries(create-demo PRIVATE pi-format)
# Now the compiler knows to look in the `include` directory
# which we defined for pi-format in the main CMakeLists.txt

add_executable(pix-inspect inspector.cpp)
target_link_libraries(pix-inspect PRIVATE pi-format)
