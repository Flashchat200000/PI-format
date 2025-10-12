# examples/CMakeLists.txt

# Create an executable file to generate demo data
add_executable(create-demo demo.cpp)

# Include our pi-format library in it.
# This command will automatically do everything necessary:
# 1. Add paths to header files from the 'include' folder.
# 2. Include all dependencies, such as Zstd and Threads.
target_link_libraries(create-demo PRIVATE pi-format)

# Create an executable file for the inspector
add_executable(pix-inspect inspector.cpp)

# And do the same with the inspector.
target_link_libraries(pix-inspect PRIVATE pi-format)
