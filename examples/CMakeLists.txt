# examples/CMakeLists.txt

# Executable for generating demo data
add_executable(create-demo demo.cpp)
# Link our library to this application
target_link_libraries(create-demo PRIVATE pi-format)

# Executable for the inspector tool
add_executable(pix-inspect inspector.cpp)
# The inspector also needs our library
target_link_libraries(pix-inspect PRIVATE pi-format)
