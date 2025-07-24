# ====================================================================================
# PIX ENGINE ULTIMATE v10.0 - PRODUCTION MAKEFILE
# 
# 🔥 МАКСИМАЛЬНАЯ ОПТИМИЗАЦИЯ + CROSS-PLATFORM SUPPORT
# 🔥 READY FOR AAA PRODUCTION BUILDS
# ====================================================================================

# Project configuration
PROJECT_NAME = pix_engine_ultimate_v10
VERSION = 10.0.0
TARGET = $(PROJECT_NAME)

# Build directories
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj
BIN_DIR = $(BUILD_DIR)/bin
LIB_DIR = $(BUILD_DIR)/lib

# Source files
SOURCES = pix_engine_ultimate_v10.cpp
DEMO_SOURCES = pix_demo_simple.cpp
HEADERS = pix_engine_ultimate_v10.hpp
OBJECTS = $(SOURCES:%.cpp=$(OBJ_DIR)/%.o)
DEMO_OBJECTS = $(DEMO_SOURCES:%.cpp=$(OBJ_DIR)/%.o)

# Compiler configuration
CXX = g++
CC = gcc

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Linux)
    PLATFORM = linux
    PLATFORM_DEFINES = -DPIX_PLATFORM_LINUX
    PLATFORM_LIBS = -lpthread -ldl -lm
    PLATFORM_FLAGS = -fPIC
endif

ifeq ($(UNAME_S),Darwin)
    PLATFORM = macos
    PLATFORM_DEFINES = -DPIX_PLATFORM_MACOS
    PLATFORM_LIBS = -lpthread -framework CoreFoundation
    PLATFORM_FLAGS = -fPIC
endif

ifeq ($(OS),Windows_NT)
    PLATFORM = windows
    PLATFORM_DEFINES = -DPIX_PLATFORM_WINDOWS
    PLATFORM_LIBS = -lwinmm -lkernel32 -luser32 -lgdi32 -lwinspool -lcomdlg32 -ladvapi32 -lshell32 -lole32 -loleaut32 -luuid -lodbc32 -lodbccp32
    PLATFORM_FLAGS = 
    TARGET := $(TARGET).exe
endif

# Build configuration
BUILD_TYPE ?= Release

# Base compiler flags
CXXFLAGS_BASE = -std=c++20 -Wall -Wextra -Wpedantic
CXXFLAGS_BASE += $(PLATFORM_DEFINES) $(PLATFORM_FLAGS)

# Debug build flags
CXXFLAGS_DEBUG = $(CXXFLAGS_BASE) -g -O0 -DDEBUG -D_DEBUG
CXXFLAGS_DEBUG += -fsanitize=address -fsanitize=undefined -fstack-protector-strong
CXXFLAGS_DEBUG += -Wno-unused-parameter -Wno-unused-variable

# Release build flags  
CXXFLAGS_RELEASE = $(CXXFLAGS_BASE) -O3 -DNDEBUG -DRELEASE
CXXFLAGS_RELEASE += -march=native -mtune=native -flto -ffast-math
CXXFLAGS_RELEASE += -fomit-frame-pointer -ffunction-sections -fdata-sections
CXXFLAGS_RELEASE += -Wl,--gc-sections -Wl,--strip-all

# Profile build flags
CXXFLAGS_PROFILE = $(CXXFLAGS_RELEASE) -pg -fprofile-arcs -ftest-coverage

# Set flags based on build type
ifeq ($(BUILD_TYPE),Debug)
    CXXFLAGS = $(CXXFLAGS_DEBUG)
    LDFLAGS = -fsanitize=address -fsanitize=undefined
endif

ifeq ($(BUILD_TYPE),Release)
    CXXFLAGS = $(CXXFLAGS_RELEASE)
    LDFLAGS = -O3 -flto -Wl,--gc-sections -Wl,--strip-all
endif

ifeq ($(BUILD_TYPE),Profile)
    CXXFLAGS = $(CXXFLAGS_PROFILE)
    LDFLAGS = -pg
endif

# Libraries
LIBS = $(PLATFORM_LIBS)

# Include directories
INCLUDES = -I. -I./include

# Default target
.PHONY: all
all: $(BIN_DIR)/$(TARGET)

# Create directories
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(OBJ_DIR): | $(BUILD_DIR)
	@mkdir -p $(OBJ_DIR)

$(BIN_DIR): | $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

$(LIB_DIR): | $(BUILD_DIR)
	@mkdir -p $(LIB_DIR)

# Compile object files
$(OBJ_DIR)/%.o: %.cpp $(HEADERS) | $(OBJ_DIR)
	@echo "🔧 Compiling $< ($(BUILD_TYPE))"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Link executable
$(BIN_DIR)/$(TARGET): $(OBJECTS) | $(BIN_DIR)
	@echo "🔗 Linking $(TARGET) ($(BUILD_TYPE))"
	$(CXX) $(OBJECTS) $(LDFLAGS) $(LIBS) -o $@
	@echo "✅ Build complete: $(BIN_DIR)/$(TARGET)"

# Build configurations
.PHONY: debug release profile

debug:
	@$(MAKE) BUILD_TYPE=Debug

release:
	@$(MAKE) BUILD_TYPE=Release

profile:
	@$(MAKE) BUILD_TYPE=Profile

# Build demo
$(BIN_DIR)/$(TARGET)_demo: $(DEMO_OBJECTS) | $(BIN_DIR)
	@echo "🔗 Linking $(TARGET)_demo ($(BUILD_TYPE))"
	$(CXX) $(DEMO_OBJECTS) $(LDFLAGS) $(LIBS) -o $@
	@echo "✅ Demo build complete: $(BIN_DIR)/$(TARGET)_demo"

# Run the demo
.PHONY: demo
demo: $(BIN_DIR)/$(TARGET)_demo
	@echo "🚀 Running PIX Engine Ultimate v$(VERSION) Demo"
	@cd $(BIN_DIR) && ./$(TARGET)_demo

# Run the engine
.PHONY: run
run: $(BIN_DIR)/$(TARGET)
	@echo "🚀 Running PIX Engine Ultimate v$(VERSION)"
	@cd $(BIN_DIR) && ./$(TARGET)

# Clean build artifacts
.PHONY: clean
clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@rm -f *.pix *.log
	@echo "✅ Clean complete"

# Install (copy to system directories)
.PHONY: install
install: $(BIN_DIR)/$(TARGET)
	@echo "📦 Installing PIX Engine Ultimate v$(VERSION)..."
ifeq ($(PLATFORM),linux)
	sudo cp $(BIN_DIR)/$(TARGET) /usr/local/bin/
	sudo cp $(HEADERS) /usr/local/include/
endif
ifeq ($(PLATFORM),macos)
	sudo cp $(BIN_DIR)/$(TARGET) /usr/local/bin/
	sudo cp $(HEADERS) /usr/local/include/
endif
	@echo "✅ Installation complete"

# Uninstall
.PHONY: uninstall
uninstall:
	@echo "🗑️ Uninstalling PIX Engine Ultimate..."
ifeq ($(PLATFORM),linux)
	sudo rm -f /usr/local/bin/$(TARGET)
	sudo rm -f /usr/local/include/pix_engine_ultimate_v10.hpp
endif
ifeq ($(PLATFORM),macos)
	sudo rm -f /usr/local/bin/$(TARGET)
	sudo rm -f /usr/local/include/pix_engine_ultimate_v10.hpp
endif
	@echo "✅ Uninstall complete"

# Static analysis
.PHONY: analyze
analyze:
	@echo "🔍 Running static analysis..."
	@which cppcheck > /dev/null 2>&1 && cppcheck --enable=all --std=c++20 $(SOURCES) || echo "⚠️ cppcheck not found"
	@which clang-tidy > /dev/null 2>&1 && clang-tidy $(SOURCES) -- $(CXXFLAGS) $(INCLUDES) || echo "⚠️ clang-tidy not found"

# Code formatting
.PHONY: format
format:
	@echo "🎨 Formatting code..."
	@which clang-format > /dev/null 2>&1 && clang-format -i $(SOURCES) $(HEADERS) || echo "⚠️ clang-format not found"

# Generate documentation
.PHONY: docs
docs:
	@echo "📚 Generating documentation..."
	@which doxygen > /dev/null 2>&1 && doxygen Doxyfile || echo "⚠️ doxygen not found"

# Memory check (Valgrind)
.PHONY: memcheck
memcheck: debug
	@echo "🔍 Running memory analysis..."
ifeq ($(PLATFORM),linux)
	@which valgrind > /dev/null 2>&1 && valgrind --leak-check=full --show-leak-kinds=all $(BIN_DIR)/$(TARGET) || echo "⚠️ valgrind not found"
else
	@echo "⚠️ Memory check only available on Linux"
endif

# Performance profiling
.PHONY: perf
perf: profile
	@echo "📊 Running performance analysis..."
ifeq ($(PLATFORM),linux)
	@which perf > /dev/null 2>&1 && perf record -g $(BIN_DIR)/$(TARGET) && perf report || echo "⚠️ perf not found"
else
	@echo "⚠️ Performance profiling only available on Linux"
endif

# Benchmarks
.PHONY: benchmark
benchmark: release
	@echo "🏃 Running benchmarks..."
	@cd $(BIN_DIR) && time ./$(TARGET)

# Package for distribution
.PHONY: package
package: release
	@echo "📦 Creating distribution package..."
	@tar -czf $(PROJECT_NAME)-v$(VERSION)-$(PLATFORM)-$(UNAME_M).tar.gz -C $(BIN_DIR) $(TARGET)
	@echo "✅ Package created: $(PROJECT_NAME)-v$(VERSION)-$(PLATFORM)-$(UNAME_M).tar.gz"

# System information
.PHONY: sysinfo
sysinfo:
	@echo "🖥️ SYSTEM INFORMATION:"
	@echo "  Platform: $(PLATFORM)"
	@echo "  Architecture: $(UNAME_M)" 
	@echo "  Compiler: $(shell $(CXX) --version | head -n1)"
	@echo "  Build Type: $(BUILD_TYPE)"
	@echo "  CPU Cores: $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 'unknown')"
	@echo "  Memory: $(shell free -h 2>/dev/null | grep Mem | awk '{print $$2}' || sysctl -n hw.memsize | awk '{print $$1/1024/1024/1024 \"GB\"}' 2>/dev/null || echo 'unknown')"

# Help
.PHONY: help
help:
	@echo "🔥 PIX ENGINE ULTIMATE v$(VERSION) - BUILD SYSTEM"
	@echo ""
	@echo "Available targets:"
	@echo "  all          - Build the engine (default)"
	@echo "  debug        - Build debug version"
	@echo "  release      - Build optimized release version"
	@echo "  profile      - Build with profiling enabled"
	@echo "  run          - Build and run the engine"
	@echo "  clean        - Remove all build artifacts"
	@echo "  install      - Install to system directories"
	@echo "  uninstall    - Remove from system directories"
	@echo "  analyze      - Run static code analysis"
	@echo "  format       - Format source code"
	@echo "  docs         - Generate documentation"
	@echo "  memcheck     - Run memory leak detection"
	@echo "  perf         - Run performance profiling"
	@echo "  benchmark    - Run performance benchmarks"
	@echo "  package      - Create distribution package"
	@echo "  sysinfo      - Show system information"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Build configurations:"
	@echo "  BUILD_TYPE=Debug    - Debug build with sanitizers"
	@echo "  BUILD_TYPE=Release  - Optimized production build"
	@echo "  BUILD_TYPE=Profile  - Profiling build"
	@echo ""
	@echo "Examples:"
	@echo "  make release        - Build optimized version"
	@echo "  make run            - Build and run"
	@echo "  make memcheck       - Check for memory leaks"
	@echo "  make package        - Create distribution package"

# Show build statistics
.PHONY: stats
stats: $(BIN_DIR)/$(TARGET)
	@echo "📊 BUILD STATISTICS:"
	@echo "  Lines of code: $(shell cat $(SOURCES) $(HEADERS) | wc -l)"
	@echo "  Binary size: $(shell ls -lh $(BIN_DIR)/$(TARGET) | awk '{print $$5}')"
	@echo "  Build time: $(shell date)"
	@echo "  Platform: $(PLATFORM)-$(UNAME_M)"

# Continuous integration
.PHONY: ci
ci: clean analyze debug release test package
	@echo "✅ Continuous integration pipeline complete"

# Test target (placeholder for future test implementation)
.PHONY: test
test: debug
	@echo "🧪 Running tests..."
	@echo "⚠️ Test suite not yet implemented"

# Development setup
.PHONY: setup
setup:
	@echo "🔧 Setting up development environment..."
	@echo "Installing dependencies..."
ifeq ($(PLATFORM),linux)
	@echo "Linux development setup..."
	@which apt-get > /dev/null 2>&1 && sudo apt-get update && sudo apt-get install -y build-essential cmake gdb valgrind cppcheck clang-tidy clang-format doxygen || echo "Manual dependency installation required"
endif
ifeq ($(PLATFORM),macos)
	@echo "macOS development setup..."
	@which brew > /dev/null 2>&1 && brew install cmake llvm cppcheck clang-format doxygen || echo "Please install Homebrew and dependencies manually"
endif
	@echo "✅ Development setup complete"

# Show compiler version and features
.PHONY: compiler-info
compiler-info:
	@echo "🔧 COMPILER INFORMATION:"
	@$(CXX) --version
	@echo ""
	@echo "C++20 feature support:"
	@echo "  concepts: $(shell echo '#include <concepts>' | $(CXX) -x c++ -std=c++20 -E - > /dev/null 2>&1 && echo 'YES' || echo 'NO')"
	@echo "  ranges: $(shell echo '#include <ranges>' | $(CXX) -x c++ -std=c++20 -E - > /dev/null 2>&1 && echo 'YES' || echo 'NO')"
	@echo "  coroutines: $(shell echo '#include <coroutine>' | $(CXX) -x c++ -std=c++20 -E - > /dev/null 2>&1 && echo 'YES' || echo 'NO')"
	@echo "  modules: $(shell echo 'export module test;' | $(CXX) -x c++ -std=c++20 -E - > /dev/null 2>&1 && echo 'YES' || echo 'NO')"

# Check dependencies
.PHONY: check-deps
check-deps:
	@echo "🔍 DEPENDENCY CHECK:"
	@echo "  GCC/G++: $(shell which $(CXX) > /dev/null 2>&1 && echo '✅ Found' || echo '❌ Missing')"
	@echo "  Make: $(shell which make > /dev/null 2>&1 && echo '✅ Found' || echo '❌ Missing')"
	@echo "  Git: $(shell which git > /dev/null 2>&1 && echo '✅ Found' || echo '❌ Missing')"
	@echo "  CMake: $(shell which cmake > /dev/null 2>&1 && echo '✅ Found' || echo '❌ Missing')"
	@echo "  Valgrind: $(shell which valgrind > /dev/null 2>&1 && echo '✅ Found' || echo '❌ Missing')"
	@echo "  Cppcheck: $(shell which cppcheck > /dev/null 2>&1 && echo '✅ Found' || echo '❌ Missing')"
	@echo "  Clang-tidy: $(shell which clang-tidy > /dev/null 2>&1 && echo '✅ Found' || echo '❌ Missing')"
	@echo "  Clang-format: $(shell which clang-format > /dev/null 2>&1 && echo '✅ Found' || echo '❌ Missing')"
	@echo "  Doxygen: $(shell which doxygen > /dev/null 2>&1 && echo '✅ Found' || echo '❌ Missing')"

.DEFAULT_GOAL := all