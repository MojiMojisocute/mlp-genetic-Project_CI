cmake_minimum_required(VERSION 3.10)
project(MLP_GA_WDBC)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/main.cpp
    src/dataset.cc
    src/mlp.cc
    src/ga.cc
    src/utils.cc
    src/results.cc
)

# Header files (for IDEs)
set(HEADERS
    include/dataset.h
    include/mlp.h
    include/ga.h
    include/utils.h
    include/results.h
)

# Create executable
add_executable(mlp_ga_wdbc ${SOURCES} ${HEADERS})

# Link math library (for some systems)
target_link_libraries(mlp_ga_wdbc m)

# Installation
install(TARGETS mlp_ga_wdbc DESTINATION bin)

# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")