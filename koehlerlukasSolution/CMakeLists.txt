cmake_minimum_required(VERSION 3.10)
project(project)

set(CMAKE_CXX_STANDARD 11)
set(GUROBI_LIBRARIES "${GUROBI_CXX_LIBRARY};${GUROBI_LIBRARY}" )

include_directories(./)

find_library( GUROBI_LIBRARY
        NAMES gurobi80
        PATHS "./"
        )

find_library( GUROBI_CXX_LIBRARY
        NAMES gurobi_g++5.2
        PATHS "./"
        )

add_executable(project main.cpp)

target_link_libraries(project ${GUROBI_LIBRARIES})