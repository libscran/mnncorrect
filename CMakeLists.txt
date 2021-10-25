cmake_minimum_required(VERSION 3.14)

project(mnncorrect
    VERSION 1.0.0
    DESCRIPTION "MNN correction in C++"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

add_library(mnncorrect INTERFACE)

target_include_directories(mnncorrect INTERFACE include/)

add_subdirectory(extern)

target_link_libraries(mnncorrect INTERFACE knncolle aarand)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif()
endif()
