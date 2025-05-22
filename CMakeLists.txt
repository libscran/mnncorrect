cmake_minimum_required(VERSION 3.24)

project(mnncorrect
    VERSION 3.0.0
    DESCRIPTION "MNN correction in C++"
    LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Library
add_library(mnncorrect INTERFACE)
add_library(libscran::mnncorrect ALIAS mnncorrect)

target_include_directories(mnncorrect INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/mnncorrect>)
target_compile_features(mnncorrect INTERFACE cxx_std_17)

# Dependencies
option(MNNCORRECT_FETCH_EXTERN "Automatically fetch mnncorrect's external dependencies." ON)
if(MNNCORRECT_FETCH_EXTERN)
    add_subdirectory(extern)
else()
    find_package(knncolle_knncolle 3.0.0 CONFIG REQUIRED)
endif()

target_link_libraries(mnncorrect INTERFACE knncolle::knncolle)

# Tests
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    option(MNNCORRECT_TESTS "Build mnncorrect's test suite." ON)
else()
    option(MNNCORRECT_TESTS "Build mnncorrect's test suite." OFF)
endif()

if(MNNCORRECT_TESTS)
    include(CTest)
    if(BUILD_TESTING)
        add_subdirectory(tests)
    endif() 
endif()

# Install
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/mnncorrect)

install(TARGETS mnncorrect
    EXPORT mnncorrectTargets)

install(EXPORT mnncorrectTargets
    FILE libscran_mnncorrectTargets.cmake
    NAMESPACE libscran::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libscran_mnncorrect)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/libscran_mnncorrectConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libscran_mnncorrect)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/libscran_mnncorrectConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libscran_mnncorrectConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/libscran_mnncorrectConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/libscran_mnncorrect)
