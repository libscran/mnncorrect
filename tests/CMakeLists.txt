include(FetchContent)
FetchContent_Declare(
  scran_tests
  GIT_REPOSITORY https://github.com/libscran/scran_tests
  GIT_TAG master
)
FetchContent_MakeAvailable(scran_tests)

option(CODE_COVERAGE "Enable coverage testing" OFF)
set(DO_CODE_COVERAGE OFF)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(DO_CODE_COVERAGE ON)
endif()

include(GoogleTest)

macro(create_test name) 
    add_executable(
        ${name} 
        src/fuse_nn_results.cpp
        src/find_mutual_nns.cpp
        src/robust_average.cpp
        src/correct_target.cpp
        src/AutomaticOrder.cpp
        src/CustomOrder.cpp
        src/restore_order.cpp
        src/utils.cpp
#        src/MnnCorrect.cpp
    )

    target_link_libraries(
        ${name}
        scran_tests
        mnncorrect 
    )

    target_compile_options(${name} PRIVATE -Wall -Wpedantic -Wextra -Werror)

    if(DO_CODE_COVERAGE)
        target_compile_options(${name} PRIVATE -O0 -g --coverage)
        target_link_options(${name} PRIVATE --coverage)
    endif()

    gtest_discover_tests(${name})
endmacro()

create_test(libtest)

create_test(cuspartest)
target_compile_definitions(cuspartest PRIVATE TEST_CUSTOM_PARALLEL)

find_package(OpenMP)
if (OpenMP_FOUND)
    create_test(omptest)
    target_link_libraries(omptest OpenMP::OpenMP_CXX)
endif()
