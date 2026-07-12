message("Building with tests.")

find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(ProjectNameTests
    test_projectname.cpp
)

target_link_libraries(ProjectNameTests
    ProjectName
    GTest::gtest_main
    ProjectName::compiler_options
)

gtest_discover_tests(ProjectNameTests)
