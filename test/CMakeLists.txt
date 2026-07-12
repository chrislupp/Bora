message("Building with tests.")

find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(BoraTests
    test_bora.cpp
    test_gradients.cpp
    test_kulfan.cpp
    test_neuralfoil.cpp
)

target_link_libraries(BoraTests
    Bora
    GTest::gtest_main
    Bora::compiler_options
)

gtest_discover_tests(BoraTests)
