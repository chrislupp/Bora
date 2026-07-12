message("Building with tests.")

find_package(GTest REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
include(GoogleTest)

add_executable(BoraTests
    test_bora.cpp
    test_reference_gradients.cpp
    test_gradients.cpp
    test_kulfan.cpp
    test_neuralfoil.cpp
)

target_link_libraries(BoraTests
    Bora
    GTest::gtest_main
    HDF5::HDF5
    Bora::compiler_options
)

target_compile_definitions(BoraTests PRIVATE
    BORA_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

gtest_discover_tests(BoraTests)

if(BORA_BUILD_PHILOTE)
    add_executable(BoraPhiloteTests
        test_philote_discipline.cpp
    )

    target_link_libraries(BoraPhiloteTests
        Bora::Philote
        GTest::gtest_main
        Bora::compiler_options
    )

    gtest_discover_tests(BoraPhiloteTests)
endif()
