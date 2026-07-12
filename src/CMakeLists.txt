add_library(Bora STATIC
    bora.cpp
    kulfan.cpp
    network.cpp
    neuralfoil.cpp
)

# Baked-in default locations for the neural network weight data. The install
# location is tried first at runtime, then the source tree (for build-tree use).
target_compile_definitions(Bora PRIVATE
    BORA_INSTALL_DATA_DIR="${CMAKE_INSTALL_FULL_DATADIR}/bora"
    BORA_SOURCE_DATA_DIR="${PROJECT_SOURCE_DIR}/data"
)

# Create namespaced alias for use in build tree and by downstream projects
add_library(Bora::Bora ALIAS Bora)

target_link_libraries(Bora
    PRIVATE project_compiler_options
)

target_include_directories(Bora
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set_target_properties(Bora PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME Bora
)

# Install library
install(TARGETS Bora
    EXPORT BoraTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

# Install headers
install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
)

# Install neural network weight data
install(DIRECTORY "${PROJECT_SOURCE_DIR}/data/"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/bora"
    FILES_MATCHING PATTERN "*.bnn" PATTERN "*.bid"
)
