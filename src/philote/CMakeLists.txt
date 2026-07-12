# Philote discipline interface (built when BORA_BUILD_PHILOTE=ON)

add_library(BoraPhilote STATIC
    philote_discipline.cpp
)

add_library(Bora::Philote ALIAS BoraPhilote)

target_link_libraries(BoraPhilote
    PUBLIC
        Bora
        PhiloteCpp::PhiloteCpp
    PRIVATE
        project_compiler_options
)

set_target_properties(BoraPhilote PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    EXPORT_NAME Philote
)

# Discipline server executable
add_executable(bora_server server.cpp)

target_link_libraries(bora_server
    PRIVATE
        BoraPhilote
        Bora::compiler_options
)

install(TARGETS BoraPhilote bora_server
    EXPORT BoraTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
