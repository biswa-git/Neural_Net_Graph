include_directories(${CMAKE_SOURCE_DIR}/thirdParty/eigen-3.3.9)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../hpp)


file( GLOB SRCS_CPP *.cpp)
file( GLOB SRCS_HPP ../hpp/*.hpp)
message(hpp: ${SRCS_HPP})
add_executable(NN
     ${SRCS_HPP}
     ${SRCS_CPP}
)

install(TARGETS NN DESTINATION ${CMAKE_SOURCE_DIR}/exe)
