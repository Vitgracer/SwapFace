set(LIB_NAME SwapFace)
project(${LIB_NAME})
message("Library name: ${LIB_NAME}")

set(SRC_DIR ${CMAKE_CURRENT_LIST_DIR})

set(SRC 
    ${SRC_DIR}/SwapFace.cpp
)

set(HDR 
    ${SRC_DIR}/SwapFace.h
    ${SRC_DIR}/SwapFace.h
)

include_directories(
    ${SRC_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

add_library(${LIB_NAME} STATIC ${SRC} ${HDR})
target_link_libraries(${LIB_NAME} ${OpenCV_LIBS})
