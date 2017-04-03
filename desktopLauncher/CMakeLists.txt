set(LAUNCHER_NAME SwapFaceDesktopLauncher)
project(${LAUNCHER_NAME})
message("Launcher name: ${LAUNCHER_NAME}")

set(SRC_DIR ${CMAKE_CURRENT_LIST_DIR})

set(SRC 
    ${SRC_DIR}/desktopLauncher.cpp
)

add_executable(${LAUNCHER_NAME} ${SRC})

include_directories(
    ${SwapFace_SOURCE_DIR}
)

target_link_libraries(${LAUNCHER_NAME} 
                      ${OpenCV_LIBS} 
                      SwapFace
)

set( THIRD_PARTY_DLLS
   ${OpenCV_DIR}/x64/vc14/bin/opencv_world320.dll
   ${OpenCV_DIR}/x64/vc14/bin/opencv_world320d.dll
)

# copy dll
foreach( file_i ${THIRD_PARTY_DLLS})
    add_custom_command(
    TARGET ${LAUNCHER_NAME}                                     
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    -E copy_if_different         			     
        ${file_i}                                          
        $<TARGET_FILE_DIR:${LAUNCHER_NAME}>
)
endforeach( file_i )	