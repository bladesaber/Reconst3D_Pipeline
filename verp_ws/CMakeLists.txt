cmake_minimum_required(VERSION 3.12)
project(zmqRemoteApi)

set(COPPELIASIM_ROOT_DIR CoppeliaSim_Edu_V4_3_0_rev12_Ubuntu18_04)
set(LIBPLUGIN_DIR ${COPPELIASIM_ROOT_DIR}/programming/libPlugin)

# if(NOT LIBPLUGIN_DIR)
#     if(DEFINED ENV{COPPELIASIM_ROOT_DIR})
#         set(LIBPLUGIN_DIR $ENV{COPPELIASIM_ROOT_DIR}/programming/libPlugin)
#     else()
#         set(LIBPLUGIN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../libPlugin)
#     endif()
# endif()

MESSAGE("COPPELIASIM_ROOT_DIR: ${COPPELIASIM_ROOT_DIR}")
MESSAGE("CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
MESSAGE("LIBPLUGIN_DIR: ${LIBPLUGIN_DIR}")

set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules
    ${LIBPLUGIN_DIR}/cmake)

find_package(CoppeliaSim 4.3.0.0 REQUIRED)

coppeliasim_add_addon("simAddOnZMQ remote API server.lua")
