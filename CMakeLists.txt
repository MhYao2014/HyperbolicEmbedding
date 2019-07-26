#
# Copyright (c) 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

cmake_minimum_required(VERSION 2.8.9)
project(HyperbolicEmbedding)

# The version number.
set (HyperbolicEmbedding_VERSION_MAJOR 0)
set (HyperbolicEmbedding_VERSION_MINOR 1)

include_directories(./)

set(CMAKE_CXX_FLAGS " -pthread -std=c++11 -funroll-loops -O3 -march=native")

set(HEADER_FILES
    src/args.h
    src/densematrix.h
    src/dictionary.h
    src/fasttext.h
    src/loss.h
    src/matrix.h
    src/meter.h
    src/model.h
    src/productquantizer.h
    src/quantmatrix.h
    src/real.h
    src/utils.h
    src/vector.h)

set(SOURCE_FILES
    src/args.cc
    src/densematrix.cc
    src/dictionary.cc
    src/fasttext.cc
    src/loss.cc
    src/main.cc
    src/matrix.cc
    src/meter.cc
    src/model.cc
    src/productquantizer.cc
    src/quantmatrix.cc
    src/utils.cc
    src/vector.cc)

add_library(HyperbolicEmbedding-shared SHARED ${SOURCE_FILES} ${HEADER_FILES})
add_library(HyperbolicEmbedding-static STATIC ${SOURCE_FILES} ${HEADER_FILES})
add_library(HyperbolicEmbedding-static_pic STATIC ${SOURCE_FILES} ${HEADER_FILES})
set_target_properties(HyperbolicEmbedding-shared PROPERTIES OUTPUT_NAME HyperbolicEmbedding)
set_target_properties(HyperbolicEmbedding-static PROPERTIES OUTPUT_NAME HyperbolicEmbedding)
set_target_properties(HyperbolicEmbedding-static_pic PROPERTIES OUTPUT_NAME HyperbolicEmbedding_pic
  POSITION_INDEPENDENT_CODE True)
add_executable(HyperbolicEmbedding-bin src/main.cc)
target_link_libraries(HyperbolicEmbedding-bin pthread HyperbolicEmbedding-static)
set_target_properties(HyperbolicEmbedding-bin PROPERTIES PUBLIC_HEADER "${HEADER_FILES}" OUTPUT_NAME HyperbolicEmbedding)
install (TARGETS HyperbolicEmbedding-shared
    LIBRARY DESTINATION lib)
install (TARGETS HyperbolicEmbedding-static
    ARCHIVE DESTINATION lib)
install (TARGETS HyperbolicEmbedding-static_pic
    ARCHIVE DESTINATION lib)
install (TARGETS HyperbolicEmbedding-bin
    RUNTIME DESTINATION bin
 PUBLIC_HEADER DESTINATION include/HyperbolicEmbedding)
