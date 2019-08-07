# Install script for directory: /mnt/c/Users/qyxs15/CLionProjects/HyperbolicEmbedding

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libHyperbolicEmbedding.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libHyperbolicEmbedding.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libHyperbolicEmbedding.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/mnt/c/Users/qyxs15/CLionProjects/HyperbolicEmbedding/libHyperbolicEmbedding.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libHyperbolicEmbedding.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libHyperbolicEmbedding.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libHyperbolicEmbedding.so")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/mnt/c/Users/qyxs15/CLionProjects/HyperbolicEmbedding/libHyperbolicEmbedding.a")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/mnt/c/Users/qyxs15/CLionProjects/HyperbolicEmbedding/libHyperbolicEmbedding_pic.a")
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/HyperbolicEmbedding" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/HyperbolicEmbedding")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/HyperbolicEmbedding"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/mnt/c/Users/qyxs15/CLionProjects/HyperbolicEmbedding/HyperbolicEmbedding")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/HyperbolicEmbedding" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/HyperbolicEmbedding")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/HyperbolicEmbedding")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/HyperbolicEmbedding" TYPE FILE FILES
    "/mnt/c/Users/qyxs15/CLionProjects/HyperbolicEmbedding/src/args.h"
    "/mnt/c/Users/qyxs15/CLionProjects/HyperbolicEmbedding/src/densematrix.h"
    "/mnt/c/Users/qyxs15/CLionProjects/HyperbolicEmbedding/src/dictionary.h"
    "/mnt/c/Users/qyxs15/CLionProjects/HyperbolicEmbedding/src/fasttext.h"
    "/mnt/c/Users/qyxs15/CLionProjects/HyperbolicEmbedding/src/loss.h"
    "/mnt/c/Users/qyxs15/CLionProjects/HyperbolicEmbedding/src/matrix.h"
    "/mnt/c/Users/qyxs15/CLionProjects/HyperbolicEmbedding/src/meter.h"
    "/mnt/c/Users/qyxs15/CLionProjects/HyperbolicEmbedding/src/model.h"
    "/mnt/c/Users/qyxs15/CLionProjects/HyperbolicEmbedding/src/productquantizer.h"
    "/mnt/c/Users/qyxs15/CLionProjects/HyperbolicEmbedding/src/quantmatrix.h"
    "/mnt/c/Users/qyxs15/CLionProjects/HyperbolicEmbedding/src/real.h"
    "/mnt/c/Users/qyxs15/CLionProjects/HyperbolicEmbedding/src/utils.h"
    "/mnt/c/Users/qyxs15/CLionProjects/HyperbolicEmbedding/src/vector.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/mnt/c/Users/qyxs15/CLionProjects/HyperbolicEmbedding/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
