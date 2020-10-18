cmake_minimum_required(VERSION 3.1...3.15)
if(${CMAKE_VERSION} VERSION_LESS 3.12)
	cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

project(
	Cubature
	VERSION 0.0.1
	DESCRIPTION "Library for the computation of multidimensional integrals using cubature."
	LANGUAGES C)

# Cubature library.
add_library(
	cubature
	cubature.c
	cubature.h)
add_library(Cubature::cubature ALIAS cubature)
target_include_directories(cubature PUBLIC ".")

# Test programs.
add_executable(test test.c)
target_link_libraries(test PRIVATE cubature m)

# Installation commands.
include(GNUInstallDirs)
set_target_properties(
	cubature PROPERTIES
	INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}"
	INSTALL_RPATH_USE_LINK_PATH TRUE)
install(
	DIRECTORY "${Cubature_SOURCE_DIR}"
	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/cubature"
	FILES_MATCHING
		PATTERN "*.h")
install(
	TARGETS cubature
	LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

