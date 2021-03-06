cmake_minimum_required(VERSION 3.1...3.15)
if(${CMAKE_VERSION} VERSION_LESS 3.12)
	cmake_policy(VERSION ${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION})
endif()

project(
	Cubature
	VERSION 0.0.1
	DESCRIPTION "Library for the computation of multidimensional integrals using cubature."
	LANGUAGES CXX)

# Cubature library.
add_library(cubature INTERFACE)
target_sources(cubature INTERFACE cubature.ipp cubature.hpp)
add_library(Cubature::cubature ALIAS cubature)
target_include_directories(cubature INTERFACE ".")

# Test programs.
add_executable(htest test.cpp)
target_link_libraries(htest PRIVATE cubature m)
target_compile_features(htest PRIVATE cxx_std_11)
set_target_properties(htest PROPERTIES CXX_EXTENSIONS OFF)
target_compile_options(
	htest
	PRIVATE
	$<$<OR:$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:-Wall -Wextra -pedantic>
	$<$<CXX_COMPILER_ID:MSVC>:/W4>)

# Installation commands.
include(GNUInstallDirs)
install(
	FILES cubature.hpp cubature.ipp
	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/cubature")
install(
	TARGETS cubature
	LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
	INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

