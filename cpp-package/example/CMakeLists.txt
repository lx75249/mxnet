if(NOT MSVC)
  set(UNITTEST_STATIC_LINK ON)
endif()

set(CPP_EXAMPLE_LIBS
  ${BEGIN_WHOLE_ARCHIVE} mxnet ${END_WHOLE_ARCHIVE}
  ${mxnet_LINKER_LIBS}
  )

set(CPP_PACKAGE_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/../include/mxnet-cpp/)

set(CPPEX_DEPS cpp_package_op_h)

file(GLOB_RECURSE CPP_PACKAGE_HEADERS
  "${CPP_PACKAGE_INCLUDE_DIR}/*.h"
  "${CPP_PACKAGE_INCLUDE_DIR}/*.hpp"
  )

add_custom_target(
  cpp_package_deploy_library ALL
  DEPENDS mxnet
  COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:mxnet> $<TARGET_FILE_DIR:mlp>
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)

add_executable(lenet_with_mxdataiter lenet_with_mxdataiter.cpp ${CPP_PACKAGE_HEADERS})
target_link_libraries(lenet_with_mxdataiter ${CPP_EXAMPLE_LIBS})
add_dependencies(lenet_with_mxdataiter ${CPPEX_DEPS})

add_executable(mlp mlp_gpu.cpp ${CPP_PACKAGE_HEADERS})
target_link_libraries(mlp ${CPP_EXAMPLE_LIBS})
add_dependencies(mlp ${CPPEX_DEPS})

add_executable(vdsr main.cpp ${CPP_PACKAGE_HEADERS})
target_link_libraries(vdsr ${CPP_EXAMPLE_LIBS})
add_dependencies(vdsr ${CPPEX_DEPS})
