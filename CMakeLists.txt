cmake_minimum_required(VERSION 3.16)
project(mrc)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(visualization_msgs REQUIRED)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters registration)
find_package(pcl_conversions REQUIRED)

# Robot announcements are provided by hector_multi_robot on the platforms
find_package(hector_multi_robot_msgs REQUIRED)

option(MRC_USE_GTSAM "Enable pose graph optimization with GTSAM" ON)

# Configurable external library paths
set(MRC_TBB_LIB_DIR "/usr/local/lib" CACHE PATH "Path to TBB libraries (used by GTSAM)")
set(MRC_FRICP_DIR "$ENV{HOME}/workspace/Fast-Robust-ICP" CACHE PATH "Path to Fast-Robust-ICP source")
set(MRC_KISS_MATCHER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../KISS-Matcher" CACHE PATH "Path to KISS-Matcher source")

if(MRC_USE_GTSAM)
  # GTSAM was built with TBB - allow configurable path for portability
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${MRC_TBB_LIB_DIR} -Wl,-rpath-link,${MRC_TBB_LIB_DIR}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${MRC_TBB_LIB_DIR} -Wl,-rpath-link,${MRC_TBB_LIB_DIR}")
  find_package(GTSAM REQUIRED)
endif()

# Add KISS-Matcher as subdirectory
set(KISS_MATCHER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(KISS_MATCHER_INSTALL OFF CACHE BOOL "" FORCE)
set(KISS_MATCHER_USE_SYSTEM_EIGEN3 ON CACHE BOOL "" FORCE)
set(USE_SYSTEM_EIGEN3 ON CACHE BOOL "" FORCE)
# Use system TBB (same version GTSAM was built with)
get_filename_component(MRC_TBB_ROOT "${MRC_TBB_LIB_DIR}" DIRECTORY)
set(TBB_ROOT ${MRC_TBB_ROOT} CACHE PATH "" FORCE)
set(USE_SYSTEM_TBB ON CACHE BOOL "" FORCE)
set(USE_SYSTEM_ROBIN ON CACHE BOOL "" FORCE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(${MRC_KISS_MATCHER_DIR} ${CMAKE_CURRENT_BINARY_DIR}/kiss_matcher)

rosidl_generate_interfaces(${PROJECT_NAME}
        "msg/KissFeatures.msg"
        "srv/GetRegistrationData.srv"
        "action/Calibrate.action"
        DEPENDENCIES std_msgs builtin_interfaces sensor_msgs geometry_msgs
)

include_directories(include)

add_library(mrc_backends
        src/registration_factory.cpp
        src/backends/fricp_backend.cpp
        src/backends/kiss_matcher_backend.cpp
        src/icp_refiner.cpp
)

target_include_directories(mrc_backends PRIVATE
        ${EIGEN3_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${MRC_FRICP_DIR}
        ${MRC_FRICP_DIR}/include
        ${MRC_KISS_MATCHER_DIR}/core
)

ament_target_dependencies(mrc_backends
        rclcpp
        sensor_msgs
        geometry_msgs
        pcl_conversions
)

target_link_libraries(mrc_backends
        ${PCL_LIBRARIES}
        kiss_matcher::kiss_matcher_core
)

rosidl_target_interfaces(mrc_backends
        ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

set(MRC_MASTER_SOURCES src/mrc_master_node.cpp)
if(MRC_USE_GTSAM)
  list(APPEND MRC_MASTER_SOURCES src/pose_graph/pose_graph_optimizer.cpp)
endif()

add_executable(mrc_master ${MRC_MASTER_SOURCES})

ament_target_dependencies(mrc_master
        rclcpp
        rclcpp_action
        std_srvs
        tf2_ros
        hector_multi_robot_msgs
        visualization_msgs
)

target_link_libraries(mrc_master mrc_backends)

if(MRC_USE_GTSAM)
  target_include_directories(mrc_master PRIVATE ${GTSAM_INCLUDE_DIR})
  # Find TBB libraries in the configured path
  find_library(TBB_LIBRARY NAMES tbb PATHS ${MRC_TBB_LIB_DIR} NO_DEFAULT_PATH)
  find_library(TBB_MALLOC_LIBRARY NAMES tbbmalloc PATHS ${MRC_TBB_LIB_DIR} NO_DEFAULT_PATH)
  if(NOT TBB_LIBRARY)
    # Fallback to versioned library names
    file(GLOB TBB_LIBRARY "${MRC_TBB_LIB_DIR}/libtbb.so*")
    list(GET TBB_LIBRARY 0 TBB_LIBRARY)
  endif()
  if(NOT TBB_MALLOC_LIBRARY)
    file(GLOB TBB_MALLOC_LIBRARY "${MRC_TBB_LIB_DIR}/libtbbmalloc.so*")
    list(GET TBB_MALLOC_LIBRARY 0 TBB_MALLOC_LIBRARY)
  endif()
  # Link TBB BEFORE gtsam to satisfy its dependencies
  target_link_libraries(mrc_master
    ${TBB_LIBRARY}
    ${TBB_MALLOC_LIBRARY}
    gtsam
  )
  target_compile_definitions(mrc_master PRIVATE MRC_USE_GTSAM=1)
  # Ensure correct TBB is found at runtime
  set_target_properties(mrc_master PROPERTIES
    BUILD_RPATH "${MRC_TBB_LIB_DIR}"
    INSTALL_RPATH "${MRC_TBB_LIB_DIR}"
  )
endif()

rosidl_target_interfaces(mrc_master
        ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

add_executable(mrc_slave
        src/mrc_slave_node.cpp
)

target_include_directories(mrc_slave PRIVATE
        ${EIGEN3_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${MRC_KISS_MATCHER_DIR}/core
)

ament_target_dependencies(mrc_slave
        rclcpp
        sensor_msgs
        pcl_conversions
        tf2_ros
        tf2_sensor_msgs
)

target_link_libraries(mrc_slave
        ${PCL_LIBRARIES}
        kiss_matcher::kiss_matcher_core
)

rosidl_target_interfaces(mrc_slave
        ${PROJECT_NAME} "rosidl_typesupport_cpp"
)

# Install executables to lib/${PROJECT_NAME} (standard for ROS 2)
install(TARGETS
        mrc_master
        mrc_slave
        DESTINATION lib/${PROJECT_NAME}
)

# Install library to lib/ so it can be found in LD_LIBRARY_PATH
install(TARGETS mrc_backends
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
        DESTINATION include/
)

install(DIRECTORY launch config
        DESTINATION share/${PROJECT_NAME}
)

ament_package()