cmake_minimum_required(VERSION 3.16)
project(mrc_test)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(rclcpp_action REQUIRED)
find_package(mrc REQUIRED)

add_executable(mrc_evaluator
  src/mrc_evaluator_node.cpp
)

ament_target_dependencies(mrc_evaluator
  rclcpp
  rclcpp_action
  std_msgs
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
  std_srvs
  rcl_interfaces
  mrc
)

install(TARGETS
  mrc_evaluator
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch worlds config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
