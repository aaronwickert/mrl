cmake_minimum_required(VERSION 3.8)
project(mrl)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS
  common
  io
  filters
  features
  kdtree
)
find_package(rosidl_typesupport_cpp REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(PCL REQUIRED)
### To include ROBIN& KISS-Matcher ###
find_package(robin REQUIRED)
find_package(kiss_matcher REQUIRED)



link_directories(${PCL_LIBRARY_DIRS})
include_directories(include
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS})

set(DEPS rclcpp rosidl_typesupport_cpp Eigen3 PCL robin kiss_matcher sensor_msgs std_msgs pcl_conversions geometry_msgs tf2 tf2_ros tf2_geometry_msgs)

add_executable(${PROJECT_NAME}_master src/mrl.cpp src/mrl_master.cpp)
ament_target_dependencies(${PROJECT_NAME}_master ${DEPS})
target_link_libraries(${PROJECT_NAME}_master kiss_matcher::kiss_matcher_core)

add_executable(${PROJECT_NAME}_slave src/mrl.cpp src/mrl_slave.cpp)
ament_target_dependencies(${PROJECT_NAME}_slave ${DEPS})
target_link_libraries(${PROJECT_NAME}_slave kiss_matcher::kiss_matcher_core)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SlaveAnnouncement.msg"
  "srv/GetFeatures.srv"
  "srv/ExecuteMRL.srv"
  "srv/GetReady.srv"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs
 )

ament_export_dependencies(rosidl_default_runtime)
rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")
target_link_libraries(${PROJECT_NAME}_master "${cpp_typesupport_target}")
target_link_libraries(${PROJECT_NAME}_slave "${cpp_typesupport_target}")


ament_export_dependencies(rosidl_default_runtime)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()



install(TARGETS 
  ${PROJECT_NAME}_master
  ${PROJECT_NAME}_slave
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config 
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
