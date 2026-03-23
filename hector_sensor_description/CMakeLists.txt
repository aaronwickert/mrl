cmake_minimum_required(VERSION 3.16)
project(hector_sensors_description)

find_package(ament_cmake REQUIRED)

# Install URDF/xacro files
install(DIRECTORY urdf gazebo meshes
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
