cmake_minimum_required(VERSION 3.8)
project(semantic_robot_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config models
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
