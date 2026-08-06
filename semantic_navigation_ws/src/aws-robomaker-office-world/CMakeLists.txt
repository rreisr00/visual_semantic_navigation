cmake_minimum_required(VERSION 3.8)
project(aws_robomaker_office_world)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY maps media models worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
