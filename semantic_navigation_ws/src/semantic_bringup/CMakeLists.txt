cmake_minimum_required(VERSION 3.8)
project(semantic_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  config
  launch
  models
  worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
