cmake_minimum_required(VERSION 3.8)
project(semantic_simulation)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  config
  maps
  worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
