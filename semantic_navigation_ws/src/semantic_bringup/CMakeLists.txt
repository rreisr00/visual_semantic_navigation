cmake_minimum_required(VERSION 3.8)
project(semantic_bringup)

find_package(ament_cmake REQUIRED)

install(DIRECTORY
  config
  launch
  maps
  models
  worlds
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/smoke_test.sh
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES README.md DESTINATION share/${PROJECT_NAME})

ament_package()
