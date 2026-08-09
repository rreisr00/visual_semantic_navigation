cmake_minimum_required(VERSION 3.8)
project(semantic_bringup)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

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
  scripts/wait_for_localization.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES README.md DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_scene_config test/test_scene_config.py)
endif()

ament_package()
