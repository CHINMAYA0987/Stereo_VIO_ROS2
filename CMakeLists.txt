cmake_minimum_required(VERSION 3.8)
project(stereo_vio)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# =============================================================================
# Dependencies
# =============================================================================
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(Sophus REQUIRED)

# =============================================================================
# Core VIO library
# (imu_integrator, feature_tracker, stereo_triangulator, eskf_estimator)
# =============================================================================
add_library(stereo_vio_core
  src/imu_integrator.cpp
  src/feature_tracker.cpp
  src/stereo_triangulator.cpp
  src/eskf_estimator.cpp
)

# Headers for this library and anything linking against it
target_include_directories(stereo_vio_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

ament_target_dependencies(stereo_vio_core
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  cv_bridge
  image_transport
  message_filters
  tf2
  tf2_ros
  tf2_geometry_msgs
)

target_link_libraries(stereo_vio_core
  ${OpenCV_LIBS}
  Eigen3::Eigen
  Sophus::Sophus
)

# =============================================================================
# Main ROS 2 node executable
# =============================================================================
add_executable(stereo_vio_node
  src/stereo_vio_node.cpp
)

target_include_directories(stereo_vio_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(stereo_vio_node
  rclcpp
  sensor_msgs
  geometry_msgs
  nav_msgs
  cv_bridge
  image_transport
  message_filters
  tf2
  tf2_ros
  tf2_geometry_msgs
)

target_link_libraries(stereo_vio_node
  stereo_vio_core
  ${OpenCV_LIBS}
  Eigen3::Eigen
  Sophus::Sophus
)

# =============================================================================
# Install
# =============================================================================
install(TARGETS stereo_vio_core stereo_vio_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

# =============================================================================
# Testing
# =============================================================================
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()