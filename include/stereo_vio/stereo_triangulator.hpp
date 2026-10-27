#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include "stereo_vio/feature_tracker.hpp"

namespace stereo_vio {

struct StereoCalibration {
  Eigen::Matrix<double,3,4> P0 = Eigen::Matrix<double,3,4>::Zero();
  Eigen::Matrix<double,3,4> P1 = Eigen::Matrix<double,3,4>::Zero();
  double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0, baseline = 0.0;
  void computeFromProjectionMatrices() {
    fx       =  P0(0,0);
    fy       =  P0(1,1);
    cx       =  P0(0,2);
    cy       =  P0(1,2);
    baseline = -P1(0,3) / fx;
  }
};

struct TriangulationParams {
  double reprojection_threshold = 2.0;
  double min_depth              = 0.2;
  double max_depth              = 20.0;
};

struct TriangulationResult {
  uint64_t        track_id   = 0;
  Eigen::Vector3d point_cam;
  Eigen::Vector3d point_world;      // ← add
  bool            has_prior  = false; // ← add
  bool            valid      = false;
  double          reproj_error_left  = 0.0;
  double          reproj_error_right = 0.0;
};

class StereoTriangulator {
public:
  StereoTriangulator() = default;
  StereoTriangulator(const StereoCalibration& calib,
                     const TriangulationParams& params);
  void setCalibration(const StereoCalibration& calib);
  void setParams(const TriangulationParams& params);

  std::unordered_map<uint64_t, TriangulationResult>
  triangulate(const FeatureFrame& frame) const;

  TriangulationResult triangulateSingle(
      uint64_t track_id,
      const cv::Point2f& pt_left,
      const cv::Point2f& pt_right) const;

private:
  bool solveDLT(const cv::Point2f& pt_left,
                const cv::Point2f& pt_right,
                Eigen::Vector3d& point_out) const;
  double reprojectionError(
      const Eigen::Matrix<double,3,4>& P,
      const Eigen::Vector3d& point_cam,
      const cv::Point2f& pt_obs) const;
  bool passesCheirality(const Eigen::Vector3d& point_cam) const;
  bool passesFilter(const Eigen::Vector3d& point_cam,
                    double reproj_left,
                    double reproj_right) const;

  StereoCalibration   calib_;
  TriangulationParams params_;
};

}  // namespace stereo_vio
