// stereo_triangulator.cpp
#include "stereo_vio/stereo_triangulator.hpp"
#include <Eigen/SVD>
#include <cmath>

namespace stereo_vio {

StereoTriangulator::StereoTriangulator(const StereoCalibration& calib,
                                       const TriangulationParams& params)
  : calib_(calib), params_(params) {}

void StereoTriangulator::setCalibration(const StereoCalibration& calib) { calib_ = calib; }
void StereoTriangulator::setParams(const TriangulationParams& params)    { params_ = params; }

std::unordered_map<uint64_t, TriangulationResult>
StereoTriangulator::triangulate(const FeatureFrame& frame) const
{
  std::unordered_map<uint64_t, TriangulationResult> results;
  results.reserve(frame.features.size());
  for (const auto& feature : frame.features) {
    if (!feature.has_stereo) continue;
    TriangulationResult result =
        triangulateSingle(feature.id, feature.pt_left, feature.pt_right);
    if (result.valid)
      results.emplace(feature.id, std::move(result));
  }
  return results;
}

TriangulationResult StereoTriangulator::triangulateSingle(
    uint64_t track_id,
    const cv::Point2f& pt_left,
    const cv::Point2f& pt_right) const
{
  TriangulationResult result;
  result.track_id = track_id;
  result.valid    = false;

  Eigen::Vector3d point_cam;
  if (!solveDLT(pt_left, pt_right, point_cam)) return result;
  if (!passesCheirality(point_cam))             return result;

  const double err_left  = reprojectionError(calib_.P0, point_cam, pt_left);
  const double err_right = reprojectionError(calib_.P1, point_cam, pt_right);

  if (!passesFilter(point_cam, err_left, err_right)) return result;

  result.point_cam          = point_cam;
  result.reproj_error_left  = err_left;
  result.reproj_error_right = err_right;
  result.valid              = true;
  return result;
}

bool StereoTriangulator::solveDLT(const cv::Point2f& pt_left,
                                  const cv::Point2f& pt_right,
                                  Eigen::Vector3d& point_out) const
{
  Eigen::Matrix<double,4,4> A;
  A.row(0) = (pt_left.x  * calib_.P0.row(2) - calib_.P0.row(0)).eval();
  A.row(1) = (pt_left.y  * calib_.P0.row(2) - calib_.P0.row(1)).eval();
  A.row(2) = (pt_right.x * calib_.P1.row(2) - calib_.P1.row(0)).eval();
  A.row(3) = (pt_right.y * calib_.P1.row(2) - calib_.P1.row(1)).eval();

  Eigen::JacobiSVD<Eigen::Matrix<double,4,4>> svd(A, Eigen::ComputeFullV);
  const Eigen::Vector4d X_h = svd.matrixV().col(3);

  constexpr double kEps = 1e-8;
  if (std::abs(X_h(3)) < kEps) return false;

  point_out = X_h.head<3>() / X_h(3);
  return true;
}

double StereoTriangulator::reprojectionError(
    const Eigen::Matrix<double,3,4>& P,
    const Eigen::Vector3d& point_cam,
    const cv::Point2f& pt_obs) const
{
  Eigen::Vector4d X_h;
  X_h << point_cam, 1.0;
  Eigen::Vector3d x_h = P * X_h;
  if (std::abs(x_h(2)) < 1e-8) return 1e9;
  const double du = x_h(0)/x_h(2) - static_cast<double>(pt_obs.x);
  const double dv = x_h(1)/x_h(2) - static_cast<double>(pt_obs.y);
  return std::sqrt(du*du + dv*dv);
}

bool StereoTriangulator::passesCheirality(const Eigen::Vector3d& point_cam) const
{
  return point_cam.z() > 0.0 && calib_.baseline > 0.0;
}

bool StereoTriangulator::passesFilter(const Eigen::Vector3d& point_cam,
                                      double reproj_left,
                                      double reproj_right) const
{
  const double Z = point_cam.z();
  if (Z < params_.min_depth || Z > params_.max_depth)           return false;
  if (reproj_left  > params_.reprojection_threshold)            return false;
  if (reproj_right > params_.reprojection_threshold)            return false;
  return true;
}

}  // namespace stereo_vio
