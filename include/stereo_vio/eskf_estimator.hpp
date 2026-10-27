#pragma once

#include <Eigen/Dense>
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include "stereo_vio/imu_integrator.hpp"
#include "stereo_vio/stereo_triangulator.hpp"

namespace stereo_vio {

struct NominalState {
  Eigen::Vector3d  p  = Eigen::Vector3d::Zero();
  Eigen::Vector3d  v  = Eigen::Vector3d::Zero();
  Sophus::SO3d     R;
  Eigen::Vector3d  ba = Eigen::Vector3d::Zero();
  Eigen::Vector3d  bg = Eigen::Vector3d::Zero();
};

struct EskfParams {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double acc_noise         = 0.01;
  double gyro_noise        = 0.005;
  double acc_bias_noise    = 2.0e-4;
  double gyro_bias_noise   = 4.0e-6;
  double image_noise       = 1.5;
  double init_duration     = 2.0;
  double gravity_magnitude = 9.81;
  double chi2_threshold    = 5.991;
  Eigen::Matrix3d R_CB = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t_CB = Eigen::Vector3d::Zero();
  Eigen::Matrix3d R_rect0 = Eigen::Matrix3d::Identity();
};

struct EskfState {
  NominalState                 nominal;
  Eigen::Matrix<double,15,15>  P =
      Eigen::Matrix<double,15,15>::Identity() * 1e-4;
  double timestamp   = 0.0;
  bool   initialised = false;
};

class EskfEstimator {
public:
  explicit EskfEstimator(const EskfParams& params);
  void feedImuForInit(const ImuData& imu);
  bool tryInitialise();
  bool isInitialised() const { return state_.initialised; }
  void propagate(const PreintegrationResult& preint);
  void update(
      const std::unordered_map<uint64_t, TriangulationResult>& landmarks,
      const StereoCalibration& calib);
  const EskfState&  getState()  const { return state_; }
  EskfState&        getState()        { return state_; }
  const EskfParams& getParams() const { return params_; }
  void updatePose(const Eigen::Vector3d& p_meas,
                const Sophus::SO3d& R_meas,
                double pos_noise,
                double rot_noise);


private:
  Eigen::Matrix<double,15,15> buildTransitionJacobian(
      const Eigen::Vector3d& a_corrected,
      const Eigen::Vector3d& w_corrected,
      double dt) const;
  Eigen::Matrix<double,15,12> buildNoiseJacobian() const;
  Eigen::Matrix<double,2,15>  buildMeasurementJacobian(
      const Eigen::Vector3d& point_world,
      const StereoCalibration& calib) const;
  void injectErrorState(const Eigen::Matrix<double,15,1>& dx);
  void resetErrorState(const Eigen::Matrix<double,15,15>& IKH);
  static Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S <<    0, -v(2),  v(1),
         v(2),     0, -v(0),
        -v(1),  v(0),     0;
    return S;
  }
  
  EskfParams           params_;
  EskfState            state_;
  Eigen::Vector3d      gravity_ = Eigen::Vector3d::Zero();
  std::vector<ImuData> init_imu_buffer_;
};

}  // namespace stereo_vio
