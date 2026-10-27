#pragma once

#include <Eigen/Dense>
#include <sophus/so3.hpp>
#include <sophus/se3.hpp>
#include <vector>

namespace stereo_vio {

struct ImuData {
  double timestamp;
  Eigen::Vector3d accel;
  Eigen::Vector3d gyro;
};

struct ImuNoise {
  double acc_noise       = 0.01;
  double gyro_noise      = 0.005;
  double acc_bias_noise  = 2.0e-4;
  double gyro_bias_noise = 4.0e-6;
};

struct PreintegrationResult {
  Sophus::SO3d              delta_R;
  Eigen::Vector3d           delta_v = Eigen::Vector3d::Zero();
  Eigen::Vector3d           delta_p = Eigen::Vector3d::Zero();
  double                    dt      = 0.0;
  Eigen::Matrix<double,15,15> P     = Eigen::Matrix<double,15,15>::Zero();
  Eigen::Matrix3d  d_delta_R_d_bg   = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d  d_delta_v_d_ba   = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d  d_delta_v_d_bg   = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d  d_delta_p_d_ba   = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d  d_delta_p_d_bg   = Eigen::Matrix3d::Zero();
};

class ImuIntegrator {
public:
  explicit ImuIntegrator(const ImuNoise& noise);
  void reset(const Eigen::Vector3d& ba, const Eigen::Vector3d& bg);
  void pushMeasurement(const ImuData& data);
  const PreintegrationResult& getResult() const { return result_; }
  void correctBias(const Eigen::Vector3d& d_ba, const Eigen::Vector3d& d_bg);

private:
  void integrateRK4(const ImuData& m0, const ImuData& m1);
  void propagateCovariance(const Eigen::Vector3d& a_mid,
                           const Eigen::Vector3d& w_mid,
                           double dt);
  static Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
    Eigen::Matrix3d S;
    S <<    0, -v(2),  v(1),
         v(2),     0, -v(0),
        -v(1),  v(0),     0;
    return S;
  }

  ImuNoise              noise_;
  Eigen::Vector3d       ba_ = Eigen::Vector3d::Zero();
  Eigen::Vector3d       bg_ = Eigen::Vector3d::Zero();
  PreintegrationResult  result_;
  ImuData               prev_meas_;
  bool                  has_prev_ = false;
};

}  // namespace stereo_vio
