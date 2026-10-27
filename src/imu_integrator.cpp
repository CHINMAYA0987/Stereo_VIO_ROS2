#include "stereo_vio/imu_integrator.hpp"
#include <stdexcept>
#include <cmath>

namespace stereo_vio {

ImuIntegrator::ImuIntegrator(const ImuNoise& noise)
  : noise_(noise), has_prev_(false)
{
  ba_.setZero();
  bg_.setZero();
}

void ImuIntegrator::reset(const Eigen::Vector3d& ba, const Eigen::Vector3d& bg)
{
  ba_       = ba;
  bg_       = bg;
  has_prev_ = false;
  result_   = PreintegrationResult();
}

void ImuIntegrator::pushMeasurement(const ImuData& data)
{
  if (!has_prev_) {
    prev_meas_ = data;
    has_prev_  = true;
    return;
  }
  if (data.timestamp <= prev_meas_.timestamp) {
    prev_meas_ = data;
    return;
  }
  integrateRK4(prev_meas_, data);
  prev_meas_ = data;
}

void ImuIntegrator::correctBias(const Eigen::Vector3d& d_ba,
                                const Eigen::Vector3d& d_bg)
{
  Eigen::Vector3d dtheta = result_.d_delta_R_d_bg * d_bg;
  result_.delta_R = result_.delta_R * Sophus::SO3d::exp(dtheta);
  result_.delta_v += result_.d_delta_v_d_ba * d_ba
                   + result_.d_delta_v_d_bg * d_bg;
  result_.delta_p += result_.d_delta_p_d_ba * d_ba
                   + result_.d_delta_p_d_bg * d_bg;
}

void ImuIntegrator::integrateRK4(const ImuData& m0, const ImuData& m1)
{
  const double dt = m1.timestamp - m0.timestamp;
  if (dt <= 0.0) return;

  const Eigen::Vector3d a0 = m0.accel - ba_;
  const Eigen::Vector3d w0 = m0.gyro  - bg_;
  const Eigen::Vector3d a1 = m1.accel - ba_;
  const Eigen::Vector3d w1 = m1.gyro  - bg_;
  const Eigen::Vector3d a_mid = 0.5 * (a0 + a1);
  const Eigen::Vector3d w_mid = 0.5 * (w0 + w1);

  Sophus::SO3d    R  = result_.delta_R;
  Eigen::Vector3d v  = result_.delta_v;

  // k1
  Eigen::Vector3d k1_v = R * a0;
  Eigen::Vector3d k1_p = v;

  // k2
  Sophus::SO3d    R2   = R * Sophus::SO3d::exp(w0 * (dt * 0.5));
  Eigen::Vector3d v2   = v + k1_v * (dt * 0.5);
  Eigen::Vector3d k2_v = R2 * a_mid;
  Eigen::Vector3d k2_p = v2;

  // k3
  Sophus::SO3d    R3   = R * Sophus::SO3d::exp(w_mid * (dt * 0.5));
  Eigen::Vector3d v3   = v + k2_v * (dt * 0.5);
  Eigen::Vector3d k3_v = R3 * a_mid;
  Eigen::Vector3d k3_p = v3;

  // k4
  Sophus::SO3d    R4   = R * Sophus::SO3d::exp(w_mid * dt);
  Eigen::Vector3d v4   = v + k3_v * dt;
  Eigen::Vector3d k4_v = R4 * a1;
  Eigen::Vector3d k4_p = v4;

  result_.delta_v += (dt / 6.0) * (k1_v + 2.0*k2_v + 2.0*k3_v + k4_v);
  result_.delta_p += (dt / 6.0) * (k1_p + 2.0*k2_p + 2.0*k3_p + k4_p);
  result_.delta_R  = result_.delta_R *
      Sophus::SO3d::exp((dt/6.0)*(w0 + 2.0*w_mid + 2.0*w_mid + w1));
  result_.dt += dt;

  const Eigen::Matrix3d R_mat = result_.delta_R.matrix();
  result_.d_delta_R_d_bg -= R_mat.transpose() *
                            Sophus::SO3d::exp(w_mid * dt).matrix() *
                            skew(w_mid) * dt;
  result_.d_delta_v_d_ba += R_mat * dt;
  result_.d_delta_v_d_bg -= R_mat * skew(a_mid) *
                            result_.d_delta_R_d_bg * dt;
  result_.d_delta_p_d_ba += result_.d_delta_v_d_ba * dt
                           + 0.5 * R_mat * dt * dt;
  result_.d_delta_p_d_bg += result_.d_delta_v_d_bg * dt
                           - 0.5 * R_mat * skew(a_mid) *
                             result_.d_delta_R_d_bg * dt * dt;

  propagateCovariance(a_mid, w_mid, dt);
}

void ImuIntegrator::propagateCovariance(const Eigen::Vector3d& a_mid,
                                        const Eigen::Vector3d& w_mid,
                                        double dt)
{
  const Eigen::Matrix3d R_mat = result_.delta_R.matrix();
  const Eigen::Matrix3d I3    = Eigen::Matrix3d::Identity();

  Eigen::Matrix<double,15,15> Fc = Eigen::Matrix<double,15,15>::Zero();
  Fc.block<3,3>(0, 3)  =  I3;
  Fc.block<3,3>(3, 6)  = -R_mat * skew(a_mid);
  Fc.block<3,3>(3, 9)  = -R_mat;
  Fc.block<3,3>(6, 6)  = -skew(w_mid);
  Fc.block<3,3>(6, 12) = -I3;

  Eigen::Matrix<double,15,15> F =
      Eigen::Matrix<double,15,15>::Identity() + Fc * dt;

  Eigen::Matrix<double,15,12> G = Eigen::Matrix<double,15,12>::Zero();
  G.block<3,3>(3,  0) = -R_mat;
  G.block<3,3>(6,  3) = -I3;
  G.block<3,3>(9,  6) =  I3;
  G.block<3,3>(12, 9) =  I3;

  Eigen::Matrix<double,12,12> Qc = Eigen::Matrix<double,12,12>::Zero();
  const double sa2  = noise_.acc_noise       * noise_.acc_noise;
  const double sw2  = noise_.gyro_noise      * noise_.gyro_noise;
  const double sba2 = noise_.acc_bias_noise  * noise_.acc_bias_noise;
  const double sbg2 = noise_.gyro_bias_noise * noise_.gyro_bias_noise;
  Qc.block<3,3>(0, 0) = sa2  * I3;
  Qc.block<3,3>(3, 3) = sw2  * I3;
  Qc.block<3,3>(6, 6) = sba2 * I3;
  Qc.block<3,3>(9, 9) = sbg2 * I3;

  result_.P = F * result_.P * F.transpose()
            + G * Qc * G.transpose() * dt;
  result_.P = 0.5 * (result_.P + result_.P.transpose());
}

}  // namespace stereo_vio
