#include "stereo_vio/eskf_estimator.hpp"
#include <cmath>
#include <iostream>

namespace stereo_vio {

EskfEstimator::EskfEstimator(const EskfParams& params)
  : params_(params)
{
  gravity_ = Eigen::Vector3d(0.0, 0.0, -params_.gravity_magnitude);
}

void EskfEstimator::feedImuForInit(const ImuData& imu)
{
  if (state_.initialised) return;
  init_imu_buffer_.push_back(imu);
}

bool EskfEstimator::tryInitialise()
{
  if (state_.initialised) return true;
  if (init_imu_buffer_.empty()) return false;
  const double t0 = init_imu_buffer_.front().timestamp;
  const double t1 = init_imu_buffer_.back().timestamp;
  if ((t1 - t0) < params_.init_duration) return false;

  Eigen::Vector3d a_mean = Eigen::Vector3d::Zero();
  Eigen::Vector3d g_mean = Eigen::Vector3d::Zero();
  for (const auto& imu : init_imu_buffer_) {
    a_mean += imu.accel;
    g_mean += imu.gyro;
  }
  const double N = static_cast<double>(init_imu_buffer_.size());
  a_mean /= N;
  g_mean /= N;

  // Find R_WB such that R_WB * a_mean = -g_world = [0,0,+9.81]
  // i.e. rotating body-frame gravity measurement to world frame gives +Z
  //
  // We want R_WB that maps a_mean_dir -> [0,0,+1]
  // axis = a_mean_dir x z_up,  angle = acos(a_mean_dir . z_up)
  const Eigen::Vector3d z_up(0.0, 0.0, 1.0);
  const Eigen::Vector3d a_dir = a_mean.normalized();

  const Eigen::Vector3d axis_wb = a_dir.cross(z_up);
  const double sin_a = axis_wb.norm();
  const double cos_a = a_dir.dot(z_up);

  Sophus::SO3d R_WB;
  if (sin_a < 1e-6) {
    R_WB = (cos_a > 0) ? Sophus::SO3d()
                       : Sophus::SO3d::exp(Eigen::Vector3d(M_PI, 0, 0));
  } else {
    R_WB = Sophus::SO3d::exp(axis_wb.normalized() * std::atan2(sin_a, cos_a));
  }

  state_.nominal.R  = R_WB;
  state_.nominal.ba = Eigen::Vector3d::Zero();
  state_.nominal.bg = g_mean;
  state_.nominal.p.setZero();
  state_.nominal.v.setZero();

  state_.P.setZero();
  state_.P.block<3,3>(0,  0) = 1e-4 * Eigen::Matrix3d::Identity();
  state_.P.block<3,3>(3,  3) = 1e-2 * Eigen::Matrix3d::Identity();
  state_.P.block<3,3>(6,  6) = 1e-1 * Eigen::Matrix3d::Identity();
  state_.P.block<3,3>(9,  9) = 1e-2 * Eigen::Matrix3d::Identity();
  state_.P.block<3,3>(12,12) = 1e-4 * Eigen::Matrix3d::Identity();

  state_.initialised = true;
  init_imu_buffer_.clear();

  // Verify: R_WB * a_mean should be close to [0,0,+9.81]
  const Eigen::Vector3d a_world = R_WB * a_mean;
  std::cout << "[ESKF] Initialised. a_norm=" << a_mean.norm()
            << " R_WB*a_mean=" << a_world.transpose()
            << " (expect ~[0,0,+9.81])\n"
            << "  bg=" << g_mean.transpose() << "\n"
            << "  R_WB=\n" << R_WB.matrix() << "\n";
  return true;
}
void EskfEstimator::propagate(const PreintegrationResult& preint)
{
  if (!state_.initialised || preint.dt <= 0.0) return;

  NominalState& nom = state_.nominal;
  const double  dt  = preint.dt;
  const Eigen::Matrix3d R_WB = nom.R.matrix();

  // Preintegrated deltas are in the preintegration local frame (body at start).
  // Convert them to world frame through current R_WB, then add gravity terms.
  nom.p = nom.p + nom.v * dt + R_WB * preint.delta_p + 0.5 * gravity_ * dt * dt;
  nom.v = nom.v + R_WB * preint.delta_v + gravity_ * dt;
  nom.R = nom.R * preint.delta_R;


  const Eigen::Vector3d a_corrected =
      preint.delta_v / std::max(dt, 1e-6);
  const Eigen::Vector3d w_corrected =
      preint.delta_R.log() / std::max(dt, 1e-6);

  const Eigen::Matrix<double,15,15> F =
      buildTransitionJacobian(a_corrected, w_corrected, dt);
  const Eigen::Matrix<double,15,12> G = buildNoiseJacobian();

  Eigen::Matrix<double,12,12> Qc = Eigen::Matrix<double,12,12>::Zero();
  const Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
  Qc.block<3,3>(0,0) = params_.acc_noise      * params_.acc_noise      * I3;
  Qc.block<3,3>(3,3) = params_.gyro_noise     * params_.gyro_noise     * I3;
  Qc.block<3,3>(6,6) = params_.acc_bias_noise * params_.acc_bias_noise * I3;
  Qc.block<3,3>(9,9) = params_.gyro_bias_noise* params_.gyro_bias_noise* I3;

  state_.P = F * state_.P * F.transpose() + G * Qc * G.transpose() * dt;
  state_.P = 0.5 * (state_.P + state_.P.transpose());
  state_.timestamp += dt;
}

void EskfEstimator::update(
    const std::unordered_map<uint64_t, TriangulationResult>& landmarks,
    const StereoCalibration& calib)
{
  if (!state_.initialised || landmarks.empty()) return;

  static int update_count = 0;

  const int max_lm = static_cast<int>(landmarks.size());
  Eigen::MatrixXd H_stack(2*max_lm, 15);
  Eigen::VectorXd z_stack(2*max_lm);
  H_stack.setZero();
  z_stack.setZero();

  const double R_scalar = params_.image_noise * params_.image_noise;
  int inlier_count = 0;

  for (const auto& [id, lm] : landmarks) {
    if (!lm.valid || !lm.has_prior) continue;

    // Predict where the anchored world point projects in current cam frame
    const Eigen::Matrix3d R_mat = state_.nominal.R.matrix();
    // const Eigen::Vector3d p_c =
    //     R_mat.transpose() * (lm.point_world - state_.nominal.p);

    // const Eigen::Vector3d p_body =
    // R_mat.transpose() * (lm.point_world - state_.nominal.p);
    // const Eigen::Vector3d p_c = params_.R_BC.transpose() * (p_body - params_.t_BC);

    // const double X=p_c.x(), Y=p_c.y(), Z=p_c.z();
    // if (Z < 0.1) continue;

    // world → body → cam
    const Eigen::Matrix3d R_WB = state_.nominal.R.matrix();
    // world → body
    const Eigen::Vector3d p_body =
        R_WB.transpose() * (lm.point_world - state_.nominal.p);
    // body → original cam0
    const Eigen::Vector3d p_cam0_orig = params_.R_CB * p_body + params_.t_CB;
    // original cam0 → rectified cam0
    const Eigen::Vector3d p_c = params_.R_rect0 * p_cam0_orig;

    const double X=p_c.x(), Y=p_c.y(), Z=p_c.z();
    if (Z < 0.1) continue;

    const double u_hat = calib.fx * X/Z + calib.cx;
    const double v_hat = calib.fy * Y/Z + calib.cy;
    const double u_obs = calib.fx * lm.point_cam.x()/lm.point_cam.z() + calib.cx;
    const double v_obs = calib.fy * lm.point_cam.y()/lm.point_cam.z() + calib.cy;

    Eigen::Vector2d z_i(u_obs - u_hat, v_obs - v_hat);  
    
    static int diag_count = 0;
    if (diag_count < 3) {
      std::cout << "[INNOV diag " << diag_count << "] "
                << "u_err=" << z_i(0) << " v_err=" << z_i(1)
                << " |z|=" << z_i.norm()
                << " p_c=(" << X << "," << Y << "," << Z << ")"
                << "\n  u_hat=" << u_hat << " v_hat=" << v_hat
                << "\n  u_obs=" << u_obs << " v_obs=" << v_obs
                << "\n  p_cam_lm=(" << lm.point_cam.transpose() << ")"
                << "\n  p_world=(" << lm.point_world.transpose() << ")"
                << "\n  p_body=(" << p_body.transpose() << ")"
                << "\n  nom_p=(" << state_.nominal.p.transpose() << ")"
                << "\n  nom_R=\n" << state_.nominal.R.matrix() << "\n";
      ++diag_count;
    }
    // Chi2 gate
    const Eigen::Matrix<double,2,15> H_i =
        buildMeasurementJacobian(lm.point_world, calib);
    const Eigen::Matrix2d S = H_i * state_.P * H_i.transpose()
                            + R_scalar * Eigen::Matrix2d::Identity();
    if (z_i.transpose() * S.inverse() * z_i > params_.chi2_threshold) continue;

    H_stack.block<2,15>(2*inlier_count, 0) = H_i;
    z_stack.segment<2>(2*inlier_count)     = z_i;
    ++inlier_count;
  }

  if (inlier_count < 8) return;

  const Eigen::MatrixXd H = H_stack.topRows(2*inlier_count);
  const Eigen::VectorXd z = z_stack.head(2*inlier_count);
  const Eigen::MatrixXd R_noise =
      R_scalar * Eigen::MatrixXd::Identity(2*inlier_count, 2*inlier_count);

  const Eigen::MatrixXd S_full = H * state_.P * H.transpose() + R_noise;
    const Eigen::LLT<Eigen::MatrixXd> llt(S_full);
    if (llt.info() != Eigen::Success) return;

    const Eigen::MatrixXd K      = state_.P * H.transpose() *
      llt.solve(Eigen::MatrixXd::Identity(2*inlier_count, 2*inlier_count));
  const Eigen::Matrix<double,15,1> dx = (K * z).head<15>();

    if (!dx.allFinite()) return;
    if (dx.segment<3>(0).norm() > 0.5 || dx.segment<3>(6).norm() > 0.35) return;

  injectErrorState(dx);

  const Eigen::Matrix<double,15,15> IKH =
      Eigen::Matrix<double,15,15>::Identity() - K * H;
  resetErrorState(IKH);

  if (update_count < 3) {
    std::cout << "[UPDATE " << update_count << "] "
              << inlier_count << "/" << landmarks.size()
              << " inliers, p=" << state_.nominal.p.transpose() << "\n";
    ++update_count;
  }
}

Eigen::Matrix<double,15,15> EskfEstimator::buildTransitionJacobian(
    const Eigen::Vector3d& a_corrected,
    const Eigen::Vector3d& w_corrected,
    double dt) const
{
  const Eigen::Matrix3d R_mat = state_.nominal.R.matrix();
  const Eigen::Matrix3d I3    = Eigen::Matrix3d::Identity();
  Eigen::Matrix<double,15,15> Fc = Eigen::Matrix<double,15,15>::Zero();
  Fc.block<3,3>(0, 3)  =  I3;
  Fc.block<3,3>(3, 6)  = -R_mat * skew(a_corrected);
  Fc.block<3,3>(3, 9)  = -R_mat;
  Fc.block<3,3>(6, 6)  = -skew(w_corrected);
  Fc.block<3,3>(6,12)  = -I3;
  return Eigen::Matrix<double,15,15>::Identity() + Fc * dt;
}

Eigen::Matrix<double,15,12> EskfEstimator::buildNoiseJacobian() const
{
  const Eigen::Matrix3d R_mat = state_.nominal.R.matrix();
  const Eigen::Matrix3d I3    = Eigen::Matrix3d::Identity();
  Eigen::Matrix<double,15,12> G = Eigen::Matrix<double,15,12>::Zero();
  G.block<3,3>(3,  0) = -R_mat;
  G.block<3,3>(6,  3) = -I3;
  G.block<3,3>(9,  6) =  I3;
  G.block<3,3>(12, 9) =  I3;
  return G;
}

Eigen::Matrix<double,2,15> EskfEstimator::buildMeasurementJacobian(
    const Eigen::Vector3d& point_world,
    const StereoCalibration& calib) const
{
  const Eigen::Matrix3d R_WB  = state_.nominal.R.matrix();
  const Eigen::Vector3d p_body =
      R_WB.transpose() * (point_world - state_.nominal.p);
  const Eigen::Vector3d p_c =
      params_.R_rect0 * (params_.R_CB * p_body + params_.t_CB);

  const double X=p_c.x(), Y=p_c.y(), Z=p_c.z(), Z2=Z*Z;

  Eigen::Matrix<double,2,3> J_proj;
  J_proj << calib.fx/Z,          0, -calib.fx*X/Z2,
                    0, calib.fy/Z, -calib.fy*Y/Z2;

  // Full chain: J = J_proj * R_rect0 * R_CB * (-R_WB^T)  for position
  //             J = J_proj * R_rect0 * R_CB * skew(p_body)  for rotation
  Eigen::Matrix<double,3,15> J_state = Eigen::Matrix<double,3,15>::Zero();
  const Eigen::Matrix3d R_full = params_.R_rect0 * params_.R_CB;
  J_state.block<3,3>(0,0) = -R_full * R_WB.transpose();
  J_state.block<3,3>(0,6) =  R_full * skew(p_body);

  return J_proj * J_state;
}

void EskfEstimator::injectErrorState(const Eigen::Matrix<double,15,1>& dx)
{
  state_.nominal.p  += dx.segment<3>(0);
  state_.nominal.v  += dx.segment<3>(3);
  state_.nominal.R   = state_.nominal.R * Sophus::SO3d::exp(dx.segment<3>(6));
  state_.nominal.ba += dx.segment<3>(9);
  state_.nominal.bg += dx.segment<3>(12);
}

void EskfEstimator::resetErrorState(const Eigen::Matrix<double,15,15>& IKH)
{
  state_.P = IKH * state_.P * IKH.transpose();
  state_.P = 0.5 * (state_.P + state_.P.transpose());
}

void EskfEstimator::updatePose(const Eigen::Vector3d& p_meas,
                                const Sophus::SO3d& R_meas,
                                double pos_noise,
                                double rot_noise)
{
  if (!state_.initialised) return;
  if (!p_meas.allFinite()) return;

  // 6-DOF pose measurement: [p_meas, R_meas]
  // Innovation: z = [p_meas - p, log(R_meas * R^{-1})]
  const Eigen::Vector3d z_p = p_meas - state_.nominal.p;
  const Eigen::Vector3d z_r =
      (R_meas * state_.nominal.R.inverse()).log();

  Eigen::Matrix<double,6,1> z;
  z.head<3>() = z_p;
  z.tail<3>() = z_r;

  // Measurement Jacobian H (6x15)
  Eigen::Matrix<double,6,15> H = Eigen::Matrix<double,6,15>::Zero();
  H.block<3,3>(0, 0) = Eigen::Matrix3d::Identity();   // dp_meas/dp
  H.block<3,3>(3, 6) = Eigen::Matrix3d::Identity();   // dR_meas/dtheta

  // Noise
  Eigen::Matrix<double,6,6> R_noise = Eigen::Matrix<double,6,6>::Zero();
  R_noise.block<3,3>(0,0) = pos_noise * pos_noise * Eigen::Matrix3d::Identity();
  R_noise.block<3,3>(3,3) = rot_noise * rot_noise * Eigen::Matrix3d::Identity();

  const Eigen::Matrix<double,6,6> S = H * state_.P * H.transpose() + R_noise;
  Eigen::LLT<Eigen::Matrix<double,6,6>> llt(S);
  if (llt.info() != Eigen::Success) {
    return;
  }

  // Use LLT decomposition instead of direct inverse
  const Eigen::Matrix<double,15,6> K =
      state_.P * H.transpose() * llt.solve(Eigen::Matrix<double,6,6>::Identity());
  const Eigen::Matrix<double,15,1> dx = K * z;

  if (!dx.allFinite()) {
    return;
  }

  // Reject numerically unstable/physically implausible correction jumps.
  if (dx.segment<3>(0).norm() > 1.0 || dx.segment<3>(6).norm() > 0.7) {
    return;
  }

  injectErrorState(dx);

  const Eigen::Matrix<double,15,15> IKH =
      Eigen::Matrix<double,15,15>::Identity() - K * H;
  resetErrorState(IKH);
}

}  // namespace stereo_vio
