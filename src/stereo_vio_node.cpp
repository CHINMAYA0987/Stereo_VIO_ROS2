// stereo_vio_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include "stereo_vio/imu_integrator.hpp"
#include "stereo_vio/feature_tracker.hpp"
#include "stereo_vio/stereo_triangulator.hpp"
#include "stereo_vio/eskf_estimator.hpp"
#include <Eigen/Dense>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <sophus/so3.hpp>
#include <deque>
#include <mutex>
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>
#include <opencv2/core/eigen.hpp>
#include <execinfo.h>
#include <signal.h>

namespace stereo_vio {

using ImageMsg    = sensor_msgs::msg::Image;
using ImuMsg      = sensor_msgs::msg::Imu;
using OdomMsg     = nav_msgs::msg::Odometry;
using PathMsg     = nav_msgs::msg::Path;
using PoseStMsg   = geometry_msgs::msg::PoseStamped;
using SyncPolicy  = message_filters::sync_policies::ApproximateTime<ImageMsg,ImageMsg>;
using Synchronizer= message_filters::Synchronizer<SyncPolicy>;

class StereoVioNode : public rclcpp::Node {
public:
  explicit StereoVioNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
    : Node("stereo_vio_node", options)
  {
    declareParameters();
    loadParameters();
    initComponents();
    setupPubSub();
    RCLCPP_INFO(get_logger(),
        "[StereoVIO] Node started. Waiting for static init (%.1f s)...",
        eskf_params_.init_duration);
  }

private:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  // std::unordered_map<uint64_t, Eigen::Vector3d> landmark_world_map_;
  struct LandmarkEntry {
    Eigen::Vector3d p_world;
    int observation_count = 0;
  };
  std::unordered_map<uint64_t, LandmarkEntry> landmark_world_map_;
  Eigen::Matrix3d R_CB_;  // rotation: cam0 frame → body(IMU) frame
  Eigen::Vector3d t_CB_;  // translation: cam0 origin in body frame
  Eigen::Matrix3d R_rect0_ = Eigen::Matrix3d::Identity();
  void declareParameters() {
    declare_parameter<std::string>("imu_topic",          "/imu0");
    declare_parameter<std::string>("left_image_topic",   "/cam0/image_raw");
    declare_parameter<std::string>("right_image_topic",  "/cam1/image_raw");
    declare_parameter<std::string>("odom_topic",         "/stereo_vio/odom");
    declare_parameter<std::string>("path_topic",         "/stereo_vio/path");
    declare_parameter<std::vector<double>>("cam0_intrinsics",
        {458.654, 457.296, 367.215, 248.375});
    declare_parameter<std::vector<double>>("cam0_distortion_coeffs",
        {-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05});
    declare_parameter<std::vector<double>>("cam1_intrinsics",
        {457.587, 456.134, 379.999, 255.238});
    declare_parameter<std::vector<double>>("cam1_distortion_coeffs",
        {-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05});
    declare_parameter<std::vector<double>>("T_cam1_cam0_rotation",
        std::vector<double>(9, 0.0));
    declare_parameter<std::vector<double>>("T_cam1_cam0_translation",
        {-0.11007381, 0.00015880, 0.00017887});
    declare_parameter<std::vector<double>>("T_cam0_imu_rotation",
        std::vector<double>(9, 0.0));
    declare_parameter<std::vector<double>>("T_cam0_imu_translation",
        std::vector<double>(3, 0.0));
    declare_parameter<double>("gyro_noise_density",   0.005);
    declare_parameter<double>("gyro_random_walk",     4.0e-6);
    declare_parameter<double>("accel_noise_density",  0.01);
    declare_parameter<double>("accel_random_walk",    2.0e-4);
    declare_parameter<double>("init_cov_rot",  1.0e-4);
    declare_parameter<double>("init_cov_vel",  1.0e-2);
    declare_parameter<double>("init_cov_pos",  1.0e-4);
    declare_parameter<double>("init_cov_bg",   1.0e-6);
    declare_parameter<double>("init_cov_ba",   1.0e-4);
    declare_parameter<int>("max_features",        150);
    declare_parameter<int>("min_feature_dist",     20);
    declare_parameter<int>("fast_threshold",       20);
    declare_parameter<int>("klt_win_size",         21);
    declare_parameter<int>("klt_max_level",         3);
    declare_parameter<double>("klt_epsilon",       0.01);
    declare_parameter<double>("ransac_threshold",  1.0);
    declare_parameter<int>("min_track_length",      3);
    declare_parameter<double>("reproj_error_thresh", 2.0);
    declare_parameter<double>("min_depth",  0.2);
    declare_parameter<double>("max_depth",  20.0);
    declare_parameter<double>("gravity",    9.81);
    declare_parameter<double>("eskf.init_duration", 2.0);
    declare_parameter<bool>("publish_tf",           true);
    declare_parameter<std::string>("world_frame_id", "world");
    declare_parameter<std::string>("body_frame_id",  "body");
    declare_parameter<std::string>("camera_frame_id","cam0");

    declare_parameter<double>("fb_error_threshold",    2.0);
    declare_parameter<double>("stereo_row_threshold",  1.5);
    declare_parameter<double>("image_noise", 1.5);

    declare_parameter<double>("chi2_threshold", 5.991);
    declare_parameter<bool>("use_direct_pnp_fusion", true);
    declare_parameter<double>("direct_pnp_alpha", 0.2);
    declare_parameter<double>("max_pnp_step_m", 1.0);
    declare_parameter<double>("max_pnp_step_rad", 0.6);
    declare_parameter<double>("pnp_assoc_reproj_thresh", 12.0);
    declare_parameter<double>("pnp_reproj_accept_thresh", 3.0);
  }

  void loadParameters() {
    left_topic_   = get_parameter("left_image_topic").as_string();
    right_topic_  = get_parameter("right_image_topic").as_string();
    odom_topic_   = get_parameter("odom_topic").as_string();
    path_topic_   = get_parameter("path_topic").as_string();
    world_frame_  = get_parameter("world_frame_id").as_string();
    body_frame_   = get_parameter("body_frame_id").as_string();
    camera_frame_ = get_parameter("camera_frame_id").as_string();
    publish_tf_   = get_parameter("publish_tf").as_bool();
    use_direct_pnp_fusion_ = get_parameter("use_direct_pnp_fusion").as_bool();
    direct_pnp_alpha_ = get_parameter("direct_pnp_alpha").as_double();
    max_pnp_step_m_ = get_parameter("max_pnp_step_m").as_double();
    max_pnp_step_rad_ = get_parameter("max_pnp_step_rad").as_double();
    pnp_assoc_reproj_thresh_ = get_parameter("pnp_assoc_reproj_thresh").as_double();
    pnp_reproj_accept_thresh_ = get_parameter("pnp_reproj_accept_thresh").as_double();
    if (direct_pnp_alpha_ < 0.0) direct_pnp_alpha_ = 0.0;
    if (direct_pnp_alpha_ > 1.0) direct_pnp_alpha_ = 1.0;

    // --- Load raw intrinsics ---
    const auto c0 = get_parameter("cam0_intrinsics").as_double_array();
    const auto c1 = get_parameter("cam1_intrinsics").as_double_array();
    const auto d0 = get_parameter("cam0_distortion_coeffs").as_double_array();
    const auto d1 = get_parameter("cam1_distortion_coeffs").as_double_array();

    eskf_params_.image_noise = get_parameter("image_noise").as_double();
    eskf_params_.R_rect0 = R_rect0_;
    eskf_params_.chi2_threshold = get_parameter("chi2_threshold").as_double();

    const auto Rc = get_parameter("T_cam0_imu_rotation").as_double_array();
    const auto tc = get_parameter("T_cam0_imu_translation").as_double_array();
    // T_cam0_imu: transforms points from IMU frame to cam0 frame
    // We need R_BC = R_cam0_imu, t_BC = t_cam0_imu
    R_CB_ << Rc[0],Rc[1],Rc[2],
            Rc[3],Rc[4],Rc[5],
            Rc[6],Rc[7],Rc[8];
    t_CB_ << tc[0], tc[1], tc[2];

    eskf_params_.R_CB = R_CB_;
    eskf_params_.t_CB = t_CB_;

    double K0_data[] = { c0[0], 0.0,   c0[2],
                        0.0,   c0[1], c0[3],
                        0.0,   0.0,   1.0 };
    cv::Mat K0 = cv::Mat(3, 3, CV_64F, K0_data).clone();

    double K1_data[] = { c1[0], 0.0,   c1[2],
                        0.0,   c1[1], c1[3],
                        0.0,   0.0,   1.0 };
    cv::Mat K1 = cv::Mat(3, 3, CV_64F, K1_data).clone();

    std::vector<double> d0_vec(d0.begin(), d0.end());
    std::vector<double> d1_vec(d1.begin(), d1.end());
    cv::Mat dist0(d0_vec, true);
    cv::Mat dist1(d1_vec, true);
    

    const auto R_vec = get_parameter("T_cam1_cam0_rotation").as_double_array();
    const auto t_vec = get_parameter("T_cam1_cam0_translation").as_double_array();

    double R_data[] = { R_vec[0], R_vec[1], R_vec[2],
                        R_vec[3], R_vec[4], R_vec[5],
                        R_vec[6], R_vec[7], R_vec[8] };
    cv::Mat R_c1_c0 = cv::Mat(3, 3, CV_64F, R_data).clone();

    double t_data[] = { t_vec[0], t_vec[1], t_vec[2] };
    cv::Mat t_c1_c0 = cv::Mat(3, 1, CV_64F, t_data).clone();

    // --- Stereo rectification ---
    // EuRoC images are 752x480
    const cv::Size img_size(752, 480);
    cv::Mat R0, R1, P0_cv, P1_cv, Q;
    cv::stereoRectify(K0, dist0, K1, dist1, img_size,
                      R_c1_c0, t_c1_c0,
                      R0, R1, P0_cv, P1_cv, Q,
                      cv::CALIB_ZERO_DISPARITY, 0);  // alpha=0: no black borders

    cv::initUndistortRectifyMap(K0, dist0, R0, P0_cv, img_size,
                                CV_32FC1, rect_map0_x_, rect_map0_y_);
    cv::initUndistortRectifyMap(K1, dist1, R1, P1_cv, img_size,
                                CV_32FC1, rect_map1_x_, rect_map1_y_);

    RCLCPP_INFO(get_logger(),
    "rect_map0 size: %dx%d type:%d  rect_map1 size: %dx%d type:%d",
    rect_map0_x_.cols, rect_map0_x_.rows, rect_map0_x_.type(),
    rect_map1_x_.cols, rect_map1_x_.rows, rect_map1_x_.type());

    // --- Fill StereoCalibration from rectified projection matrices ---
    // Fill rectification rotation (3x3 only)
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 3; ++c)
        R_rect0_(r,c) = R0.at<double>(r,c);

    // Fill projection matrices (3x4)
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c) {
        stereo_calib_.P0(r,c) = P0_cv.at<double>(r,c);
        stereo_calib_.P1(r,c) = P1_cv.at<double>(r,c);
      }
    stereo_calib_.computeFromProjectionMatrices();

    // After stereoRectify call:
    if (R0.empty()) {
      RCLCPP_ERROR(get_logger(), "R0 is empty after stereoRectify!");
      throw std::runtime_error("stereoRectify failed");
    }
    // cv::Mat R0_64f;
    // R0.convertTo(R0_64f, CV_64F);  // ensure correct type
    // for (int r = 0; r < 3; ++r)
    //   for (int c = 0; c < 3; ++c)
    //     R_rect0_(r,c) = R0_64f.at<double>(r,c);
    // eskf_params_.R_rect0 = R_rect0_;
    RCLCPP_INFO(get_logger(), "R_rect0 loaded: [%.4f %.4f %.4f]",
        R_rect0_(0,0), R_rect0_(0,1), R_rect0_(0,2));

    RCLCPP_INFO(get_logger(),
        "[StereoVIO] Rectified baseline=%.4f m, fx=%.2f",
        stereo_calib_.baseline,
        stereo_calib_.P0(0,0));

    // --- IMU / ESKF / tracker params (unchanged) ---
    imu_noise_.gyro_noise      = get_parameter("gyro_noise_density").as_double();
    imu_noise_.acc_noise       = get_parameter("accel_noise_density").as_double();
    imu_noise_.gyro_bias_noise = get_parameter("gyro_random_walk").as_double();
    imu_noise_.acc_bias_noise  = get_parameter("accel_random_walk").as_double();

    eskf_params_.acc_noise         = imu_noise_.acc_noise;
    eskf_params_.gyro_noise        = imu_noise_.gyro_noise;
    eskf_params_.acc_bias_noise    = imu_noise_.acc_bias_noise;
    eskf_params_.gyro_bias_noise   = imu_noise_.gyro_bias_noise;
    eskf_params_.gravity_magnitude = get_parameter("gravity").as_double();
    eskf_params_.init_duration     = get_parameter("eskf.init_duration").as_double();

    init_cov_rot_ = get_parameter("init_cov_rot").as_double();
    init_cov_vel_ = get_parameter("init_cov_vel").as_double();
    init_cov_pos_ = get_parameter("init_cov_pos").as_double();
    init_cov_bg_  = get_parameter("init_cov_bg").as_double();
    init_cov_ba_  = get_parameter("init_cov_ba").as_double();

    tracker_params_.max_features         = get_parameter("max_features").as_int();
    tracker_params_.min_feature_distance = get_parameter("min_feature_dist").as_int();
    tracker_params_.fast_threshold       = get_parameter("fast_threshold").as_int();
    tracker_params_.klt_window_size      = get_parameter("klt_win_size").as_int();
    tracker_params_.klt_max_level        = get_parameter("klt_max_level").as_int();
    tracker_params_.fb_error_threshold   = get_parameter("fb_error_threshold").as_double();
    tracker_params_.stereo_row_threshold = get_parameter("stereo_row_threshold").as_double();

    tri_params_.reprojection_threshold = get_parameter("reproj_error_thresh").as_double();
    tri_params_.min_depth              = get_parameter("min_depth").as_double();
    tri_params_.max_depth              = get_parameter("max_depth").as_double();
}

  void initComponents() {
    integrator_   = std::make_unique<ImuIntegrator>(imu_noise_);
    tracker_      = std::make_unique<FeatureTracker>(tracker_params_);
    triangulator_ = std::make_unique<StereoTriangulator>(stereo_calib_, tri_params_);
    estimator_    = std::make_unique<EskfEstimator>(eskf_params_);
  }

  void setupPubSub() {
    odom_pub_      = create_publisher<OdomMsg>(odom_topic_, 10);
    path_pub_      = create_publisher<PathMsg>(path_topic_, 10);
    debug_img_pub_ = create_publisher<ImageMsg>("/vio/debug_image", 10);

    if (publish_tf_)
      tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    left_sub_.subscribe( this, left_topic_);
    right_sub_.subscribe(this, right_topic_);
    sync_ = std::make_shared<Synchronizer>(SyncPolicy(10), left_sub_, right_sub_);
    sync_->registerCallback(
        std::bind(&StereoVioNode::stereoCallback, this,
                  std::placeholders::_1, std::placeholders::_2));

    const std::string imu_topic = get_parameter("imu_topic").as_string();
    imu_sub_ = create_subscription<ImuMsg>(imu_topic, 500,
        std::bind(&StereoVioNode::imuCallback, this, std::placeholders::_1));
  }

  void applyInitCovariance() {
    EskfState& state = estimator_->getState();
    state.P.setZero();
    state.P.block<3,3>(0,  0)  = init_cov_pos_ * Eigen::Matrix3d::Identity();
    state.P.block<3,3>(3,  3)  = init_cov_vel_ * Eigen::Matrix3d::Identity();
    state.P.block<3,3>(6,  6)  = init_cov_rot_ * Eigen::Matrix3d::Identity();
    state.P.block<3,3>(9,  9)  = init_cov_ba_  * Eigen::Matrix3d::Identity();
    state.P.block<3,3>(12,12)  = init_cov_bg_  * Eigen::Matrix3d::Identity();
  }

  void imuCallback(const ImuMsg::SharedPtr msg) {
    ImuData data;
    data.timestamp = rclcpp::Time(msg->header.stamp).seconds();
    data.accel = Eigen::Vector3d(msg->linear_acceleration.x,
                                 msg->linear_acceleration.y,
                                 msg->linear_acceleration.z);
    data.gyro  = Eigen::Vector3d(msg->angular_velocity.x,
                                 msg->angular_velocity.y,
                                 msg->angular_velocity.z);
    if (!estimator_->isInitialised()) {
      // During init: feed to init buffer only, do NOT add to integration buffer
      estimator_->feedImuForInit(data);
      if (estimator_->tryInitialise()) {
        applyInitCovariance();
        // Clear integration buffer — discard all IMU data from init window
        std::lock_guard<std::mutex> lock(imu_mutex_);
        imu_buffer_.clear();
        last_img_timestamp_ = data.timestamp;  // reset timestamp anchor
        RCLCPP_INFO(get_logger(), "[StereoVIO] Static initialisation complete.");
      }
    } else {
      // After init: buffer for integration
      std::lock_guard<std::mutex> lock(imu_mutex_);
      imu_buffer_.push_back(data);
      if (imu_buffer_.size() > 2000) imu_buffer_.pop_front();
    }
  }

  void stereoCallback(const ImageMsg::ConstSharedPtr& msg_left,
                      const ImageMsg::ConstSharedPtr& msg_right)
  {
    if (!estimator_->isInitialised()) {
      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
          "[StereoVIO] Waiting for initialisation...");
      return;
    }
    const double t_img = rclcpp::Time(msg_left->header.stamp).seconds();

    cv::Mat img_left, img_right;
    try {
      img_left  = cv_bridge::toCvShare(msg_left,  "mono8")->image;
      img_right = cv_bridge::toCvShare(msg_right, "mono8")->image;
    } catch (const cv_bridge::Exception& e) {
      RCLCPP_ERROR(get_logger(), "[StereoVIO] cv_bridge: %s", e.what());
      return;
    }

    cv::Mat img_left_rect, img_right_rect;
    cv::remap(img_left,  img_left_rect,  rect_map0_x_, rect_map0_y_, cv::INTER_LINEAR);
    cv::remap(img_right, img_right_rect, rect_map1_x_, rect_map1_y_, cv::INTER_LINEAR);


    std::vector<ImuData> imu_chunk;
    {
      std::lock_guard<std::mutex> lock(imu_mutex_);
      for (const auto& d : imu_buffer_)
        if (d.timestamp > last_img_timestamp_ && d.timestamp <= t_img)
          imu_chunk.push_back(d);
      while (!imu_buffer_.empty() &&
             imu_buffer_.front().timestamp <= t_img)
        imu_buffer_.pop_front();
    }

    {
      const EskfState& s = estimator_->getState();
      integrator_->reset(s.nominal.ba, s.nominal.bg);
    }
    for (const auto& d : imu_chunk) integrator_->pushMeasurement(d);
    const PreintegrationResult& preint = integrator_->getResult();
    if (preint.dt > 0.0 && !use_direct_pnp_fusion_) estimator_->propagate(preint);

    // In stereoCallback, after extracting imu_chunk:
    RCLCPP_INFO_ONCE(get_logger(),
        "First imu_chunk: size=%zu, t_img=%.4f, last_img_ts=%.4f, dt_span=%.4f",
        imu_chunk.size(),
        t_img,
        last_img_timestamp_,
        imu_chunk.empty() ? 0.0 : 
            imu_chunk.back().timestamp - imu_chunk.front().timestamp);

    // After remap, before tracker->track:
    RCLCPP_INFO_ONCE(get_logger(), 
        "rect img size: %dx%d, raw img size: %dx%d",
        img_left_rect.cols, img_left_rect.rows,
        img_left.cols, img_left.rows);
    // Verify a specific pixel value changed (rectification actually ran)
    RCLCPP_INFO_ONCE(get_logger(),
        "raw[240,376]=%d  rect[240,376]=%d",
        img_left.at<uint8_t>(240,376),
        img_left_rect.at<uint8_t>(240,376));
      
    static Sophus::SO3d prev_R;
    static bool first = true;
    if (first) { prev_R = estimator_->getState().nominal.R; first = false; }
    else {
      Sophus::SO3d dR = prev_R.inverse() * estimator_->getState().nominal.R;
      RCLCPP_INFO_ONCE(get_logger(), "dR between frames: %.6f rad (%.3f deg)",
          dR.log().norm(), dR.log().norm() * 180.0 / M_PI);
      prev_R = estimator_->getState().nominal.R;
    }
    RCLCPP_INFO_ONCE(get_logger(),
    "preint: dt=%.4f delta_R_angle=%.6f rad delta_v=(%f,%f,%f) delta_p=(%f,%f,%f)",
    preint.dt,
    preint.delta_R.log().norm(),
    preint.delta_v.x(), preint.delta_v.y(), preint.delta_v.z(),
    preint.delta_p.x(), preint.delta_p.y(), preint.delta_p.z());

    // const FeatureFrame frame = tracker_->track(t_img, img_left, img_right);
    // using rectified image instead
    const FeatureFrame frame = tracker_->track(t_img, img_left_rect, img_right_rect);
    const auto landmarks     = triangulator_->triangulate(frame);

    if (landmark_world_map_.size() > 500) {
      std::unordered_set<uint64_t> current_ids;
      for (const auto& f : frame.features) current_ids.insert(f.id);
      for (auto it = landmark_world_map_.begin(); it != landmark_world_map_.end(); ) {
        it = (current_ids.count(it->first) == 0)
          ? landmark_world_map_.erase(it) : std::next(it);
      }
    }


    std::unordered_map<uint64_t, TriangulationResult> update_landmarks;

    const NominalState& nom = estimator_->getState().nominal;
    // In stereoCallback, after triangulation:
    // Collect 3D-2D correspondences for PnP
    std::vector<cv::Point3f> pts3d;
    std::vector<cv::Point2f> pts2d;

    for (const auto& [id, lm] : landmarks) {
      if (!lm.valid) continue;
      auto it = landmark_world_map_.find(id);
      if (it != landmark_world_map_.end() && it->second.observation_count >= 3) {
        const Eigen::Vector3d& pw = it->second.p_world;

        // Re-association gate: only keep/use landmarks geometrically consistent
        // with current pose estimate and current observation.
        const Eigen::Vector3d p_body_pred = nom.R.inverse() * (pw - nom.p);
        const Eigen::Vector3d p_cam_pred = R_rect0_ * (R_CB_ * p_body_pred + t_CB_);
        if (!p_cam_pred.allFinite() || p_cam_pred.z() <= 0.1) {
          landmark_world_map_.erase(id);
          continue;
        }

        const double u_hat = stereo_calib_.fx * p_cam_pred.x() / p_cam_pred.z() + stereo_calib_.cx;
        const double v_hat = stereo_calib_.fy * p_cam_pred.y() / p_cam_pred.z() + stereo_calib_.cy;
        const double u_obs = stereo_calib_.fx * lm.point_cam.x()/lm.point_cam.z() + stereo_calib_.cx;
        const double v_obs = stereo_calib_.fy * lm.point_cam.y()/lm.point_cam.z() + stereo_calib_.cy;
        const double reproj_assoc_err = std::hypot(u_obs - u_hat, v_obs - v_hat);

        if (!std::isfinite(reproj_assoc_err) || reproj_assoc_err > pnp_assoc_reproj_thresh_) {
          landmark_world_map_.erase(id);
          continue;
        }

        // 3D point in world frame
        pts3d.push_back(cv::Point3f(pw.x(), pw.y(), pw.z()));
        // 2D observation in rectified cam frame
        pts2d.push_back(cv::Point2f(u_obs, v_obs));

        TriangulationResult lm_update = lm;
        lm_update.point_world = pw;
        lm_update.has_prior = true;
        update_landmarks[id] = lm_update;

        it->second.observation_count++;
      } else if (it == landmark_world_map_.end()) {
        // anchor new landmark
        const Eigen::Vector3d p_cam0_orig = R_rect0_.transpose() * lm.point_cam;
        const Eigen::Vector3d p_body = R_CB_.transpose() * (p_cam0_orig - t_CB_);
        LandmarkEntry entry;
        entry.p_world = nom.R * p_body + nom.p;
        entry.observation_count = 1;
        landmark_world_map_[id] = entry;
      } else {
        it->second.observation_count++;
      }
    }
    bool pts_valid = true;
    for (const auto& p : pts3d) {
      if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
        pts_valid = false; break;
      }
    }
    if (!pts_valid) {
      RCLCPP_WARN_ONCE(get_logger(), "PnP: invalid 3D points, skipping");
    }

    // Use PnP to get camera pose estimate
    else if (pts3d.size() >= 6) {
      // Build camera matrix for rectified cam0
      double K_rect_data[] = { stereo_calib_.fx, 0.0, stereo_calib_.cx,
                         0.0, stereo_calib_.fy, stereo_calib_.cy,
                         0.0, 0.0, 1.0 };
      cv::Mat K_rect = cv::Mat(3, 3, CV_64F, K_rect_data).clone();
      cv::Mat dist_zero = cv::Mat::zeros(4, 1, CV_64F);  // already rectified
      cv::Mat rvec, tvec;
      std::vector<int> inliers;
      bool ok = cv::solvePnPRansac(pts3d, pts2d, K_rect, dist_zero,
                                    rvec, tvec, false, 100, 4.0, 0.99, inliers);
      const double inlier_ratio = pts3d.empty() ? 0.0 :
          static_cast<double>(inliers.size()) / static_cast<double>(pts3d.size());

      double mean_reproj_err = 1e9;
      if (ok && !inliers.empty()) {
        std::vector<cv::Point3f> inlier_pts3d;
        std::vector<cv::Point2f> inlier_pts2d;
        inlier_pts3d.reserve(inliers.size());
        inlier_pts2d.reserve(inliers.size());
        for (int idx : inliers) {
          if (idx < 0 || static_cast<size_t>(idx) >= pts3d.size()) continue;
          inlier_pts3d.push_back(pts3d[idx]);
          inlier_pts2d.push_back(pts2d[idx]);
        }

        std::vector<cv::Point2f> reproj_pts;
        cv::projectPoints(inlier_pts3d, rvec, tvec, K_rect, dist_zero, reproj_pts);
        if (reproj_pts.size() == inlier_pts2d.size() && !reproj_pts.empty()) {
          double sum_err = 0.0;
          for (size_t i = 0; i < reproj_pts.size(); ++i)
            sum_err += cv::norm(reproj_pts[i] - inlier_pts2d[i]);
          mean_reproj_err = sum_err / static_cast<double>(reproj_pts.size());
        }
      }

      if (ok && inliers.size() >= 5 && inlier_ratio >= 0.10 && mean_reproj_err < pnp_reproj_accept_thresh_) {
        // Convert PnP result to world pose
        cv::Mat R_pnp;
        cv::Rodrigues(rvec, R_pnp);
        // R_pnp, tvec give: p_cam = R_pnp * p_world + tvec
        // We need world pose of body: 
        // p_cam = R_rect0 * R_CB * R_WB^T * (p_world - p) + R_rect0 * t_CB
        // So R_pnp = R_rect0 * R_CB * R_WB^T
        // R_WB = (R_rect0 * R_CB)^T * R_pnp^T ... no, R_WB^T = R_CB^T * R_rect0^T * R_pnp
        Eigen::Matrix3d R_pnp_e;
        cv::cv2eigen(R_pnp, R_pnp_e);
        Eigen::Vector3d t_pnp_e;
        cv::cv2eigen(tvec, t_pnp_e);

        const Eigen::Matrix3d R_full = R_rect0_ * R_CB_;
        // Use p_meas and R_WB_meas as direct pose measurements in ESKF
        const double pos_noise = 0.05;
        const double rot_noise = 0.02;

        // Validate PnP result before using it
        const Eigen::Matrix3d R_WB_meas_raw = R_pnp_e.transpose() * R_full;

        // Project onto SO(3) via SVD to ensure valid rotation matrix
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(
            R_WB_meas_raw, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d R_WB_meas = svd.matrixU() * svd.matrixV().transpose();

        // Ensure det = +1 (not -1)
        if (R_WB_meas.determinant() < 0)
          R_WB_meas = svd.matrixU() *
                      Eigen::DiagonalMatrix<double,3>(1, 1, -1) *
                      svd.matrixV().transpose();

        // t_pnp = R_full * R_WB^T * (-p) + R_rect0 * t_CB
        // → -p = R_WB * R_full^T * (t_pnp - R_rect0 * t_CB)
        const Eigen::Vector3d p_meas = -R_WB_meas * R_full.transpose() *
                                        (t_pnp_e - R_rect0_ * t_CB_);

        if (!R_WB_meas.allFinite() || !p_meas.allFinite()) {
          RCLCPP_WARN(get_logger(), "PnP result invalid, skipping");
        } else {
          bool temporal_ok = true;
          if (has_last_pnp_pose_) {
            const double dp_step = (p_meas - last_pnp_p_).norm();
            const double dtheta_step =
                (Sophus::SO3d(R_WB_meas) * last_pnp_R_.inverse()).log().norm();
            if (dp_step > max_pnp_step_m_ || dtheta_step > max_pnp_step_rad_) {
              RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                  "PnP temporal gate: dp_step=%.3f dtheta_step=%.3f", dp_step, dtheta_step);
              temporal_ok = false;
            }
          }

          if (temporal_ok) {
            const Eigen::Vector3d dp = p_meas - nom.p;
            const double dp_norm = dp.norm();
            const double dtheta = (Sophus::SO3d(R_WB_meas) * nom.R.inverse()).log().norm();

            // Gate out abrupt pose jumps from weak/degenerate PnP solutions.
            const bool pos_ok = dp_norm < 3.0;   // meters
            const bool rot_ok = dtheta < 0.8;    // radians (~46 deg)
            if (!pos_ok || !rot_ok) {
              RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                "PnP gated: inliers=%zu/%zu ratio=%.2f reproj=%.2f dp=%.3f dtheta=%.3f",
                inliers.size(), pts3d.size(), inlier_ratio, mean_reproj_err, dp_norm, dtheta);
            } else {
              Eigen::Quaterniond q_meas(R_WB_meas);
              if (!std::isfinite(q_meas.norm()) || q_meas.norm() < 1e-12) {
                RCLCPP_WARN(get_logger(), "PnP quaternion invalid, skipping");
              } else {
                q_meas.normalize();
                if (use_direct_pnp_fusion_) {
                  EskfState& state = estimator_->getState();
                  const double a = direct_pnp_alpha_;
                  state.nominal.p = (1.0 - a) * state.nominal.p + a * p_meas;
                  Eigen::Quaterniond q_nom(state.nominal.R.matrix());
                  Eigen::Quaterniond q_fused = q_nom.slerp(a, q_meas);
                  q_fused.normalize();
                  state.nominal.R = Sophus::SO3d(q_fused);
                  state.nominal.v *= (1.0 - a);
                  last_pnp_p_ = p_meas;
                  last_pnp_R_ = Sophus::SO3d(q_meas);
                  has_last_pnp_pose_ = true;
                } else {
                  estimator_->updatePose(p_meas, Sophus::SO3d(q_meas), pos_noise, rot_noise);
                  last_pnp_p_ = p_meas;
                  last_pnp_R_ = Sophus::SO3d(q_meas);
                  has_last_pnp_pose_ = true;
                }
              }
            }
          }
        }

        static int pnp_count = 0;
        if (pnp_count++ < 5)
          RCLCPP_INFO(get_logger(), "[PnP] %zu inliers, p_meas=(%.3f,%.3f,%.3f)",
              inliers.size(), p_meas.x(), p_meas.y(), p_meas.z());
      } else if (ok) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
            "PnP rejected: inliers=%zu/%zu ratio=%.2f reproj=%.2f",
            inliers.size(), pts3d.size(), inlier_ratio, mean_reproj_err);
      }
    }

    // Landmark update path is currently disabled due unstable innovation geometry.
    // Keep PnP pose fusion active as the visual correction source.
    // if (!update_landmarks.empty())
    //   estimator_->update(update_landmarks, stereo_calib_);

    // Debug — print first 3 frames
    static int frame_count = 0;
    if (frame_count < 3) {
      RCLCPP_INFO(get_logger(),
          "[DEBUG frame %d] features=%zu stereo=%d landmarks=%zu",
          frame_count,
          frame.features.size(),
          frame.stereoCount(),
          landmarks.size());
      ++frame_count;
    }
    // if (!landmarks.empty()) estimator_->update(landmarks, stereo_calib_);

    publishOdometry(msg_left->header.stamp);
    // publishDebugImage(img_left, frame, msg_left->header.stamp);
    publishDebugImage(img_left_rect, frame, msg_left->header.stamp);
    last_img_timestamp_ = t_img;
  }

  void publishOdometry(const rclcpp::Time& stamp) {
    const NominalState& nom = estimator_->getState().nominal;
    const Eigen::Quaterniond q(nom.R.matrix());

    OdomMsg odom;
    odom.header.stamp    = stamp;
    odom.header.frame_id = world_frame_;
    odom.child_frame_id  = body_frame_;
    odom.pose.pose.position.x    = nom.p.x();
    odom.pose.pose.position.y    = nom.p.y();
    odom.pose.pose.position.z    = nom.p.z();
    odom.pose.pose.orientation.w = q.w();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    const Eigen::Vector3d v_body = nom.R.inverse() * nom.v;
    odom.twist.twist.linear.x = v_body.x();
    odom.twist.twist.linear.y = v_body.y();
    odom.twist.twist.linear.z = v_body.z();
    odom_pub_->publish(odom);

    PoseStMsg ps; ps.header = odom.header; ps.pose = odom.pose.pose;
    path_msg_.header = odom.header;
    path_msg_.poses.push_back(ps);
    if (path_msg_.poses.size() > 10000)
      path_msg_.poses.erase(path_msg_.poses.begin());
    path_pub_->publish(path_msg_);

    if (publish_tf_ && tf_broadcaster_) {
      geometry_msgs::msg::TransformStamped tf_msg;
      tf_msg.header.stamp    = stamp;
      tf_msg.header.frame_id = world_frame_;
      tf_msg.child_frame_id  = body_frame_;
      tf_msg.transform.translation.x = nom.p.x();
      tf_msg.transform.translation.y = nom.p.y();
      tf_msg.transform.translation.z = nom.p.z();
      tf_msg.transform.rotation.w = q.w();
      tf_msg.transform.rotation.x = q.x();
      tf_msg.transform.rotation.y = q.y();
      tf_msg.transform.rotation.z = q.z();
      tf_broadcaster_->sendTransform(tf_msg);
    }
  }

  void publishDebugImage(const cv::Mat& img_left,
                         const FeatureFrame& frame,
                         const rclcpp::Time& stamp) {
    if (debug_img_pub_->get_subscription_count() == 0) return;
    const cv::Mat vis = tracker_->drawTracks(img_left, frame);
    std_msgs::msg::Header hdr;
    hdr.stamp    = stamp;
    hdr.frame_id = camera_frame_;
    debug_img_pub_->publish(
        *cv_bridge::CvImage(hdr, "bgr8", vis).toImageMsg());
  }

  std::unique_ptr<ImuIntegrator>       integrator_;
  std::unique_ptr<FeatureTracker>      tracker_;
  std::unique_ptr<StereoTriangulator>  triangulator_;
  std::unique_ptr<EskfEstimator>       estimator_;

  ImuNoise              imu_noise_;
  FeatureTrackerParams  tracker_params_;
  TriangulationParams   tri_params_;
  StereoCalibration     stereo_calib_;
  EskfParams            eskf_params_;

  double init_cov_rot_=1e-4, init_cov_vel_=1e-2, init_cov_pos_=1e-4;
  double init_cov_bg_=1e-6,  init_cov_ba_=1e-4;

  std::string left_topic_, right_topic_, odom_topic_, path_topic_;
  std::string world_frame_, body_frame_, camera_frame_;
  bool        publish_tf_ = true;
  bool        use_direct_pnp_fusion_ = true;
  double      direct_pnp_alpha_ = 0.2;
  double      max_pnp_step_m_ = 1.0;
  double      max_pnp_step_rad_ = 0.6;
  double      pnp_assoc_reproj_thresh_ = 12.0;
  double      pnp_reproj_accept_thresh_ = 3.0;
  bool        has_last_pnp_pose_ = false;
  Eigen::Vector3d last_pnp_p_ = Eigen::Vector3d::Zero();
  Sophus::SO3d    last_pnp_R_;

  message_filters::Subscriber<ImageMsg> left_sub_, right_sub_;
  std::shared_ptr<Synchronizer>         sync_;
  rclcpp::Subscription<ImuMsg>::SharedPtr imu_sub_;
  rclcpp::Publisher<OdomMsg>::SharedPtr   odom_pub_;
  rclcpp::Publisher<PathMsg>::SharedPtr   path_pub_;
  rclcpp::Publisher<ImageMsg>::SharedPtr  debug_img_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

  std::deque<ImuData> imu_buffer_;
  std::mutex          imu_mutex_;
  double              last_img_timestamp_ = 0.0;
  PathMsg             path_msg_;

  cv::Mat rect_map0_x_, rect_map0_y_;
  cv::Mat rect_map1_x_, rect_map1_y_;
};

}  // namespace stereo_vio

void sigsegv_handler(int) {
  void* array[30];
  int size = backtrace(array, 30);
  fprintf(stderr, "\n=== SIGSEGV stack trace ===\n");
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  fprintf(stderr, "===========================\n");
  _exit(1);
}

void sigabrt_handler(int) {
  void* array[30];
  int size = backtrace(array, 30);
  fprintf(stderr, "\n=== SIGABRT stack trace ===\n");
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  fprintf(stderr, "===========================\n");
  _exit(1);
}

int main(int argc, char** argv) {
  signal(SIGSEGV, sigsegv_handler);
  signal(SIGABRT, sigabrt_handler);
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<stereo_vio::StereoVioNode>());
  rclcpp::shutdown();
  return 0;
}