#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <unordered_map>
#include <cstdint>

namespace stereo_vio {

struct FeatureTrackerParams {
  int    fast_threshold       = 20;
  int    max_features         = 150;
  int    min_feature_distance = 20;
  int    grid_rows            = 4;
  int    grid_cols            = 5;
  int    klt_window_size      = 21;
  int    klt_max_level        = 3;
  double klt_max_error        = 30.0;
  double fb_error_threshold   = 1.0;
  double stereo_row_threshold = 1.5;
  double min_depth            = 0.2;
  double max_depth            = 20.0;
};

struct TrackedFeature {
  uint64_t    id         = 0;
  cv::Point2f pt_left;
  cv::Point2f pt_right;
  bool        has_stereo = false;
  int         age        = 0;
};

struct FeatureFrame {
  double                      timestamp = 0.0;
  std::vector<TrackedFeature> features;
  int stereoCount() const {
    int n = 0;
    for (const auto& f : features) n += f.has_stereo ? 1 : 0;
    return n;
  }
};

class FeatureTracker {
public:
  FeatureTracker() = default;
  explicit FeatureTracker(const FeatureTrackerParams& params);
  void setParams(const FeatureTrackerParams& params);
  FeatureFrame track(double timestamp,
                     const cv::Mat& img_left,
                     const cv::Mat& img_right);
  cv::Mat drawTracks(const cv::Mat& img_left,
                     const FeatureFrame& frame) const;
  int activeCount() const { return static_cast<int>(active_left_.size()); }

private:
  void trackTemporal(const cv::Mat& img_left_prev,
                     const cv::Mat& img_left_curr);
  void trackStereo(const cv::Mat& img_left_curr,
                   const cv::Mat& img_right_curr);
  std::vector<uchar> forwardBackwardCheck(
      const cv::Mat& img_src,
      const cv::Mat& img_dst,
      const std::vector<cv::Point2f>& pts_src,
      const std::vector<cv::Point2f>& pts_dst) const;
  void detectNewFeatures(const cv::Mat& img_left);
  cv::Mat buildOccupancyMask(const cv::Size& img_size) const;
  uint64_t nextId() { return next_id_++; }

  FeatureTrackerParams             params_;
  std::vector<uint64_t>            active_ids_;
  std::vector<cv::Point2f>         active_left_;
  std::vector<cv::Point2f>         active_right_;
  std::vector<bool>                active_has_stereo_;
  std::vector<int>                 active_age_;
  cv::Mat                          prev_left_;
  bool                             has_prev_ = false;
  uint64_t                         next_id_  = 0;
  cv::Ptr<cv::FastFeatureDetector> detector_;
  cv::TermCriteria                 klt_criteria_;
};

}  // namespace stereo_vio
