// feature_tracker.cpp
#include "stereo_vio/feature_tracker.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <iostream>

namespace stereo_vio {

FeatureTracker::FeatureTracker(const FeatureTrackerParams& params)
{
  setParams(params);
}

void FeatureTracker::setParams(const FeatureTrackerParams& params)
{
  params_   = params;
  detector_ = cv::FastFeatureDetector::create(
      params_.fast_threshold, true,
      cv::FastFeatureDetector::TYPE_9_16);
  klt_criteria_ = cv::TermCriteria(
      cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
}

FeatureFrame FeatureTracker::track(double timestamp,
                                   const cv::Mat& img_left,
                                   const cv::Mat& img_right)
{
  cv::Mat grey_left, grey_right;
  if (img_left.channels()  == 3) cv::cvtColor(img_left,  grey_left,  cv::COLOR_BGR2GRAY);
  else grey_left  = img_left.clone();
  if (img_right.channels() == 3) cv::cvtColor(img_right, grey_right, cv::COLOR_BGR2GRAY);
  else grey_right = img_right.clone();

  if (!has_prev_) {
    detectNewFeatures(grey_left);
    prev_left_ = grey_left.clone();
    has_prev_  = true;
    FeatureFrame frame;
    frame.timestamp = timestamp;
    for (std::size_t i = 0; i < active_ids_.size(); ++i) {
      TrackedFeature f;
      f.id      = active_ids_[i];
      f.pt_left = active_left_[i];
      f.age     = active_age_[i];
      frame.features.push_back(f);
    }
    return frame;
  }

  trackTemporal(prev_left_, grey_left);
  trackStereo(grey_left, grey_right);

  if (static_cast<int>(active_ids_.size()) < params_.max_features / 2)
    detectNewFeatures(grey_left);

  for (auto& a : active_age_) ++a;

  FeatureFrame frame;
  frame.timestamp = timestamp;
  frame.features.reserve(active_ids_.size());
  for (std::size_t i = 0; i < active_ids_.size(); ++i) {
    TrackedFeature f;
    f.id         = active_ids_[i];
    f.pt_left    = active_left_[i];
    f.pt_right   = active_right_[i];
    f.has_stereo = active_has_stereo_[i];
    f.age        = active_age_[i];
    frame.features.push_back(f);
  }

  prev_left_ = grey_left.clone();
  return frame;
}

void FeatureTracker::trackTemporal(const cv::Mat& img_prev,
                                   const cv::Mat& img_curr)
{
  if (active_left_.empty()) return;
  const cv::Size win(params_.klt_window_size, params_.klt_window_size);
  std::vector<cv::Point2f> pts_curr;
  std::vector<uchar> status_fwd;
  std::vector<float> err_fwd;
  cv::calcOpticalFlowPyrLK(img_prev, img_curr, active_left_, pts_curr,
                            status_fwd, err_fwd, win,
                            params_.klt_max_level, klt_criteria_);

  std::vector<uchar> status_fb =
      forwardBackwardCheck(img_prev, img_curr, active_left_, pts_curr);

  std::size_t j = 0;
  for (std::size_t i = 0; i < active_left_.size(); ++i) {
    if (status_fwd[i] && status_fb[i]) {
      active_ids_[j]        = active_ids_[i];
      active_left_[j]       = pts_curr[i];
      active_right_[j]      = active_right_[i];
      active_has_stereo_[j] = false;
      active_age_[j]        = active_age_[i];
      ++j;
    }
  }
  active_ids_.resize(j);
  active_left_.resize(j);
  active_right_.resize(j);
  active_has_stereo_.resize(j);
  active_age_.resize(j);
}

void FeatureTracker::trackStereo(const cv::Mat& img_left,
                                 const cv::Mat& img_right)
{
  if (active_left_.empty()) return;

  static int stereo_debug = 0;
  const cv::Size win(params_.klt_window_size, params_.klt_window_size);

  // Forward pass: left→right, seeded at same row (rectified epipolar)
  std::vector<cv::Point2f> pts_right(active_left_);
  std::vector<uchar> status_fwd;
  std::vector<float> err_fwd;
  cv::calcOpticalFlowPyrLK(img_left, img_right, active_left_, pts_right,
                            status_fwd, err_fwd, win,
                            params_.klt_max_level, klt_criteria_,
                            cv::OPTFLOW_USE_INITIAL_FLOW);

  // Y-lock: force right points to same row as left (epipolar constraint)
  for (std::size_t i = 0; i < pts_right.size(); ++i)
    pts_right[i].y = active_left_[i].y;

  // Backward pass: right→left, seeded from y-locked right points
  std::vector<cv::Point2f> pts_back(pts_right);
  std::vector<uchar> status_back;
  std::vector<float> err_back;
  cv::calcOpticalFlowPyrLK(img_right, img_left, pts_right, pts_back,
                            status_back, err_back, win,
                            params_.klt_max_level, klt_criteria_,
                            cv::OPTFLOW_USE_INITIAL_FLOW);

  // Y-lock backward result too
  for (std::size_t i = 0; i < pts_back.size(); ++i)
    pts_back[i].y = active_left_[i].y;

  // FB check: only horizontal round-trip error matters
  std::vector<uchar> status_fb(active_left_.size(), 0);
  for (std::size_t i = 0; i < active_left_.size(); ++i) {
    if (!status_back[i]) continue;
    const float dx = active_left_[i].x - pts_back[i].x;
    if (std::abs(dx) < static_cast<float>(params_.fb_error_threshold))
      status_fb[i] = 1;
  }

  active_right_.resize(active_left_.size());
  active_has_stereo_.resize(active_left_.size());
  int stereo_ok_count = 0;
  int fwd_fail = 0, fb_fail = 0, disp_fail = 0;

  for (std::size_t i = 0; i < active_left_.size(); ++i) {
    if (!status_fwd[i]) {
      ++fwd_fail;
      active_right_[i] = {0, 0}; active_has_stereo_[i] = false; continue;
    }
    if (!status_fb[i]) {
      ++fb_fail;
      active_right_[i] = {0, 0}; active_has_stereo_[i] = false; continue;
    }
    const float disparity = active_left_[i].x - pts_right[i].x;
    if (disparity < 1.0f || disparity > 200.0f) {
      ++disp_fail;
      active_right_[i] = {0, 0}; active_has_stereo_[i] = false;
    } else {
      active_right_[i]      = pts_right[i];
      active_has_stereo_[i] = true;
      ++stereo_ok_count;
    }
  }

  if (stereo_debug < 5) {
    std::cout << "[STEREO debug " << stereo_debug << "] "
              << "total=" << active_left_.size()
              << " fwd_fail=" << fwd_fail
              << " fb_fail=" << fb_fail
              << " disp_fail=" << disp_fail
              << " ok=" << stereo_ok_count << "\n";
    int shown = 0;
    for (std::size_t i = 0; i < active_left_.size() && shown < 3; ++i) {
      if (status_fwd[i] && status_fb[i]) {
        std::cout << "  pt " << i
                  << " disp=" << (active_left_[i].x - pts_right[i].x)
                  << " fb_err=" << std::abs(active_left_[i].x - pts_back[i].x) << "\n";
        ++shown;
      }
    }
    ++stereo_debug;
  }
}

std::vector<uchar> FeatureTracker::forwardBackwardCheck(
    const cv::Mat& img_src,
    const cv::Mat& img_dst,
    const std::vector<cv::Point2f>& pts_src,
    const std::vector<cv::Point2f>& pts_dst) const
{
  if (pts_src.empty()) return {};
  const cv::Size win(params_.klt_window_size, params_.klt_window_size);
  std::vector<cv::Point2f> pts_back;
  std::vector<uchar> status_back;
  std::vector<float> err_back;
  cv::calcOpticalFlowPyrLK(img_dst, img_src, pts_dst, pts_back,
                            status_back, err_back, win,
                            params_.klt_max_level, klt_criteria_);
  std::vector<uchar> result(pts_src.size(), 0);
  for (std::size_t i = 0; i < pts_src.size(); ++i) {
    if (!status_back[i]) continue;
    const float dx  = pts_src[i].x - pts_back[i].x;
    const float dy  = pts_src[i].y - pts_back[i].y;
    if (std::sqrt(dx*dx + dy*dy) < static_cast<float>(params_.fb_error_threshold))
      result[i] = 1;
  }
  return result;
}

void FeatureTracker::detectNewFeatures(const cv::Mat& img_left)
{
  const int current_count = static_cast<int>(active_ids_.size());
  if (current_count >= params_.max_features) return;
  const int needed = params_.max_features - current_count;
  cv::Mat mask = buildOccupancyMask(img_left.size());

  const int cell_w = img_left.cols / params_.grid_cols;
  const int cell_h = img_left.rows / params_.grid_rows;
  std::vector<cv::KeyPoint> new_kps;
  new_kps.reserve(needed);

  for (int row = 0; row < params_.grid_rows && (int)new_kps.size() < needed; ++row) {
    for (int col = 0; col < params_.grid_cols && (int)new_kps.size() < needed; ++col) {
      const int x0 = col * cell_w;
      const int y0 = row * cell_h;
      const int w  = (col == params_.grid_cols-1) ? img_left.cols - x0 : cell_w;
      const int h  = (row == params_.grid_rows-1) ? img_left.rows - y0 : cell_h;
      cv::Rect cell_roi(x0, y0, w, h);
      std::vector<cv::KeyPoint> cell_kps;
      detector_->detect(img_left(cell_roi), cell_kps, mask(cell_roi));
      if (cell_kps.empty()) continue;
      auto best = std::max_element(cell_kps.begin(), cell_kps.end(),
          [](const cv::KeyPoint& a, const cv::KeyPoint& b){
            return a.response < b.response; });
      cv::KeyPoint kp = *best;
      kp.pt.x += static_cast<float>(x0);
      kp.pt.y += static_cast<float>(y0);
      new_kps.push_back(kp);
      cv::circle(mask,
                 cv::Point(static_cast<int>(kp.pt.x),
                           static_cast<int>(kp.pt.y)),
                 params_.min_feature_distance, cv::Scalar(0), -1);
    }
  }
  for (const auto& kp : new_kps) {
    active_ids_.push_back(nextId());
    active_left_.push_back(kp.pt);
    active_right_.push_back(cv::Point2f(0.f, 0.f));
    active_has_stereo_.push_back(false);
    active_age_.push_back(0);
  }
}

cv::Mat FeatureTracker::buildOccupancyMask(const cv::Size& img_size) const
{
  cv::Mat mask(img_size, CV_8UC1, cv::Scalar(255));
  for (const auto& pt : active_left_)
    cv::circle(mask,
               cv::Point(static_cast<int>(pt.x),
                         static_cast<int>(pt.y)),
               params_.min_feature_distance, cv::Scalar(0), -1);
  return mask;
}

cv::Mat FeatureTracker::drawTracks(const cv::Mat& img_left,
                                   const FeatureFrame& frame) const
{
  cv::Mat vis;
  if (img_left.channels() == 1) cv::cvtColor(img_left, vis, cv::COLOR_GRAY2BGR);
  else vis = img_left.clone();
  for (const auto& f : frame.features) {
    const cv::Scalar colour = f.has_stereo
        ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);
    cv::circle(vis,
               cv::Point(static_cast<int>(f.pt_left.x),
                         static_cast<int>(f.pt_left.y)),
               4, colour, -1);
  }
  return vis;
}

}  // namespace stereo_vio
