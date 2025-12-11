#include <../include/mrl/mrl.h>

#include <kiss_matcher/FasterPFH.hpp>
#include <kiss_matcher/GncSolver.hpp>
#include <kiss_matcher/KISSMatcher.hpp>


std::vector<Eigen::Vector3f> MRL::convert_cloud_to_vec(const pcl::PointCloud<pcl::PointXYZ>& cloud) {
  std::vector<Eigen::Vector3f> vec;
  vec.reserve(cloud.size());
  for (const auto& pt : cloud.points) {
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) continue;
    vec.emplace_back(pt.x, pt.y, pt.z);
  }
  return vec;
}

std::pair<std::vector<Eigen::Vector3f>& , std::vector<Eigen::VectorXf>&> MRL::get_fpfh(const pcl::PointCloud<pcl::PointXYZ>& pcl){
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*pcl, *pcl, indices);  
    const auto& vec = convertCloudToVec(*pcl);
    return KISSMatcher::extract_feature(vec);
}

kiss_matcher::RegistrationSolution MRL::registrate(std::pair<std::vector<Eigen::Vector3f>& , std::vector<Eigen::VectorXf>&> src, std::pair<std::vector<Eigen::Vector3f>& , std::vector<Eigen::VectorXf>&> tgt){
    return KISSMatcher::registrate(src, tgt);
}