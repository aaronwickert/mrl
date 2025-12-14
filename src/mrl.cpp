#include <../include/mrl/mrl.h>

MRL::MRL(){
    
}

std::pair<std::vector<Eigen::Vector3f>& , std::vector<Eigen::VectorXf>&> MRL::get_fpfh(const pcl::PointCloud<pcl::PointXYZ>::Ptr pcl) {
  clear();
  auto cloud_vec = convert_cloud_to_vec(*pcl);
  auto processInput = [&](const std::vector<Eigen::Vector3f> &input_cloud) {
  if (config_.use_voxel_sampling_) {
    return kiss_matcher::VoxelgridSampling(input_cloud, config_.voxel_size_);
  }
    return input_cloud;
  };

  auto t_init = std::chrono::high_resolution_clock::now();

  preprocessed_cloud_ = processInput(cloud_vec);

  auto t_process = std::chrono::high_resolution_clock::now();

  faster_pfh_->setInputCloud(preprocessed_cloud_);
  faster_pfh_->ComputeFeature(key_points_, descriptors_);

  auto t_extract = std::chrono::high_resolution_clock::now();

  processing_time_ =
      std::chrono::duration_cast<std::chrono::duration<double>>(t_process - t_init).count();
  extraction_time_ =
      std::chrono::duration_cast<std::chrono::duration<double>>(t_extract - t_process).count();

  return {key_points_, descriptors_};
}