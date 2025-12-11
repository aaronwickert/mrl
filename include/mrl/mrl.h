#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>  

class MRL {
    public:
        std::vector<Eigen::Vector3f> convert_cloud_to_vec(const pcl::PointCloud<pcl::PointXYZ>& cloud);
        std::pair<std::vector<Eigen::Vector3f>& , std::vector<Eigen::VectorXf>&> get_fpfh(const pcl::PointCloud<pcl::PointXYZ>::Ptr pcl);
        kiss_matcher::KISSMatcher::RegistrationSolution MRL::registrate(std::pair<std::vector<Eigen::Vector3f>& , std::vector<Eigen::VectorXf>&> src, std::pair<std::vector<Eigen::Vector3f>& , std::vector<Eigen::VectorXf>&> tgt);
};  