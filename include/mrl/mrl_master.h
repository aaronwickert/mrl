#include <rclcpp/rclcpp.hpp>

#include "mrl/include/mrl/mrl.h"
#include "mrl/iclude/mrl/mrl_utils.hpp"

#include "mrl/srv/get_ready.hpp"
#include "mrl/srv/get_features.hpp"
#include "mrl/srv/execute_mrl.hpp"

// MOVE TO MRL
enum class MRLState {
    NOT_READY,
    READY,
    PROCESSING
};

class MRLMaster : public rclcpp::Node {
public:
    MRLMaster();
    ~MRLMaster();

private:
    // ROS NODE SETUP
    void declare_parameters();
    void get_parameters();
    void init_publishers();
    void init_subscribers();
    void init_services();

    void mrl_setup();
    void mrl_reset();
    void mrl_discover_and_create_clients();
    bool mrl_check_for_master();
    bool mrl_check_ready();

    void get_master_features();


    void handle_execute_mrl(const std::shared_ptr<mrl::srv::ExecuteMRL::Request> request,
        std::shared_ptr<mrl::srv::ExecuteMRL::Response> response);



    // ROS VAR
    rclcpp::Service<mrl::srv::ExecuteMRL>::SharedPtr srv_execute_mrl; 
    rclcpp::Service<mrl::srv::GetFeatures>::SharedPtr srv_get_features;    
    rclcpp::Service<mrl::srv::GetReady>::SharedPtr srv_get_status;

    std::vector<rclcpp::Client<mrl::srv::GetReady>::SharedPtr> slave_clients_status;
    std::vector<rclcpp::Client<mrl::srv::GetFeatures>::SharedPtr> slave_clients_features;

    // VAR
    bool is_master = false;
    bool is_ready = false;
    MRLState state = MRLState::NOT_READY;
    std::pair<std::vector<Eigen::Vector3f>&, std::vector<Eigen::VectorXf>&> master_features;
    std::vector<std::pair<std::vector<Eigen::Vector3f>&, std::vector<Eigen::VectorXf>&>> slave_features;
    std::vector<Eigen::Quaternion> slave_transforms;

};