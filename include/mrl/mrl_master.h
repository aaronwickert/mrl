#include <rclcpp/rclcpp.hpp>

#include "mrl/srv/get_ready.hpp"
#include "mrl/srv/get_features.hpp"
#include "mrl/srv/execute_mrl.hpp"


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
    void mrl_discover_and_create_clients();
    bool mrl_check_for_master();
    bool mrl_check_ready();

    std::string extract_namespace(const std::string& full_node_name);

    void handle_execute_mrl(const std::shared_ptr<mrl::srv::ExecuteMRL::Request> request,
        std::shared_ptr<mrl::srv::ExecuteMRL::Response> response);



    // ROS STORAGE
    rclcpp::Service<mrl::srv::ExecuteMRL>::SharedPtr srv_execute_mrl; 
    rclcpp::Service<mrl::srv::GetFeatures>::SharedPtr srv_get_features;    
    rclcpp::Service<mrl::srv::GetReady>::SharedPtr srv_get_status;

    std::vector<rclcpp::Client<mrl::srv::GetReady>::SharedPtr> slave_clients_status;
    std::vector<rclcpp::Client<mrl::srv::GetFeatures>::SharedPtr> slave_clients_features;

    // STORAGE
    bool is_master = false;
    bool is_ready = false;
    MRLState state = MRLState::NOT_READY;
};