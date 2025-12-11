#include <../include/mrl/mrl_slave.h>


MRLSlave::MRLSlave() : Node("mrl_slave"){
    declare_parameters();
    get_parameters();
    init_publishers();
    init_subscribers();
    init_services();
    mrl_setup();
}

MRLSlave::~MRLSlave(){

}

void MRLSlave::declare_parameters(){
    RCLCPP_INFO(this->get_logger(), "Declaring ros2 parameters.");

    this->declare_parameter("is_master", false);

    RCLCPP_INFO(this->get_logger(), "Finished declaring ros2 parameters.");
}

void MRLSlave::get_parameters(){
    RCLCPP_INFO(this->get_logger(), "Getting ros2 parameters.");

    is_master = this->get_parameter("is_master").as_bool();

    RCLCPP_INFO(this->get_logger(), "Finished getting ros2 parameters.");
}

void MRLSlave::init_publishers(){
    RCLCPP_INFO(this->get_logger(), "Initializing ros2 publishers.");


    RCLCPP_INFO(this->get_logger(), "Finished initializing ros2 publishers.");
}

void MRLSlave::init_subscribers(){
    RCLCPP_INFO(this->get_logger(), "Initializing ros2 subcribers.");


    RCLCPP_INFO(this->get_logger(), "Finished initializing ros2 subcribers.");
}

void MRLSlave::init_services(){
    RCLCPP_INFO(this->get_logger(), "Initializing ros2 services.");

    srv_get_features= this->create_service<mrl::srv::GetFeatures>("get_features", 
        std::bind(&MRLSlave::handle_get_features, this, std::placeholders::_1, std::placeholders::_2));

    srv_get_status = this->create_service<mrl::srv::GetReady>("get_status",
        std::bind(&MRLSlave::handle_get_status, this, std::placeholders::_1, std::placeholders::_2));
    
    RCLCPP_INFO(this->get_logger(), "Finished initializing ros2 services.");
}

bool MRLSlave::mrl_check_for_master(){
    auto node_names = this->get_node_names();
    return std::any_of(node_names.begin(), node_names.end(), [&](const std::string& s){
        return s.find("mrl_master") != std::string::npos;
    });
}

void MRLSlave::mrl_setup(){
    RCLCPP_INFO(this->get_logger(), "Starting MRL setup.");

master_check:
    if(mrl_check_for_master()){
        RCLCPP_INFO(this->get_logger(), "MRL master running. MLR slave READY.");
        this->state = MRLState::READY;
    }
    else {
        RCLCPP_WARN(this->get_logger(), "No MRL master running. Retrying in 3 seconds.");
        std::this_thread::sleep_for(std::chrono::seconds(3));
        goto master_check;
    } 


    RCLCPP_INFO(this->get_logger(), "Finished MRL setup.");
}


void MRLSlave::handle_get_features(const std::shared_ptr<mrl::srv::GetFeatures::Request> request,
        std::shared_ptr<mrl::srv::GetFeatures::Response> response) {


}

void MRLSlave::handle_get_status(const std::shared_ptr<mrl::srv::GetReady::Request> request, 
        std::shared_ptr<mrl::srv::GetReady::Response> response){
    if(state == MRLState::READY){
        response->ready = true;
    }
    else {
        response->ready = false;
    }
}

std::string MRLSlave::extract_namespace(const std::string& full_node_name)
{
    size_t last_slash = full_node_name.find_last_of('/');
    if (last_slash == 0) return "/";
    if (last_slash != std::string::npos) {
      return full_node_name.substr(0, last_slash);
    }
    return "/";
}


int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<MRLSlave>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}