#include <../include/mrl/mrl_master.h>



MRLMaster::MRLMaster() : Node("mrl_master"){
    declare_parameters();
    get_parameters();
    init_publishers();
    init_subscribers();
    init_services();
    mrl_setup();
}

MRLMaster::~MRLMaster(){

}

void MRLMaster::declare_parameters(){
    RCLCPP_INFO(this->get_logger(), "Declaring ros2 parameters.");

    this->declare_parameter("is_master", false);

    RCLCPP_INFO(this->get_logger(), "Finished declaring ros2 parameters.");
}

void MRLMaster::get_parameters(){
    RCLCPP_INFO(this->get_logger(), "Getting ros2 parameters.");

    is_master = this->get_parameter("is_master").as_bool();

    RCLCPP_INFO(this->get_logger(), "Finished getting ros2 parameters.");
}

void MRLMaster::init_publishers(){
    RCLCPP_INFO(this->get_logger(), "Initializing ros2 publishers.");


    RCLCPP_INFO(this->get_logger(), "Finished initializing ros2 publishers.");
}

void MRLMaster::init_subscribers(){
    RCLCPP_INFO(this->get_logger(), "Initializing ros2 subcribers.");


    RCLCPP_INFO(this->get_logger(), "Finished initializing ros2 subcribers.");
}

void MRLMaster::init_services(){
    RCLCPP_INFO(this->get_logger(), "Initializing ros2 services.");

    srv_execute_mrl = this->create_service<mrl::srv::ExecuteMRL>("execute_mrl",
    std::bind(&MRLMaster::handle_execute_mrl, this, std::placeholders::_1, std::placeholders::_2));
    

    
    RCLCPP_INFO(this->get_logger(), "Finished initializing ros2 services.");
}

void MRLMaster::mrl_setup(){
    RCLCPP_INFO(this->get_logger(), "Starting MRL setup.");


 

    RCLCPP_INFO(this->get_logger(), "Finished MRL setup.");
}

bool MRLMaster::mrl_check_for_master(){
    auto node_names = this->get_node_names();
    return std::any_of(node_names.begin(), node_names.end(), [&](const std::string& s){
        return s.find("mrl_master") != std::string::npos;
    });
}

bool MRLMaster::mrl_check_ready(){
    bool ready = true;
    auto request = std::make_shared<mrl::srv::GetReady::Request>();
    RCLCPP_INFO(this->get_logger(), "%d", slave_clients_status.size());
    for(auto& client : slave_clients_status){
        auto future = client->async_send_request(request);

        while(true){
            RCLCPP_INFO(this->get_logger(), "waiting for better days");
            auto result = future.wait_for(std::chrono::milliseconds(1000));
            if(result == std::future_status::ready){
                break;
            }
        }

        auto response = future.get();
        ready = ready && response->ready;
    
    }
    return ready;
}

void MRLMaster::mrl_discover_and_create_clients(){
    auto node_names = this->get_node_names();

    std::vector<std::string> namespaces;
    for(const auto& node_name : node_names){
        if(node_name.find("mrl_slave") != std::string::npos) {
            namespaces.push_back(extract_namespace(node_name));
        }
    }

    if(namespaces.size()  == 0) {
        RCLCPP_WARN(this->get_logger(), "No slaves found. Start slaves and try again.");
        return;
    }

    std::vector<std::string> services;
    for(auto ns : namespaces){
        std::string status_service = ns + "/get_status";
        std::string features_service = ns + "/get_features";
        slave_clients_status.push_back(this->create_client<mrl::srv::GetReady>(status_service));
        slave_clients_features.push_back(this->create_client<mrl::srv::GetFeatures>(features_service));
    }
}



void MRLMaster::handle_execute_mrl(const std::shared_ptr<mrl::srv::ExecuteMRL::Request> request,
        std::shared_ptr<mrl::srv::ExecuteMRL::Response> response) {
    RCLCPP_INFO(this->get_logger(), "testse");
    mrl_discover_and_create_clients();
    RCLCPP_INFO(this->get_logger(), "testse");
    if(mrl_check_ready()){
      
    }
    RCLCPP_INFO(this->get_logger(), "testse");
    response->success = true;
    
}




std::string MRLMaster::extract_namespace(const std::string& full_node_name)
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
    auto node = std::make_shared<MRLMaster>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

