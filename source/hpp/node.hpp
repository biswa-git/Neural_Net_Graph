#pragma once
#include <vector>
#include <Eigen/Core>
#include <activation.hpp>
class weight;

class node
{
public:
    node();
    ~node();

    static int count;

    int get_id();

    std::vector<weight*>& get_back_weights();
    std::vector<weight*>& get_front_weights();

    void set_bias(const double& bias);
    const double& get_bias() const;

    //void set_value(const Eigen::VectorXd& value);
    //const Eigen::VectorXd& get_value() const;

    void set_activation_value(const Eigen::VectorXd& value);
    const Eigen::VectorXd& get_activation_value() const;

    void set_derivative_value(const Eigen::VectorXd& value);
    const Eigen::VectorXd& get_derivative_value() const;

    void set_activation(activation::activation* activation_pointer);
    activation::activation* get_activation() const;

    void set_delta(const Eigen::VectorXd& delta);
    const Eigen::VectorXd& get_delta() const;

    void set_chain(const Eigen::VectorXd& chain);
    const Eigen::VectorXd& get_chain() const;

private:
    int id;
    std::vector<weight*> back_weights;
    std::vector<weight*> front_weights;
    double bias;
    //Eigen::VectorXd value;
    Eigen::VectorXd activation_value;
    Eigen::VectorXd derivative_value;
    activation::activation* activation_pointer;
    Eigen::VectorXd delta;
    Eigen::VectorXd chain;

};