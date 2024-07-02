#include "activation.hpp"
#include "activation.hpp"

linear_activation::linear_activation()
{
}

linear_activation::~linear_activation()
{
}

Eigen::VectorXd linear_activation::calculate(const Eigen::VectorXd&)
{
    return Eigen::VectorXd();
}

activation::activation()
{
}

activation::~activation()
{
}
