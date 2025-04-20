#pragma once
#include<Eigen/Core>

namespace error
{
	class error
	{
	public:
		error();
		~error();
		virtual Eigen::VectorXd calculate(const Eigen::VectorXd&, const Eigen::VectorXd&) const = 0;
		virtual Eigen::VectorXd calculate_derivative(const Eigen::VectorXd&, const Eigen::VectorXd&) const = 0;
	private:

	};

	class mse: public error
	{
	public:
		mse();
		~mse();
		Eigen::VectorXd calculate(const Eigen::VectorXd&, const Eigen::VectorXd&) const;
		Eigen::VectorXd calculate_derivative(const Eigen::VectorXd&, const Eigen::VectorXd&) const;

	private:

	};

}
