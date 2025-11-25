#pragma once
#include<layer.hpp>
#include<error.hpp>
#include<optimizer.hpp>

class optimizer;

class sequential
{
public:
	sequential(const std::vector<std::vector<int>>&);
	~sequential();

	// Optimizer setters
	void set_optimizer_basic();
	void set_optimizer_momentum(double beta = 0.9);
	void set_optimizer_adam(double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8);
	void set_learning_rate(double lr);

	//to be deleted
	std::vector<layer*>& get_layers();
	//to be deleted

	void fit(const std::vector< Eigen::VectorXd>&, const std::vector< Eigen::VectorXd>&, const int&, const int& = 1);
	void print_network() const;
	void generate_graphviz(const std::string& filename) const;
	const std::vector<Eigen::VectorXd> predict(const std::vector<Eigen::VectorXd>& x) const;
private:
	std::vector<layer*> layers;
	std::vector< Eigen::VectorXd> x;
	std::vector< Eigen::VectorXd> y;
	int batch_size;
	error::error* error;
	optimizer* opt;

	void initialize();
	void forward_pass();
	void backpropagate();

};
