#pragma once
#include<layer.hpp>
#include<error.hpp>

class sequential
{
public:
	sequential(const std::vector<std::vector<int>>&);
	~sequential();

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


	void initialize();
	void forward_pass();
	void backpropagate();

};
