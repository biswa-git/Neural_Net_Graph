#pragma once
//#include<Eigen/Core>
#include<sequential.hpp>

class optimizer
{
public:
	optimizer();
	~optimizer();

private:

};

class momentum: public optimizer
{
public:
	momentum();
	~momentum();
	void calculate(sequential&);

private:

};
