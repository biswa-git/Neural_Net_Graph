#pragma once
//#include<Eigen/Core>
#pragma once
#include<sequential.hpp>

class sequential;

class optimizer
{
public:
	optimizer();
	~optimizer();
	virtual void calculate(sequential& model) = 0;
private:

};

class basic : public optimizer
{
public:
	basic();
	~basic();
	void calculate(sequential&);

private:

};

class momentum : public optimizer
{
public:
	momentum();
	~momentum();
	void calculate(sequential&);

private:

};
