#pragma once
#include<sequential.hpp>

class sequential;

class optimizer
{
public:
	optimizer();
	virtual ~optimizer();
	virtual void calculate(sequential& model) = 0;
	void set_epoch_count(int);
	int get_epoch_count() const;
	void set_learning_rate(double lr);
	double get_learning_rate() const;
private:
	int t;
	double learning_rate;
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
	void set_momentum_beta(double beta);
	double get_momentum_beta() const;

private:
	double momentum_beta;

};


class adam : public optimizer
{
public:
	adam();
	~adam();
	void calculate(sequential&);
	void set_beta1(double b);
	void set_beta2(double b);
	void set_epsilon(double e);
	double get_beta1() const;
	double get_beta2() const;
	double get_epsilon() const;
private:
	double beta1;
	double beta2;
	double epsilon;
};
