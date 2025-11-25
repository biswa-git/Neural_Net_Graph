#pragma once
#include <Eigen/Core>
#include <vector>
#include <string>
#include <filesystem>
#include <random>
#include <algorithm>

class MNISTLoader
{
public:
	struct Sample
	{
		Eigen::VectorXd image;  // Flattened 28x28 image (784 values)
		int label;              // Digit label (0-9)
	};

	MNISTLoader(int image_width = 28, int image_height = 28);
	~MNISTLoader();

	// Load all images from the training directory
	// directory_path should point to the parent folder containing 0-9 subdirectories
	bool loadTrainingData(const std::string& directory_path);

	// Get all loaded samples
	const std::vector<Sample>& getSamples() const;

	// Get total number of samples loaded
	size_t getSampleCount() const;

	// Shuffle samples
	void shuffle();

	// Normalize pixel values to [0, 1]
	void normalize();

	// Get image dimensions
	int getImageWidth() const { return image_width; }
	int getImageHeight() const { return image_height; }
	int getImageSize() const { return image_width * image_height; }

	// Split data into training and test sets
	void splitData(double train_ratio, 
		std::vector<Sample>& train_samples,
		std::vector<Sample>& test_samples);

	// Convert samples to format expected by sequential model
	void convertToModelInput(
		const std::vector<Sample>& samples,
		std::vector<Eigen::VectorXd>& input_data,
		std::vector<Eigen::VectorXd>& output_data);

private:
	int image_width;
	int image_height;
	std::vector<Sample> samples;

	// Load image from JPEG file and convert to grayscale
	bool loadImageFromFile(const std::string& filepath, Eigen::VectorXd& image_data);

	// Convert RGB to grayscale
	double rgbToGrayscale(unsigned char r, unsigned char g, unsigned char b) const;
};
