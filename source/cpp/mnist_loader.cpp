#include "mnist_loader.hpp"
#include <iostream>
#include <cmath>
#include <windows.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

MNISTLoader::MNISTLoader(int width, int height)
	: image_width(width), image_height(height)
{
}

MNISTLoader::~MNISTLoader()
{
}

bool MNISTLoader::loadTrainingData(const std::string& directory_path)
{
	// Check if directory exists
	DWORD attrib = GetFileAttributesA(directory_path.c_str());
	if (attrib == INVALID_FILE_ATTRIBUTES || !(attrib & FILE_ATTRIBUTE_DIRECTORY)) {
		std::cerr << "Directory does not exist: " << directory_path << std::endl;
		return false;
	}

	samples.clear();
	int total_loaded = 0;

	// Iterate through digit directories (0-9)
	for (int digit = 0; digit < 10; ++digit) {
		std::string digit_dir = directory_path + "\\" + std::to_string(digit);

		// Check if digit directory exists
		attrib = GetFileAttributesA(digit_dir.c_str());
		if (attrib == INVALID_FILE_ATTRIBUTES || !(attrib & FILE_ATTRIBUTE_DIRECTORY)) {
			std::cerr << "Digit directory not found: " << digit_dir << std::endl;
			continue;
		}

		int digit_count = 0;

		// Load all JPEG files in the digit directory
		WIN32_FIND_DATAA findFileData;
		HANDLE findHandle = FindFirstFileA((digit_dir + "\\*").c_str(), &findFileData);

		if (findHandle == INVALID_HANDLE_VALUE) {
			std::cerr << "Failed to open directory: " << digit_dir << std::endl;
			continue;
		}

		do {
			// Skip directories
			if (findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
				continue;
			}

			std::string filename = findFileData.cFileName;
			size_t dot_pos = filename.find_last_of(".");
			if (dot_pos == std::string::npos) {
				continue;
			}

			std::string extension = filename.substr(dot_pos);
			// Convert to lowercase for comparison
			std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

			// Check if file is JPEG
			if (extension == ".jpg" || extension == ".jpeg") {
				Sample sample;
				sample.label = digit;

				std::string filepath = digit_dir + "\\" + filename;
				if (loadImageFromFile(filepath, sample.image)) {
					samples.push_back(sample);
					digit_count++;
					total_loaded++;
				}
				else {
					std::cerr << "Failed to load image: " << filepath << std::endl;
				}
			}
		} while (FindNextFileA(findHandle, &findFileData));

		FindClose(findHandle);
		std::cout << "Loaded " << digit_count << " images for digit " << digit << std::endl;
	}

	std::cout << "Total images loaded: " << total_loaded << std::endl;
	return total_loaded > 0;
}

bool MNISTLoader::loadImageFromFile(const std::string& filepath, Eigen::VectorXd& image_data)
{
	int width, height, channels;
	unsigned char* image_data_raw = stbi_load(filepath.c_str(), &width, &height, &channels, 0);

	if (!image_data_raw) {
		std::cerr << "stbi_load failed for " << filepath << ": " << stbi_failure_reason() << std::endl;
		return false;
	}

	// Resize to target dimensions if necessary
	if (width != image_width || height != image_height) {
		std::cerr << "Warning: Image size mismatch. Expected " << image_width << "x" << image_height
			<< ", got " << width << "x" << height << " for " << filepath << std::endl;
		stbi_image_free(image_data_raw);
		return false;
	}

	image_data.resize(image_width * image_height);

	// Convert to grayscale and normalize to [0, 1]
	for (int i = 0; i < image_width * image_height; ++i) {
		if (channels == 1) {
			// Grayscale image
			image_data[i] = image_data_raw[i] / 255.0;
		}
		else if (channels == 3) {
			// RGB image
			unsigned char r = image_data_raw[i * 3];
			unsigned char g = image_data_raw[i * 3 + 1];
			unsigned char b = image_data_raw[i * 3 + 2];
			image_data[i] = rgbToGrayscale(r, g, b) / 255.0;
		}
		else if (channels == 4) {
			// RGBA image
			unsigned char r = image_data_raw[i * 4];
			unsigned char g = image_data_raw[i * 4 + 1];
			unsigned char b = image_data_raw[i * 4 + 2];
			image_data[i] = rgbToGrayscale(r, g, b) / 255.0;
		}
		else {
			stbi_image_free(image_data_raw);
			return false;
		}
	}

	stbi_image_free(image_data_raw);
	return true;
}

double MNISTLoader::rgbToGrayscale(unsigned char r, unsigned char g, unsigned char b) const
{
	// Standard grayscale conversion formula
	return 0.299 * r + 0.587 * g + 0.114 * b;
}

const std::vector<MNISTLoader::Sample>& MNISTLoader::getSamples() const
{
	return samples;
}

size_t MNISTLoader::getSampleCount() const
{
	return samples.size();
}

void MNISTLoader::shuffle()
{
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(samples.begin(), samples.end(), g);
}

void MNISTLoader::normalize()
{
	for (auto& sample : samples) {
		// Pixel values should already be in [0, 1] from loading, but ensure it
		sample.image = sample.image.cwiseMin(1.0).cwiseMax(0.0);
	}
}

void MNISTLoader::splitData(double train_ratio,
	std::vector<Sample>& train_samples,
	std::vector<Sample>& test_samples)
{
	train_samples.clear();
	test_samples.clear();

	size_t train_count = static_cast<size_t>(samples.size() * train_ratio);

	train_samples.insert(train_samples.end(), samples.begin(), samples.begin() + train_count);
	test_samples.insert(test_samples.end(), samples.begin() + train_count, samples.end());
}

void MNISTLoader::convertToModelInput(
	const std::vector<Sample>& samples,
	std::vector<Eigen::VectorXd>& input_data,
	std::vector<Eigen::VectorXd>& output_data)
{
	input_data.clear();
	output_data.clear();

	if (samples.empty()) {
		return;
	}

	// Create single input vector with all pixel values
	int total_samples = samples.size();
	Eigen::VectorXd input(total_samples);
	std::vector<Eigen::VectorXd> outputs(10); // 10 output nodes for digits 0-9

	for (int i = 0; i < 10; ++i) {
		outputs[i].resize(total_samples);
		outputs[i].setZero();
	}

	// Flatten all images into single vectors
	for (int i = 0; i < image_width * image_height; ++i) {
		input.resize(i + 1);
		Eigen::VectorXd pixel_column(total_samples);

		for (size_t j = 0; j < samples.size(); ++j) {
			pixel_column[j] = samples[j].image[i];
		}

		input_data.push_back(pixel_column);
	}

	// Create one-hot encoded outputs
	for (size_t i = 0; i < samples.size(); ++i) {
		int label = samples[i].label;
		for (int j = 0; j < 10; ++j) {
			outputs[j][i] = (j == label) ? 1.0 : 0.0;
		}
	}

	output_data = outputs;
}
