/* stb_image - v2.27 - public domain image loader - http://nothings.org/stb_image.h

   no warranty implied; use at your own risk

   This header file is a minimal version of stb_image for JPEG support.
   For the full version, visit https://github.com/nothings/stb/blob/master/stb_image.h
*/

#ifndef STB_IMAGE_H_INCLUDED
#define STB_IMAGE_H_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

typedef unsigned char stbi_uc;
typedef unsigned short stbi_us;

extern int stbi_load_from_memory(stbi_uc const *buffer, int len, int *x, int *y, int *channels_in_file, int desired_channels);
extern unsigned char *stbi_load(char const *filename, int *x, int *y, int *channels_in_file, int desired_channels);
extern void stbi_image_free(void *retval_from_stbi_load);
extern const char *stbi_failure_reason(void);

#ifdef __cplusplus
}
#endif

#ifdef STB_IMAGE_IMPLEMENTATION

#include <cstdio>
#include <cstdlib>
#include <cstring>

static const char* stbi__g_failure_reason = "";
static ULONG_PTR stbi__gdiplus_token = 0;
static bool stbi__gdiplus_initialized = false;

const char *stbi_failure_reason(void)
{
	return stbi__g_failure_reason;
}

static void stbi__err(const char *str)
{
	stbi__g_failure_reason = str;
}

#ifdef _WIN32
	#include <windows.h>
	#pragma comment(lib, "gdiplus.lib")
	#include <gdiplus.h>
	using namespace Gdiplus;

	static void stbi__gdiplus_init()
	{
		if (!stbi__gdiplus_initialized) {
			GdiplusStartupInput gdiplusStartupInput;
			Gdiplus::Status status = Gdiplus::GdiplusStartup(&stbi__gdiplus_token, &gdiplusStartupInput, NULL);
			if (status != Gdiplus::Ok) {
				stbi__err("GDI+ initialization failed");
				stbi__gdiplus_token = 0;
				stbi__gdiplus_initialized = false;
				return;
			}
			stbi__gdiplus_initialized = true;
		}
	}
#endif

unsigned char *stbi_load(char const *filename, int *x, int *y, int *channels_in_file, int desired_channels)
{
	if (!filename) {
		stbi__err("filename is NULL");
		return NULL;
	}

#ifdef _WIN32
	stbi__gdiplus_init();

	if (!stbi__gdiplus_initialized || stbi__gdiplus_token == 0) {
		stbi__err("GDI+ not initialized");
		return NULL;
	}

	// Convert filename to wide string
	int len = MultiByteToWideChar(CP_ACP, 0, filename, -1, NULL, 0);
	wchar_t* wide_filename = new wchar_t[len];
	if (!wide_filename) {
		stbi__err("Out of memory");
		return NULL;
	}
	MultiByteToWideChar(CP_ACP, 0, filename, -1, wide_filename, len);

	// Load image using GDI+
	Gdiplus::Image* image = Gdiplus::Image::FromFile(wide_filename);
	delete[] wide_filename;

	if (!image || image->GetLastStatus() != Gdiplus::Ok) {
		stbi__err("Failed to load image with GDI+");
		if (image) delete image;
		return NULL;
	}

	*x = image->GetWidth();
	*y = image->GetHeight();
	*channels_in_file = 3; // RGB
	
	if (desired_channels == 0) desired_channels = 3;

	// Create bitmap and convert to desired format
	Gdiplus::Bitmap* bitmap = new Gdiplus::Bitmap(*x, *y, PixelFormat24bppRGB);
	if (!bitmap) {
		stbi__err("Failed to create bitmap");
		delete image;
		return NULL;
	}

	Gdiplus::Graphics g(bitmap);
	if (g.GetLastStatus() != Gdiplus::Ok) {
		stbi__err("Failed to create graphics object");
		delete bitmap;
		delete image;
		return NULL;
	}

	g.DrawImage(image, 0, 0);

	// Extract pixel data
	Gdiplus::BitmapData bitmapData;
	Gdiplus::Rect rect(0, 0, *x, *y);
	Gdiplus::Status lockStatus = bitmap->LockBits(&rect, Gdiplus::ImageLockModeRead, PixelFormat24bppRGB, &bitmapData);
	
	if (lockStatus != Gdiplus::Ok) {
		stbi__err("Failed to lock bitmap bits");
		delete bitmap;
		delete image;
		return NULL;
	}

	unsigned char* result = (unsigned char*)malloc(*x * *y * desired_channels);
	if (!result) {
		stbi__err("Out of memory");
		bitmap->UnlockBits(&bitmapData);
		delete bitmap;
		delete image;
		return NULL;
	}

	unsigned char* src = (unsigned char*)bitmapData.Scan0;
	unsigned char* dst = result;

	for (int row = 0; row < *y; ++row) {
		for (int col = 0; col < *x; ++col) {
			unsigned char b = src[row * bitmapData.Stride + col * 3];
			unsigned char g_val = src[row * bitmapData.Stride + col * 3 + 1];
			unsigned char r = src[row * bitmapData.Stride + col * 3 + 2];

			if (desired_channels == 1) {
				// Convert to grayscale
				*dst++ = (unsigned char)(0.299 * r + 0.587 * g_val + 0.114 * b);
			}
			else {
				*dst++ = r;
				*dst++ = g_val;
				*dst++ = b;
				if (desired_channels == 4) {
					*dst++ = 255; // Alpha
				}
			}
		}
	}

	bitmap->UnlockBits(&bitmapData);
	delete bitmap;
	delete image;

	return result;

#else
	stbi__err("Image loading not implemented for this platform");
	return NULL;
#endif
}

void stbi_image_free(void *retval_from_stbi_load)
{
	free(retval_from_stbi_load);
}

#endif // STB_IMAGE_IMPLEMENTATION

#endif // STB_IMAGE_H_INCLUDED
