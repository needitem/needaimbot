#pragma once

#include "../cuda/simple_cuda_mat.h"
#include <string>

namespace ImageIO {

// Save image to file
bool saveImage(const SimpleMat& image, const std::string& filename, int quality = 95);

// Save screenshot (wrapper for saveImage with directory handling)
void saveScreenshot(const SimpleMat& frame, const std::string& directory = "screenshots");

} // namespace ImageIO