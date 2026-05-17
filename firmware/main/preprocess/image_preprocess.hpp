#pragma once

#include "dl_image_define.hpp"
#include "dl_tensor_base.hpp"

namespace preprocess {

/**
 * @brief Normalize a grayscale image and quantize it to INT8 for ESP-DL models.
 * 
 * Formula: quantized = round(((pixel / 255.0) - MEAN) / STD / (2^exponent))
 * 
 * @param src_img Input grayscale image (48x48)
 * @param dest_tensor Output tensor to be populated
 */
void normalize_and_quantize(const dl::image::img_t &src_img, dl::TensorBase *dest_tensor);

} // namespace preprocess
