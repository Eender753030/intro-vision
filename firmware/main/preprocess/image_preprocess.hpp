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


/**
 * @brief Crop an RGB565 image based on a face bounding box, convert it to grayscale, and resize it to target dimensions.
 * @param src Source image buffer
 * @param src_w Source image width
 * @param src_h Source image height
 * @param x1 face boundary coordinate x
 * @param y1 face boundary coordinate y
 * @param face_w face boundary width
 * @param face_h face boundary height
 * @param dst Destination image buffer
 * @param dst_w Target image width
 * @param dst_h Target image height
 */
void crop_and_resize_rgb565_to_gray(
    const uint8_t *src, 
    int src_w, int src_h,
    int x1, int y1,
    int face_w, int face_h,
    uint8_t *dst, 
    int dst_w, int dst_h
);

} // namespace preprocess
