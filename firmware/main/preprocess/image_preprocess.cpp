#include "image_preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>


namespace preprocess {

void normalize_and_quantize(const dl::image::img_t &src_img, dl::TensorBase *dest_tensor) {
    constexpr float MEAN = 0.4793299436569214F;
    constexpr float STD = 0.23705872893333435F;
    
    uint8_t *src = reinterpret_cast<uint8_t *>(src_img.data);
    if (dest_tensor->dtype == dl::DATA_TYPE_INT8) {
        int8_t *dst = reinterpret_cast<int8_t *>(dest_tensor->data);
        const float scale = std::ldexp(1.0F, static_cast<int>(dest_tensor->exponent));
        const int total_pixels = src_img.width * src_img.height;
        
        const float lhs = 1 / (255.0F * STD * scale);
        const float rhs = MEAN / (STD * scale);

        for (int i = 0; i < total_pixels; i++) {
            const int quantized = static_cast<int>(roundf(src[i] * lhs - rhs));
            
            dst[i] = static_cast<int8_t>(std::clamp(quantized, -128, 127));
        }
    }
}


void crop_and_resize_rgb565_to_gray(
    const uint8_t *src, 
    int src_w, int src_h,
    int x1, int y1,
    int face_w, int face_h,
    uint8_t *dst, 
    int dst_w, int dst_h
) {
    // 1. Coordinates clamping to prevent out of bounds
    const int clean_x1 = std::clamp(x1, 0, src_w - 1);
    const int clean_y1 = std::clamp(y1, 0, src_h - 1);
    const int clean_x2 = std::clamp(x1 + face_w, 0, src_w);
    const int clean_y2 = std::clamp(y1 + face_h, 0, src_h);
    
    const int final_w = clean_x2 - clean_x1;
    const int final_h = clean_y2 - clean_y1;
    
    if (final_w < 2 || final_h < 2) {
        // Fallback: zero fill the destination
        std::memset(dst, 0, dst_w * dst_h);
        return;
    }

    // 2. Perform nearest-neighbor resizing and RGB565 to Grayscale conversion
    const float scale_x = static_cast<float>(final_w - 1) / dst_w;
    const float scale_y = static_cast<float>(final_h - 1) / dst_h;

    auto get_gray_pixel = [&](int sx, int sy) -> uint32_t {
        const int offset = (sy * src_w + sx) * 2;
        const uint16_t pixel = ((src[offset] << 8) | (src[offset + 1]));
        const uint32_t r = ((pixel >> 11) & 0x1F) << 3;
        const uint32_t g = ((pixel >> 5) & 0x3F) << 2;
        const uint32_t b = (pixel & 0x1F) << 3;
        // Convert to grayscale using fast integer weights
        return (r * 77 + g * 150 + b * 29) >> 8;
    };

    for (int y = 0; y < dst_h; y++) {
        const float fy = clean_y1 + y * scale_y;
        int sy = static_cast<int>(fy);
        if (sy >= src_h - 1) { sy = src_h - 2; }
        
        const uint32_t wy = static_cast<int>((fy - sy) * 256.0F);
        const uint32_t inv_wy = 256 - wy;

        for (int x = 0; x < dst_w; x++) {
            const float fx = clean_x1 + x * scale_x;
            int sx = static_cast<int>(fx);
            if (sx >= src_w - 1) { sx = src_w - 2; }

            const uint32_t wx = static_cast<int>((fx - sx) * 256.0F);
            const uint32_t inv_wx = 256 - wx;
            
            const uint32_t g11 = get_gray_pixel(sx, sy);  // Top left
            const uint32_t g21 = get_gray_pixel(sx + 1, sy);  // Top right
            const uint32_t g12 = get_gray_pixel(sx, sy + 1);  // Bottom left
            const uint32_t g22 = get_gray_pixel(sx + 1, sy + 1);  // Bottom right

            const uint32_t top = (g11 * inv_wx + g21 * wx) >> 8;
            const uint32_t bottom = (g12 * inv_wx + g22 * wx) >> 8;
            const uint32_t final_gray = (top * inv_wy + bottom * wy) >> 8;

            dst[y * dst_w + x] = static_cast<uint8_t>(final_gray);
        }
    }
}

} // namespace preprocess
