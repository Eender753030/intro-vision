#include "image_preprocess.hpp"
#include <cmath>

namespace preprocess {

void normalize_and_quantize(const dl::image::img_t &src_img, dl::TensorBase *dest_tensor) {
    const float MEAN = 0.4793299436569214f;
    const float STD = 0.23705872893333435f;
    
    uint8_t *src = (uint8_t *)src_img.data;
    if (dest_tensor->dtype == dl::DATA_TYPE_INT8) {
        int8_t *dst = (int8_t *)dest_tensor->data;
        float scale = powf(2.0f, (float)(int)dest_tensor->exponent);
        int total_pixels = src_img.width * src_img.height;
        
        for (int i = 0; i < total_pixels; i++) {
            float normalized = ((src[i] / 255.0f) - MEAN) / STD;
            int quantized = (int)roundf(normalized / scale);
            
            // Essential clamping for INT8
            if (quantized > 127) quantized = 127;
            if (quantized < -128) quantized = -128;
            
            dst[i] = (int8_t)quantized;
        }
    }
}

} // namespace preprocess
