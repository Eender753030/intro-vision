#include "emotion_model.hpp"

#include <array>
#include <cmath>
#include <cstring>

#include "emotion_model_data.h"
#include "image_preprocess.hpp"
#include "dl_tensor_base.hpp"
#include "utils/emotion_type.hpp"


EmotionModel::EmotionModel(): 
    dl::Model(reinterpret_cast<const char *>(emotion_model_data)),
    m_max_idx(NUM_CLASSES - 1) {}


void EmotionModel::inference(dl::image::img_t &img) {
    dl::TensorBase *input_tensor = this->get_input();
    
    // 1. Modular Preprocessing & Quantization
    preprocess::normalize_and_quantize(img, input_tensor);
    
    // Run Forward
    this->run();
    
    // 2. Output Handling
    const dl::TensorBase *output_tensor = this->get_output();
    std::array<float, NUM_CLASSES> raw_scores;
    float max_raw = -1e9F;
    
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (output_tensor->dtype == dl::DATA_TYPE_INT8) {
            raw_scores[i] = static_cast<float>((reinterpret_cast<int8_t *>(output_tensor->data))[i]);
        } else {
            raw_scores[i] = reinterpret_cast<float *>(output_tensor->data)[i];
        }
        if (raw_scores[i] > max_raw) { max_raw = raw_scores[i]; }
    }

    // 3. Softmax for Percentages
    float sum_exp = 0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        m_probs[i] = expf(raw_scores[i] - max_raw); 
        sum_exp += m_probs[i];
    }

    m_max_idx = 0;
    for (int i = 0; i < NUM_CLASSES; i++) {
        m_probs[i] /= sum_exp;
        if (m_probs[i] > m_probs[m_max_idx]) { m_max_idx = i; }
    }
}

EmotionType EmotionModel::get_top_emotion(float &confidence) {
    confidence = m_probs[m_max_idx] * 100.0F;
    return static_cast<EmotionType>(m_max_idx);
}
