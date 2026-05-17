#include "emotion_model.hpp"
#include "emotion_model_data.h"
#include "esp_log.h"
#include <cmath>
#include <cstring>

static const char *TAG = "EMO_MODEL";

const char* EmotionModel::EMOTION_LABELS[7] = {
    "Surprise", "Fear", "Disgust", "Happiness", "Sadness", "Anger", "Neutral"
};

EmotionModel::EmotionModel() : 
    dl::Model((const char *)model_espdl, NULL, dl::fbs::MODEL_LOCATION_IN_FLASH_RODATA, 0),
    m_max_idx(6) 
{
    for(int i=0; i<7; i++) m_probs[i] = 0.0f;
}

void EmotionModel::inference(dl::image::img_t &img) {
    dl::TensorBase *input_tensor = this->get_input();
    
    // 1. Precise Normalization [ (x/255.0 - mean) / std ]
    const float MEAN = 0.4793299436569214f;
    const float STD = 0.23705872893333435f;
    
    uint8_t *src = (uint8_t *)img.data;
    if (input_tensor->dtype == dl::DATA_TYPE_INT8) {
        int8_t *dst = (int8_t *)input_tensor->data;
        float scale = powf(2.0f, (float)(int)input_tensor->exponent);
        for (int i = 0; i < img.width * img.height; i++) {
            float normalized = ((src[i] / 255.0f) - MEAN) / STD;
            int quantized = (int)roundf(normalized / scale);
            
            // Essential clamping for INT8
            if (quantized > 127) quantized = 127;
            if (quantized < -128) quantized = -128;
            
            dst[i] = (int8_t)quantized;
        }
    }

    // Run Forward
    this->run();
    
    // 2. Output Handling
    dl::TensorBase *output_tensor = this->get_output();
    float raw_scores[7];
    float max_raw = -1e9;
    
    for (int i = 0; i < 7; i++) {
        if (output_tensor->dtype == dl::DATA_TYPE_INT8) {
            raw_scores[i] = (float)((int8_t *)output_tensor->data)[i];
        } else {
            raw_scores[i] = ((float *)output_tensor->data)[i];
        }
        if (raw_scores[i] > max_raw) max_raw = raw_scores[i];
    }

    // 3. Softmax for Percentages
    float sum_exp = 0;
    for (int i = 0; i < 7; i++) {
        m_probs[i] = expf(raw_scores[i] - max_raw); 
        sum_exp += m_probs[i];
    }

    m_max_idx = 0;
    for (int i = 0; i < 7; i++) {
        m_probs[i] /= sum_exp;
        if (m_probs[i] > m_probs[m_max_idx]) m_max_idx = i;
    }
}

const char* EmotionModel::get_top_emotion(float &confidence) {
    confidence = m_probs[m_max_idx] * 100.0f;
    return EMOTION_LABELS[m_max_idx];
}
