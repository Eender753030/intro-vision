#pragma once

#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"
#include "dl_image_base.hpp"
#include <vector>
#include <string>

class EmotionModel : public dl::Model {
public:
    /**
     * @brief Construct a new Emotion Model object
     */
    EmotionModel();

    /**
     * @brief Run inference on a grayscale image
     * 
     * @param img Input image (must be 48x48 GRAY)
     */
    void inference(dl::image::img_t &img);

    /**
     * @brief Get the top emotion name and probability
     */
    const char* get_top_emotion(float &confidence);

private:
    static const char* EMOTION_LABELS[7];
    float m_probs[7];
    int m_max_idx;
};
