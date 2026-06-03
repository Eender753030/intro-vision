#pragma once

#include <array>

#include "dl_model_base.hpp"
#include "dl_image_define.hpp"
#include "utils/emotion_type.hpp"


class EmotionModel: public dl::Model {
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
    EmotionType get_top_emotion(float &confidence);

    /**
     * @brief Get the full emotion probabilities array
     */
    const std::array<float, NUM_CLASSES> &get_probabilities() const { return m_probs; }

private:
    std::array<float, NUM_CLASSES> m_probs;
    uint8_t m_max_idx;
};
