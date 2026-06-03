#pragma once

#include <cstdint>
#include <string>


constexpr int NUM_CLASSES = 7;


enum class EmotionType: uint8_t {
    SURPRISE = 0,
    FEAR = 1,
    DISGUST = 2,
    HAPPINESS = 3,
    SADNESS = 4,
    ANGER = 5,
    NEUTRAL = 6
};


inline const char* emotion_to_string(EmotionType emotion) {
    switch (emotion) {
        case EmotionType::SURPRISE:  return "Suprise";
        case EmotionType::FEAR:      return "Fear";
        case EmotionType::DISGUST:   return "Disgust";
        case EmotionType::HAPPINESS: return "Happiness";
        case EmotionType::SADNESS:   return "Sadness";
        case EmotionType::ANGER:     return "Anger";
        case EmotionType::NEUTRAL:   return "Neutral";
        default:                     return "Unknown";
    }
}

