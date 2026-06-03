#pragma once


#include "utils/emotion_type.hpp"


namespace haptics {

/**
 * @brief Initialize the haptic motor.
 */
void init();


/**
 * @brief Trigger a specific haptic pattern based on the recognized emotion.
 *        This function runs in a non-blocking, asynchronous FreeRTOS task.
 * 
 * Patterns (DRV2605L Library A ERM Open-Loop Mode):
 *   - SURPRISE: Strong Quadruple Click (Effect 1 x4 + Delays, 400ms duration, 3-click feel)
 *   - FEAR: Triple Heartbeat (Effect 1 + Effect 3 + Delays, 1000ms duration)
 *   - DISGUST: Septuple Rough Buzz 60% (Effect 22 x7, 2000ms duration)
 *   - SADNESS: Transition Ramp Down Long Smooth 1-100% to 0% (Effect 70, 1000ms duration)
 *   - ANGER: Long Double Strong Buzz 100% (Effect 14 x2, 1500ms duration)
 *   - HAPPINESS / NEUTRAL: Silent (no physical trigger to prevent UI cluttering)
 * 
 * @param emotion The detected emotion to actuate
 */
void trigger(EmotionType emotion);


/**
 * @brief Turn off the motor immediately.
 */
void stop();

} // namespace haptics
