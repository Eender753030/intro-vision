#include <array>
#include <stdio.h>

#include "dl_detect_define.hpp"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_camera.h"
#include "img_converters.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"

// Custom Modules
#include "emotion_model.hpp"
#include "motor_driver.hpp"
#include "image_preprocess.hpp"
#include "human_face_detect.hpp"
#include "utils/performance_profiler.hpp"
#include "utils/emotion_type.hpp"

static constexpr char TAG[] = "MAIN";

static constexpr int FACE_SIZE = 48 * 48;

static constexpr uint32_t DEBOUNCE_THRESHOLD_MS = 100; // Must sustain negative emotion for 100ms
static constexpr uint32_t COOLDOWN_DURATION_MS = 5000;  // 5 seconds cooldown

// FreeRTOS Queue to pass image buffer pointers from Core 0 to Core 1
static QueueHandle_t xImageQueue = nullptr;


// Base64 encoder with buffered chunked stdout writes.
// Base64 encodes 3 bytes → 4 chars (1.33x overhead) vs hex's 2x overhead,
// reducing JPEG transmission payload by ~33% over the hex approach.
static void print_base64(const uint8_t *buf, size_t len) {
    static const char b64_chars[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    char chunk[769]; // 768 output chars per write = 576 input bytes
    size_t out_idx = 0;
    for (size_t i = 0; i < len; i += 3) {
        const uint8_t b0 = buf[i];
        const uint8_t b1 = (i + 1 < len) ? buf[i + 1] : 0;
        const uint8_t b2 = (i + 2 < len) ? buf[i + 2] : 0;
        chunk[out_idx++] = b64_chars[(b0 >> 2) & 0x3F];
        chunk[out_idx++] = b64_chars[((b0 << 4) | (b1 >> 4)) & 0x3F];
        chunk[out_idx++] = (i + 1 < len) ? b64_chars[((b1 << 2) | (b2 >> 6)) & 0x3F] : '=';
        chunk[out_idx++] = (i + 2 < len) ? b64_chars[b2 & 0x3F] : '=';
        if (out_idx >= 768) {
            fwrite(chunk, 1, out_idx, stdout);
            out_idx = 0;
        }
    }
    if (out_idx > 0) {
        fwrite(chunk, 1, out_idx, stdout);
    }
    putchar('\n');
}

// Core 0: Real-Time Camera Capture & Preprocessing Task
[[maybe_unused]] static void vCameraCaptureTask(void *pvParameters) {
    ESP_LOGI(TAG, "Camera Capture & Preprocess Task started on Core %d", xPortGetCoreID());

    // 1. Initialize Camera using successfully verified pins
    camera_config_t config;
    config.pin_pwdn = 42;
    config.pin_reset = 40;
    config.pin_xclk = -1; // Onboard Active Oscillator
    config.pin_sccb_sda = 39;
    config.pin_sccb_scl = 38;

    // Data Pins
    config.pin_d7 = 13;
    config.pin_d6 = 12;
    config.pin_d5 = 11;
    config.pin_d4 = 10;
    config.pin_d3 = 9;
    config.pin_d2 = 8;
    config.pin_d1 = 5;
    config.pin_d0 = 4;

    // Control Pins
    config.pin_vsync = 6;
    config.pin_href = 7;
    config.pin_pclk = 14;

    config.xclk_freq_hz = 10000000; // Lower XCLK to 10 MHz to prevent PSRAM DMA FIFO overrun (fixes green/purple stripes)
    config.ledc_timer = LEDC_TIMER_0;
    config.ledc_channel = LEDC_CHANNEL_0;
    
    config.pixel_format = PIXFORMAT_RGB565; // Capture in RGB565 for ESP-WHO Face Detection
    config.frame_size = FRAMESIZE_QVGA; // 320x240
    config.fb_count = 2; // Enable double-buffering to pipeline frame capture & face detection
    config.fb_location = CAMERA_FB_IN_PSRAM; // Store larger RGB565 frames in PSRAM
    config.grab_mode = CAMERA_GRAB_LATEST; // Always grab the latest frame to prevent lag and maximize FPS
    config.sccb_i2c_port = -1; // Auto-allocate independent I2C port for the camera

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera initialization FAILED! System halted.");
        while (1) {
            vTaskDelay(pdMS_TO_TICKS(1000));
        }
    }
    ESP_LOGI(TAG, "Camera initialized successfully on Core %d! [ESP_OK]", xPortGetCoreID());

    // Setup sensor settings (flip image to match physical mounting if needed)
    sensor_t *s = esp_camera_sensor_get();
    if (s != nullptr) {
        s->set_vflip(s, 1);
        s->set_hmirror(s, 1);
    }

    // 2. Initialize ESP-WHO Face Detector
    // Load static model MSRMNP_S8_V1 immediately (lazy_load = false)
    HumanFaceDetect *face_detector = new HumanFaceDetect(HumanFaceDetect::MSRMNP_S8_V1, false);
    ESP_LOGI(TAG, "ESP-WHO HumanFaceDetect initialized successfully!");

    uint32_t frame_count = 0;
    int64_t last_frame_time_us = 0;
    while (1) {
        // Yield briefly to let other tasks run, but keep latency minimal
        vTaskDelay(pdMS_TO_TICKS(1));

        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGE(TAG, "Failed to capture frame!");
            continue;
        }

        frame_count++;

        // Calculate and print Camera Acquisition FPS
        const int64_t now_us = esp_timer_get_time();
        if (last_frame_time_us > 0) {
            const float fps = 1000000.0f / static_cast<float>(now_us - last_frame_time_us);
            printf("CAMERA_FPS:%.2f\n", fps);
        }
        last_frame_time_us = now_us;

        // 3. Wrap camera frame buffer as an image for ESP-WHO
        dl::image::img_t img;
        img.width = fb->width;
        img.height = fb->height;
        img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_RGB565BE; // ESP-WHO Face Detection expects RGB565BE
        img.data = fb->buf;

        // 4. Execute Face Detection with latency profiling
        const int64_t detect_start = esp_timer_get_time();
        std::list<dl::detect::result_t> &results = face_detector->run(img);
        const float detect_latency_ms = static_cast<float>(esp_timer_get_time() - detect_start) / 1000.0f;
        printf("FACE_DETECT_LATENCY:%.2f\n", detect_latency_ms);

        if (frame_count % 20 == 0) {
            ESP_LOGI(TAG, "[Core 0] Scanning... Checked %lu frames. Face detected: %s",
                     static_cast<unsigned long>(frame_count), results.empty() ? "No" : "Yes");
        }

        // 5. Reactive Check: Bypassing if no face is detected
        if (results.empty()) {
            // Check if we should send a background preview frame (once every 5 frames for much higher responsiveness)
            if (frame_count % 2 == 0) {
                uint8_t *jpg_buf = nullptr;
                size_t jpg_len = 0;
                // Quality=20: good enough for debug preview, significantly reduces frame payload
                if (frame2jpg(fb, 20, &jpg_buf, &jpg_len)) {
                    printf("FACE_BOX:0,0,0,0\n");
                    printf("---BEGIN_B64---\n");
                    print_base64(jpg_buf, jpg_len);
                    printf("---END_B64---\n");
                    free(jpg_buf);
                }
            }
            // Immediately return the camera frame buffer to the driver
            esp_camera_fb_return(fb);
            continue;
        }

        ESP_LOGI(TAG, ">>> SUCCESS: Face Detected! Crop & resize to 48x48...");

        // Select the first detected face box
        const dl::detect::result_t &face = results.front();
        const int x1 = static_cast<int>(face.box[0]);
        const int y1 = static_cast<int>(face.box[1]);
        const int face_w = static_cast<int>(face.box[2] - x1);
        const int face_h = static_cast<int>(face.box[3] - y1);
        
        // Always send face coordinates every frame for smooth neon box tracking on the UI HUD
        printf("FACE_BOX:%d,%d,%d,%d\n", x1, y1, face_w, face_h);
        
        // Send JPEG once every 2 frames when face is detected to prevent clogging UART.
        // We still run face detection and send the FACE_BOX coordinates every single frame for smooth HUD tracking.
        if (frame_count % 2 == 0) {
            uint8_t *jpg_buf = nullptr;
            size_t jpg_len = 0;
            if (frame2jpg(fb, 20, &jpg_buf, &jpg_len)) {
                printf("---BEGIN_B64---\n");
                print_base64(jpg_buf, jpg_len);
                printf("---END_B64---\n");
                free(jpg_buf);
            }
        }

        // Allocate 48x48 buffer dynamically in internal fast SRAM
        uint8_t *crop_buf = reinterpret_cast<uint8_t *>(heap_caps_malloc(FACE_SIZE, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL));
        if (crop_buf == nullptr) {
            ESP_LOGE(TAG, "Failed to allocate SRAM for resized face buffer!");
            esp_camera_fb_return(fb);
            continue;
        }

        // Crop the exact face box region from RGB565, convert to grayscale and resize to 48x48
        preprocess::crop_and_resize_rgb565_to_gray(fb->buf, fb->width, fb->height, x1, y1, face_w, face_h, crop_buf, 48, 48);

        // Immediately return the camera frame buffer to the driver
        esp_camera_fb_return(fb);

        // Zero-copy: push the pointer to the queue for Core 1 to process and free
        if (xQueueSend(xImageQueue, &crop_buf, pdMS_TO_TICKS(50)) != pdPASS) {
            ESP_LOGE(TAG, "Queue Full! Discarding frame...");
            heap_caps_free(crop_buf);
        }
    }

    delete face_detector;
}


// Core 1: Inference & Anti-Jitter State Machine Task
[[maybe_unused]] static void vEmotionInferenceTask(void *pvParameters) {
    ESP_LOGI(TAG, "Inference & Haptic Task started on Core %d", xPortGetCoreID());

    // Initialize Model and Motor Driver
    EmotionModel *emotion_model = new EmotionModel();
    haptics::init();

    // Anti-Jitter State Variables (tracks last triggered emotion for debounce; NEUTRAL = idle)
    EmotionType last_emotion = EmotionType::NEUTRAL;
    uint64_t negative_start_time_ms = 0;
    uint64_t cooldown_end_time_ms = 0;

    while (1) {
        uint8_t *image_data = nullptr;

        // Block indefinitely until an image is received from Core 0
        if (xQueueReceive(xImageQueue, &image_data, portMAX_DELAY) == pdPASS) {
            
            // 1. Performance Profiling (Start Timer)
            profiler::start_timer();

            // 2. Wrap as Image and Execute Inference
            dl::image::img_t gray_img;
            gray_img.width = 48;
            gray_img.height = 48;
            gray_img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_GRAY;
            gray_img.data = image_data;

            emotion_model->inference(gray_img);

            // 3. Performance Profiling (Stop Timer & Log heap)
            const float latency_ms = profiler::stop_timer_ms();
            ESP_LOGI(TAG, ">>> ESP-DL Inference Latency: %.2f ms <<<", latency_ms);
            profiler::log_heap_status();

            // 4. Decision Processing
            float confidence = 0;
            const EmotionType current_emotion = emotion_model->get_top_emotion(confidence);

            // Print full probability distribution for Web UI
            const std::array<float, NUM_CLASSES> &probs = emotion_model->get_probabilities();
            printf("EMOTION_PROBS:Surprise:%.4f,Fear:%.4f,Disgust:%.4f,Happiness:%.4f,Sadness:%.4f,Anger:%.4f,Neutral:%.4f\n",
                   probs[0], probs[1], probs[2], probs[3], probs[4], probs[5], probs[6]);

            const uint64_t now_ms = esp_timer_get_time() / 1000;
            const char *emotion_name = emotion_to_string(current_emotion);
            ESP_LOGI(TAG, "Prediction: %s (Confidence: %.1f%%)", emotion_name, confidence);
            printf("Final Decision: %s\n", emotion_name); // Compatibility with test_esp32_hardware.py
            printf("DONE\n"); // Acknowledge completion of this frame for the PC script

            // 5. Dual-Jitter / Debounce Filtering State Machine
            // HAPPINESS is now included: it uses the same debounce path with its own gentle haptic pattern.
            // Only NEUTRAL is treated as "no action" and resets the timer.
            const bool is_triggerable = (current_emotion != EmotionType::NEUTRAL);

            if (is_triggerable) {
                if (current_emotion == last_emotion) {
                    // Continuous detection of the same negative emotion
                    const uint64_t elapsed = now_ms - negative_start_time_ms;
                    ESP_LOGI(TAG, "[DEBOUNCE] Sustained %s: %llu ms / %u ms", emotion_name, static_cast<unsigned long long>(elapsed), static_cast<unsigned int>(DEBOUNCE_THRESHOLD_MS));
                    printf("DEBOUNCE:SUSTAIN:%llu:%u\n", static_cast<unsigned long long>(elapsed), static_cast<unsigned int>(DEBOUNCE_THRESHOLD_MS));

                    if (elapsed >= DEBOUNCE_THRESHOLD_MS) {
                        // Check if cooling down
                        if (now_ms >= cooldown_end_time_ms) {
                            ESP_LOGI(TAG, "[DEBOUNCE] >>> SUCCESS: Triggering Haptic Feedback for %s <<<", emotion_name);
                            printf("HAPTIC_TRIGGER:%s,%u\n", emotion_name, static_cast<unsigned int>(current_emotion));
                            haptics::trigger(current_emotion);
                            
                            // Start Cooldown period
                            cooldown_end_time_ms = now_ms + COOLDOWN_DURATION_MS;
                            // Immediately signal the UI to start its smooth countdown from full duration
                            printf("DEBOUNCE:COOLDOWN:%u:%u\n", static_cast<unsigned int>(COOLDOWN_DURATION_MS), static_cast<unsigned int>(COOLDOWN_DURATION_MS));
                            
                            // Reset debouncing
                            last_emotion = EmotionType::NEUTRAL;
                            negative_start_time_ms = 0;
                        } else {
                            ESP_LOGI(TAG, "[DEBOUNCE] Warning verified but skipped due to ACTIVE COOLDOWN (Remaining: %llu ms)", 
                                     static_cast<unsigned long long>(cooldown_end_time_ms - now_ms));
                            printf("DEBOUNCE:COOLDOWN:%llu:%u\n", static_cast<unsigned long long>(cooldown_end_time_ms - now_ms), static_cast<unsigned int>(COOLDOWN_DURATION_MS));
                        }
                    }
                } else {
                    // New emotion detected (different from last): start validation timer
                    last_emotion = current_emotion;
                    negative_start_time_ms = now_ms;
                    // Use INFO for positive emotion, ERROR for negative — reflected in log console colors
                    if (current_emotion == EmotionType::HAPPINESS) {
                        ESP_LOGI(TAG, "[DEBOUNCE] Positive: %s detected. Debouncing for %u ms...", emotion_name, static_cast<unsigned int>(DEBOUNCE_THRESHOLD_MS));
                    } else {
                        ESP_LOGE(TAG, "[DEBOUNCE] New Warning: %s detected. Debouncing for %u ms...", emotion_name, static_cast<unsigned int>(DEBOUNCE_THRESHOLD_MS));
                    }
                }
            } else {
                // Neutral detected: reset debounce timer and return to idle
                if (last_emotion != EmotionType::NEUTRAL) {
                    ESP_LOGI(TAG, "[DEBOUNCE] Transitioned to safe state (%s). Debounce reset.", emotion_name);
                    printf("DEBOUNCE:IDLE\n");
                }
                last_emotion = EmotionType::NEUTRAL;
                negative_start_time_ms = 0;
            }

            // Always free the dynamically allocated buffer
            heap_caps_free(image_data);
        }
    }

    delete emotion_model;
}


extern "C" void app_main() {
    ESP_LOGI(TAG, "Initializing Emotion Vision Assistant (Dual-Core Async Standalone Camera mode)...");

    // Create FreeRTOS Queue for pointers
    xImageQueue = xQueueCreate(2, sizeof(uint8_t *));
    if (xImageQueue == nullptr) {
        ESP_LOGE(TAG, "Failed to create FreeRTOS Queue! System halt.");
        return;
    }

    // Spawn Core 0 Task: High Priority Camera Capture & Preprocessor
    xTaskCreatePinnedToCore(
        vCameraCaptureTask,
        "CameraTask",
        4096 * 2,
        nullptr,
        6, // Slightly higher priority than inference
        nullptr,
        0 // Pinned to Core 0
    );

    // Spawn Core 1 Task: AI Inference & Decision Anti-Jitter Task
    xTaskCreatePinnedToCore(
        vEmotionInferenceTask,
        "EmotionTask",
        4096 * 8, // Large stack for deep neural network inference
        nullptr,
        5,
        nullptr,
        1 // Pinned to Core 1
    );

    ESP_LOGI(TAG, "Dual-core scheduling active. Core 0: Camera Capture, Core 1: AI Inference.");
}
