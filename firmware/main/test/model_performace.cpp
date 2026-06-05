#include <cstring>

#include "esp_log.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "emotion_model.hpp"
#include "utils/performance_profiler.hpp"

static const char TAG[] = "MODEL_PERF";

extern "C" void app_main() {
    ESP_LOGI(TAG, "Initializing model performance evaluation tool...");
    
    // Wait for the OS to stabilize and complete basic boot routines
    vTaskDelay(pdMS_TO_TICKS(1000));
    
    ESP_LOGI(TAG, "--------------------------------------------------");
    ESP_LOGI(TAG, "[Mem Check] Memory status before Model initialization:");
    profiler::log_heap_status();
    
    ESP_LOGI(TAG, "Initializing EmotionModel...");
    const int64_t init_start = esp_timer_get_time();
    EmotionModel *emotion_model = new EmotionModel();
    const float init_duration = static_cast<float>(esp_timer_get_time() - init_start) / 1000.0f;
    ESP_LOGI(TAG, "EmotionModel successfully initialized in %.2f ms", init_duration);
    
    ESP_LOGI(TAG, "[Mem Check] Memory status after Model initialization:");
    profiler::log_heap_status();
    ESP_LOGI(TAG, "--------------------------------------------------");
    
    // Prepare a zero-filled 48x48 input image buffer in fast internal SRAM
    const int width = 48;
    const int height = 48;
    const int face_size = width * height;
    
    uint8_t *input_data = reinterpret_cast<uint8_t *>(heap_caps_malloc(face_size, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL));
    if (input_data == nullptr) {
        ESP_LOGE(TAG, "FATAL: Failed to allocate %d bytes in SRAM for input buffer!", face_size);
        delete emotion_model;
        return;
    }
    std::memset(input_data, 0, face_size);
    
    // Wrap memory block into the standard esp-dl image format
    dl::image::img_t gray_img;
    gray_img.width = width;
    gray_img.height = height;
    gray_img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_GRAY;
    gray_img.data = input_data;
    
    // 1. Perform 10 Warmup iterations
    ESP_LOGI(TAG, "Starting 2 warmup iterations to stabilize caches...");
    for (int i = 0; i < 2; i++) {
        emotion_model->inference(gray_img);
        vTaskDelay(pdMS_TO_TICKS(5)); // Give a brief pause to satisfy task scheduling
    }
    ESP_LOGI(TAG, "Warmup completed successfully.");
    
    // 2. Perform 100 benchmark iterations
    ESP_LOGI(TAG, "Starting 10 iterations performance benchmark...");
    const int64_t start_time = esp_timer_get_time();
    for (int i = 0; i < 10; i++) {
        emotion_model->inference(gray_img);
    }
    const int64_t end_time = esp_timer_get_time();
    
    // 3. Performance stats calculation
    const float total_time_ms = static_cast<float>(end_time - start_time) / 1000.0f;
    const float avg_time_ms = total_time_ms / 10.0f;
    
    ESP_LOGI(TAG, "\n"
                  "=========================================================\n"
                  "        ESP32-S3 Emotion Model Performance Report       \n"
                  "=========================================================\n"
                  " Input Size        : %d x %d (Grayscale)\n"
                  " Warmup Iterations : 2\n"
                  " Test Iterations   : 10\n"
                  " Total Elapsed Time: %.2f ms\n"
                  " Average Latency   : %.2f ms per inference\n"
                  " Est. Throughput   : %.2f FPS\n"
                  "=========================================================",
             width, height, total_time_ms, avg_time_ms, 1000.0f / avg_time_ms);
    
    ESP_LOGI(TAG, "[Mem Check] Memory status during active inference loop:");
    profiler::log_heap_status();
    
    // Cleanup allocated resources
    heap_caps_free(input_data);
    delete emotion_model;
    
    ESP_LOGI(TAG, "--------------------------------------------------");
    ESP_LOGI(TAG, "[Mem Check] Memory status after resources cleanup:");
    profiler::log_heap_status();
    ESP_LOGI(TAG, "--------------------------------------------------");
    
    ESP_LOGI(TAG, "Model performance test completed. System entering idle loop.");
    while (true) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
