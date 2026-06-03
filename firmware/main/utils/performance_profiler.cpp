#include "performance_profiler.hpp"

#include <cstdint>

#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_log.h"


static const char TAG[] = "PROFILER";
static int64_t start_time_us = 0;


namespace profiler {

void start_timer() {
    start_time_us = esp_timer_get_time();
}


float stop_timer_ms() {
    const int64_t stop_time_us = esp_timer_get_time();
    return static_cast<float>(stop_time_us - start_time_us) / 1000.0F;
}


void log_heap_status() {
    // Get free size for MALLOC_CAP_INTERNAL (Internal SRAM)
    const size_t free_sram = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    const size_t min_sram = heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    
    // Get free size for MALLOC_CAP_SPIRAM (External PSRAM)
    const size_t free_psram = heap_caps_get_free_size(MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM);
    
    ESP_LOGI(TAG, "Heap Status Check:");
    ESP_LOGI(TAG, "  - Internal SRAM Free: %lu Bytes (Min Ever: %lu Bytes)", static_cast<unsigned long>(free_sram), static_cast<unsigned long>(min_sram));
    ESP_LOGI(TAG, "  - External PSRAM Free: %lu Bytes", static_cast<unsigned long>(free_psram));
}

} // namespace profiler
