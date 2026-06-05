#include "motor_driver.hpp"

#include <array>

#include "driver/i2c_master.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/projdefs.h"
#include "freertos/task.h"
#include "utils/emotion_type.hpp"



// Hardware Pin Definitions from docs/DRV2605L_ESP32_pins.csv
#define SCL_PIN GPIO_NUM_18
#define SDA_PIN GPIO_NUM_17

// DRV2605L I2C Constants
#define DRV2605_ADDR 0x5A    // Default 7-bit I2C Address of DRV2605L
#define I2C_PORT     I2C_NUM_0 // Use I2C Port 0 (unoccupied by Camera SCCB on Port 1)

static const char TAG[] = "HAPTICS";
static TaskHandle_t haptic_task_handle = nullptr;

// New ESP-IDF v5 I2C Driver Handles
static i2c_master_bus_handle_t bus_handle = nullptr;
static i2c_master_dev_handle_t dev_handle = nullptr;


namespace haptics {

// Helper function to write to a DRV2605L register using the new I2C Master driver
static esp_err_t drv2605_write_reg(uint8_t reg, uint8_t val) {
    if (dev_handle == nullptr) {
        return ESP_ERR_INVALID_STATE;
    }
    const std::array<uint8_t, 2> write_buf = { reg, val };
    return i2c_master_transmit(dev_handle, write_buf.data(), write_buf.size(), pdMS_TO_TICKS(50));
}


// Background task to run haptic patterns asynchronously
static void haptic_pattern_task(void *pvParameters) {
    const EmotionType emotion = static_cast<EmotionType>(reinterpret_cast<uintptr_t>(pvParameters));
    
    uint32_t duration_ms = 500; // Default active duration to let the motor run

    switch (emotion) {
        case EmotionType::SURPRISE:
            ESP_LOGI(TAG, "[ACTUATOR] >>> SURPRISE: Triggering DRV2605L Strong Triple Click (強烈 3 連擊) <<<");
            drv2605_write_reg(0x04, 1);    // Strong Click 1 (Upgraded from Effect 4 for stronger tactile feel)
            drv2605_write_reg(0x05, 0x85); // Delay 50ms
            drv2605_write_reg(0x06, 1);    // Strong Click 2
            drv2605_write_reg(0x07, 0x85); // Delay 50ms
            drv2605_write_reg(0x08, 1);    // Strong Click 3
            drv2605_write_reg(0x09, 0);    // End sequence
            drv2605_write_reg(0x0C, 1);    // GO!
            duration_ms = 400;
            break;

        case EmotionType::FEAR:
            ESP_LOGI(TAG, "[ACTUATOR] >>> FEAR: Triggering DRV2605L Triple Heartbeat (三波心跳) <<<");
            drv2605_write_reg(0x04, 1);    // Heartbeat 1: Strong Click
            drv2605_write_reg(0x05, 3);    // Heartbeat 1: Medium Click
            drv2605_write_reg(0x06, 0x9E); // Delay 300ms (0x80 + 30)
            drv2605_write_reg(0x07, 1);    // Heartbeat 2: Strong Click
            drv2605_write_reg(0x08, 3);    // Heartbeat 2: Medium Click
            drv2605_write_reg(0x09, 0x9E); // Delay 300ms
            drv2605_write_reg(0x0A, 1);    // Heartbeat 3: Strong Click
            drv2605_write_reg(0x0B, 3);    // Heartbeat 3: Medium Click
            drv2605_write_reg(0x0C, 1);    // GO!
            duration_ms = 1000;
            break;

        case EmotionType::DISGUST:
            ESP_LOGI(TAG, "[ACTUATOR] >>> DISGUST: Triggering DRV2605L Rough Buzz 60%% (連續長 7 倍粗糙不適震感) <<<");
            drv2605_write_reg(0x04, 22);   // Rough Buzz 60% - Part 1
            drv2605_write_reg(0x05, 22);   // Rough Buzz 60% - Part 2
            drv2605_write_reg(0x06, 22);   // Rough Buzz 60% - Part 3 
            drv2605_write_reg(0x07, 22);   // Rough Buzz 60% - Part 4 
            drv2605_write_reg(0x08, 22);   // Rough Buzz 60% - Part 5 
            drv2605_write_reg(0x09, 22);   // Rough Buzz 60% - Part 6 
            drv2605_write_reg(0x0A, 22);   // Rough Buzz 60% - Part 7 
            drv2605_write_reg(0x0B, 0);    // End sequence
            drv2605_write_reg(0x0C, 1);    // GO!
            duration_ms = 2000;
            break;

        case EmotionType::SADNESS:
            ESP_LOGI(TAG, "[ACTUATOR] >>> SADNESS: Triggering DRV2605L Effect 70 (Transition Ramp Down Long Smooth 1-100 to 0%%) <<<");
            drv2605_write_reg(0x04, 70);   // Effect 1: Transition Ramp Down Long Smooth 1-100 to 0%
            drv2605_write_reg(0x05, 0);    // End sequence
            drv2605_write_reg(0x0C, 1);    // GO!
            duration_ms = 1000;
            break;

        case EmotionType::ANGER:
            ESP_LOGI(TAG, "[ACTUATOR] >>> ANGER: Triggering DRV2605L Long Double Strong Buzz (持續強震動) <<<");
            drv2605_write_reg(0x04, 14);   // Strong Buzz 100% - Part 1
            drv2605_write_reg(0x05, 14);   // Strong Buzz 100% - Part 2
            drv2605_write_reg(0x06, 0);    // End sequence
            drv2605_write_reg(0x0C, 1);    // GO!
            duration_ms = 1500;
            break;
            
        case EmotionType::HAPPINESS:
            ESP_LOGI(TAG, "[ACTUATOR] >>> HAPPINESS: Triggering DRV2605L Gentle Soft Double Pulse (溫和雙脈衝) <<<");
            drv2605_write_reg(0x04, 8);    // Soft Bump 60% - warm first gentle pulse
            drv2605_write_reg(0x05, 0x85); // Delay 50ms between pulses
            drv2605_write_reg(0x06, 8);    // Soft Bump 60% - warm second gentle pulse
            drv2605_write_reg(0x07, 0);    // End sequence
            drv2605_write_reg(0x0C, 1);    // GO!
            duration_ms = 400;
            break;

        default:
            break;
    }
    
    // Hold task execution to allow the DRV2605L hardware sequence to play fully
    vTaskDelay(pdMS_TO_TICKS(duration_ms));
    
    ESP_LOGI(TAG, "[ACTUATOR] >>> Pattern Completed. Motor Idle <<<");
    haptic_task_handle = nullptr;
    vTaskDelete(nullptr);
}


void init() {
    // 1. Initialize new I2C Master Bus for DRV2605L (using driver_ng to prevent conflicts)
    i2c_master_bus_config_t bus_config = {};
    bus_config.i2c_port = I2C_PORT;
    bus_config.sda_io_num = SDA_PIN;
    bus_config.scl_io_num = SCL_PIN;
    bus_config.clk_source = I2C_CLK_SRC_DEFAULT;
    bus_config.glitch_ignore_cnt = 7;
    bus_config.flags.enable_internal_pullup = true;
    
    esp_err_t err = i2c_new_master_bus(&bus_config, &bus_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "I2C master bus initialization failed!");
        return;
    }
    
    // Add DRV2605L device config to the bus
    i2c_device_config_t dev_config = {};
    dev_config.dev_addr_length = I2C_ADDR_BIT_LEN_7;
    dev_config.device_address = DRV2605_ADDR;
    dev_config.scl_speed_hz = 100000; // 100 kHz SCL Speed
    
    err = i2c_master_bus_add_device(bus_handle, &dev_config, &dev_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to add DRV2605L device to I2C bus!");
        return;
    }
    
    ESP_LOGI(TAG, "New I2C Master Bus & device registered on SCL: GPIO %d, SDA: GPIO %d. Port: %d", SCL_PIN, SDA_PIN, I2C_PORT);

    // 2. Initialize DRV2605L Device
    // Mode register (0x01) -> 0x80 (Software Reset)
    drv2605_write_reg(0x01, 0x80);
    vTaskDelay(pdMS_TO_TICKS(10)); // Settle down

    // Mode register (0x01) -> 0x00 (Out of standby, internal trigger mode)
    drv2605_write_reg(0x01, 0x00);

    // Library Selection (0x03) -> 0x01 (ERM Library A)
    drv2605_write_reg(0x03, 0x01);

    // Feedback Control (0x16) -> 0x36 (Select ERM mode with default settings)
    drv2605_write_reg(0x16, 0x36);

    // Control3 (0x1A) -> 0x20 (Enable ERM Open-Loop Mode to support cheap coin motors without auto-calibration)
    drv2605_write_reg(0x1A, 0x20);

    ESP_LOGI(TAG, "DRV2605L Haptic Motor Driver successfully initialized!");
}


void trigger(EmotionType emotion) {
    // Non-preemptive to prevent deleting tasks executing I2C operations (which deadlocks mutexes)
    if (haptic_task_handle != nullptr) {
        ESP_LOGW(TAG, "Haptic motor is busy. Skipping trigger for emotion %d", static_cast<int>(emotion));
        return;
    }
    
    if (emotion == EmotionType::NEUTRAL) {
        return; // Silent
    }
    
    // Spawn a short-lived FreeRTOS task on Core 1 to handle exact delays
    xTaskCreatePinnedToCore(
        haptic_pattern_task,
        "haptic_task",
        2048,
        reinterpret_cast<void *>(static_cast<uintptr_t>(emotion)),
        5, // High priority to ensure exact haptic timing
        &haptic_task_handle,
        1 // Pinned to Core 1
    );
}


void stop() {
    if (haptic_task_handle != nullptr) {
        vTaskDelete(haptic_task_handle);
        haptic_task_handle = nullptr;
    }
    drv2605_write_reg(0x0C, 0);
    ESP_LOGI(TAG, "Motor stopped immediately.");
}

} // namespace haptics
