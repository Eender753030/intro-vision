#include <stdio.h>
#include <list>
#include "esp_log.h"
#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_heap_caps.h"
#include "test_image.h"
#include "human_face_detect.hpp"
#include "who_detect.hpp"
#include "emotion_model.hpp"
#include "driver/uart.h"
#include "driver/usb_serial_jtag.h"

static const char *TAG = "MAIN";

extern "C" void app_main() {
    ESP_LOGI(TAG, "Starting Emotion Vision Assistant (Structured Mode)...");

    // 1. Initialize Components
    HumanFaceDetect *detector = new HumanFaceDetect();
    EmotionModel *emotion_model = new EmotionModel();

    // Initialize USB_SERIAL_JTAG for data transfer
    usb_serial_jtag_driver_config_t usb_serial_jtag_config = {
        .tx_buffer_size = 1024 * 4,
        .rx_buffer_size = 1024 * 4,
    };
    usb_serial_jtag_driver_install(&usb_serial_jtag_config);

    // 2. Buffer for receiving face data (48x48)
    const int FACE_SIZE = 48 * 48;
    uint8_t *serial_buf = (uint8_t *)heap_caps_malloc(FACE_SIZE, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);

    ESP_LOGI(TAG, "System Ready. Waiting for images via Serial...");

    while (1) {
        int read_len = 0;
        
        // Handshake
        while (read_len == 0) {
            printf("WAITING_FOR_IMAGE\n");
            int len = usb_serial_jtag_read_bytes(serial_buf, 1, pdMS_TO_TICKS(1000));
            if (len > 0) read_len = 1;
        }

        // Receive Data
        while (read_len < FACE_SIZE) {
            int len = usb_serial_jtag_read_bytes(serial_buf + read_len, FACE_SIZE - read_len, pdMS_TO_TICKS(10));
            if (len > 0) read_len += len;
        }

        // 3. Wrap as Image and Infer
        dl::image::img_t gray_img;
        gray_img.width = 48;
        gray_img.height = 48;
        gray_img.pix_type = dl::image::DL_IMAGE_PIX_TYPE_GRAY;
        gray_img.data = serial_buf;

        emotion_model->inference(gray_img);
        
        // 4. Report Result
        float confidence = 0;
        const char* emotion = emotion_model->get_top_emotion(confidence);
        ESP_LOGI(TAG, ">>> Final Decision: %s (Confidence: %.1f%%) <<<", emotion, confidence);
        
        printf("DONE\n");
    }

    delete detector;
    delete emotion_model;
    heap_caps_free(serial_buf);
}
