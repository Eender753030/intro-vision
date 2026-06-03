idf_component_register(SRCS "main.cpp"
                            "model/emotion_model.cpp"
                            "preprocess/image_preprocess.cpp"
                            "utils/performance_profiler.cpp"
                            "haptics/motor_driver.cpp"
                    INCLUDE_DIRS "." "model" "preprocess" "haptics"
                    PRIV_REQUIRES esp_timer who_detect human_face_detect espressif__esp-dl)