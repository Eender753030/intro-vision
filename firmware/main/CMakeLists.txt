if(TEST_MODEL_PERFORMANCE)
    message(STATUS "Intro-Vision: Building in standalone MODEL PERFORMANCE TEST mode.")
    idf_component_register(SRCS "test/model_performace.cpp"
                                "model/emotion_model.cpp"
                                "preprocess/image_preprocess.cpp"
                                "utils/performance_profiler.cpp"
                        INCLUDE_DIRS "." "model" "preprocess" "haptics"
                        PRIV_REQUIRES esp_timer espressif__esp-dl)
else()
    message(STATUS "Intro-Vision: Building in standard Emotion Recognition system mode.")
    idf_component_register(SRCS "main.cpp"
                                "model/emotion_model.cpp"
                                "preprocess/image_preprocess.cpp"
                                "utils/performance_profiler.cpp"
                                "haptics/motor_driver.cpp"
                        INCLUDE_DIRS "." "model" "preprocess" "haptics"
                        PRIV_REQUIRES esp_timer who_detect human_face_detect espressif__esp-dl)
endif()