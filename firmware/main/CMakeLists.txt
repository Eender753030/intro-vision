idf_component_register(SRCS "main.cpp"
                            "model/emotion_model.cpp"
                            "preprocess/image_preprocess.cpp"
                    INCLUDE_DIRS "." "model" "preprocess" "test"
                    PRIV_REQUIRES esp_timer who_detect human_face_detect espressif__esp-dl)