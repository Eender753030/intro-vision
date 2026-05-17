idf_component_register(SRCS "main.cpp" "emotion_model.cpp"
                    INCLUDE_DIRS "."
                    PRIV_REQUIRES esp_timer who_detect human_face_detect)
