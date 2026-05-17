cmake_minimum_required(VERSION 3.16)

set(EXTRA_COMPONENT_DIRS 
    "components/esp-who/components/who_task"
    "components/esp-who/components/who_detect"
    "components/esp-who/components/who_frame_cap"
    "components/esp-who/components/who_peripherals/who_cam"
    "components/esp-who/components/who_peripherals/who_usb"
)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(emotion_recognition_firmware)
