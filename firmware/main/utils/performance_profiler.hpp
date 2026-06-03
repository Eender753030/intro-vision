#pragma once


namespace profiler {

/**
 * @brief Start the high-resolution timer.
 */
void start_timer();

/**
 * @brief Stop the timer and get the elapsed time in milliseconds.
 * 
 * @return float Elapsed time in milliseconds
 */
float stop_timer_ms();

/**
 * @brief Logs the current heap status of both internal SRAM and external PSRAM.
 *        This is essential to detect any potential memory leaks.
 */
void log_heap_status();

} // namespace profiler
