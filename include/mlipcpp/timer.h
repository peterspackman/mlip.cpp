#pragma once

#include <chrono>
#include <map>
#include <mutex>
#include <string>

namespace mlipcpp {

/**
 * @brief Categories for timing different operations
 */
enum class TimerCategory {
  // High-level phases
  Total,      // End-to-end inference
  WeightLoad, // GGUF file loading

  // Batch preparation (CPU)
  NeighborList,  // Building neighbor lists
  BatchPrep,     // NEF format conversion, tensor allocation
  AttentionMask, // Computing attention masks (CPU)

  // Graph operations (CPU)
  GraphBuild, // Building computation graph metadata
  GraphAlloc, // Scheduler allocating buffers

  // Execution (Mixed CPU/GPU)
  Compute, // Actual graph computation (scheduler decides CPU/GPU)

  // Data movement
  ResultCopy, // Copying results from backend buffers
};

/**
 * @brief High-precision timer for performance profiling
 *
 * Thread-safe timer that tracks time spent in different categories.
 * Use ScopedTimer for automatic RAII-based timing.
 *
 * Usage:
 * @code
 * Timer::instance().reset();
 * {
 *     ScopedTimer t(TimerCategory::Compute);
 *     // Computation work here
 * } // Automatically recorded
 * Timer::instance().print_summary();
 * @endcode
 */
class Timer {
public:
  /**
   * @brief Get the singleton instance
   */
  static Timer &instance();

  /**
   * @brief Start timing a category
   *
   * @param category The category to time
   */
  void start(TimerCategory category);

  /**
   * @brief Stop timing a category and accumulate duration
   *
   * @param category The category to stop timing
   */
  void stop(TimerCategory category);

  /**
   * @brief Get elapsed time for a category
   *
   * @param category The category to query
   * @return Elapsed time in nanoseconds
   */
  std::chrono::nanoseconds elapsed(TimerCategory category) const;

  /**
   * @brief Reset all timers
   */
  void reset();

  /**
   * @brief Print timing summary to stderr
   *
   * Shows hierarchical breakdown of time spent in each category
   */
  void print_summary() const;

  /**
   * @brief Enable/disable timing (default: enabled)
   *
   * When disabled, timing calls have minimal overhead
   */
  void set_enabled(bool enabled);

  /**
   * @brief Check if timing is enabled
   */
  bool is_enabled() const { return enabled_; }

private:
  Timer() = default;
  ~Timer() = default;

  // Non-copyable, non-movable
  Timer(const Timer &) = delete;
  Timer &operator=(const Timer &) = delete;
  Timer(Timer &&) = delete;
  Timer &operator=(Timer &&) = delete;

  // Convert category to string for display
  static const char *category_to_string(TimerCategory category);

  // Thread-safe storage
  mutable std::mutex mutex_;
  bool enabled_ = true;

  // Accumulated durations per category
  std::map<TimerCategory, std::chrono::nanoseconds> accumulated_;

  // Start times for active timings
  std::map<TimerCategory, std::chrono::steady_clock::time_point> start_times_;
};

/**
 * @brief RAII timer for automatic scoped timing
 *
 * Automatically starts timing on construction and stops on destruction.
 *
 * Usage:
 * @code
 * {
 *     ScopedTimer timer(TimerCategory::Compute);
 *     // Computation work here
 * } // Automatically recorded
 * @endcode
 */
class ScopedTimer {
public:
  /**
   * @brief Start timing a category
   *
   * @param category The category to time
   */
  explicit ScopedTimer(TimerCategory category);

  /**
   * @brief Stop timing and record duration
   */
  ~ScopedTimer();

  // Non-copyable, non-movable
  ScopedTimer(const ScopedTimer &) = delete;
  ScopedTimer &operator=(const ScopedTimer &) = delete;
  ScopedTimer(ScopedTimer &&) = delete;
  ScopedTimer &operator=(ScopedTimer &&) = delete;

private:
  TimerCategory category_;
};

} // namespace mlipcpp
