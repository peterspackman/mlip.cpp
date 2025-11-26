#include "mlipcpp/timer.h"
#include "core/log.h"
#include <algorithm>

namespace mlipcpp {

Timer &Timer::instance() {
  static Timer instance;
  return instance;
}

const char *Timer::category_to_string(TimerCategory category) {
  switch (category) {
  case TimerCategory::Total:
    return "Total";
  case TimerCategory::WeightLoad:
    return "Weight Load";
  case TimerCategory::NeighborList:
    return "Neighbor List";
  case TimerCategory::BatchPrep:
    return "Batch Prep";
  case TimerCategory::AttentionMask:
    return "Attention Mask";
  case TimerCategory::GraphBuild:
    return "Graph Build";
  case TimerCategory::GraphAlloc:
    return "Graph Alloc";
  case TimerCategory::Compute:
    return "Compute";
  case TimerCategory::ResultCopy:
    return "Result Copy";
  default:
    return "Unknown";
  }
}

void Timer::start(TimerCategory category) {
  if (!enabled_)
    return;

  std::lock_guard<std::mutex> lock(mutex_);
  start_times_[category] = std::chrono::steady_clock::now();
}

void Timer::stop(TimerCategory category) {
  if (!enabled_)
    return;

  auto now = std::chrono::steady_clock::now();
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = start_times_.find(category);
  if (it != start_times_.end()) {
    auto duration = now - it->second;
    accumulated_[category] +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(duration);
    start_times_.erase(it);
  }
}

std::chrono::nanoseconds Timer::elapsed(TimerCategory category) const {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = accumulated_.find(category);
  if (it != accumulated_.end()) {
    return it->second;
  }
  return std::chrono::nanoseconds(0);
}

void Timer::reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  accumulated_.clear();
  start_times_.clear();
}

void Timer::set_enabled(bool enabled) {
  std::lock_guard<std::mutex> lock(mutex_);
  enabled_ = enabled;
}

void Timer::print_summary() const {
  std::lock_guard<std::mutex> lock(mutex_);

  if (accumulated_.empty()) {
    return;
  }

  // Get total time for percentage calculations
  auto total_it = accumulated_.find(TimerCategory::Total);
  double total_ms = 0.0;
  if (total_it != accumulated_.end()) {
    total_ms = total_it->second.count() / 1e6;
  } else {
    // If no explicit total, sum all categories
    for (const auto &[cat, dur] : accumulated_) {
      total_ms += dur.count() / 1e6;
    }
  }

  log::info("\n=== Performance Summary ===");

  // Print total first if available
  if (total_it != accumulated_.end()) {
    log::info("Total:                {:6.1f} ms", total_ms);
  }

  // Helper to print category if it exists
  auto print_category = [&](const char *label, TimerCategory cat) {
    auto it = accumulated_.find(cat);
    if (it != accumulated_.end()) {
      double ms = it->second.count() / 1e6;
      double percent = (total_ms > 0.0) ? (ms / total_ms * 100.0) : 0.0;
      log::info("  {:<18} {:6.1f} ms ({:5.1f}%)", label, ms, percent);
    }
  };

  // CPU Operations
  log::info("\nCPU Operations:");
  print_category("Neighbor List", TimerCategory::NeighborList);
  print_category("Batch Prep", TimerCategory::BatchPrep);
  print_category("Attention Mask", TimerCategory::AttentionMask);
  print_category("Graph Build", TimerCategory::GraphBuild);
  print_category("Graph Alloc", TimerCategory::GraphAlloc);

  // Computation
  log::info("\nComputation:");
  print_category("Compute", TimerCategory::Compute);

  // I/O Operations
  log::info("\nI/O:");
  print_category("Weight Load", TimerCategory::WeightLoad);
  print_category("Result Copy", TimerCategory::ResultCopy);

  log::info("");
}

ScopedTimer::ScopedTimer(TimerCategory category) : category_(category) {
  Timer::instance().start(category_);
}

ScopedTimer::~ScopedTimer() { Timer::instance().stop(category_); }

} // namespace mlipcpp
