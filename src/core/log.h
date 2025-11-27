/**
 * @file log.h
 * @brief Lightweight logging system with fmt integration
 *
 * Usage:
 *   log::info("Message with {} args", 42);
 *   log::warn("Warning!");
 *   log::error("Error: {}", err_msg);
 *   log::debug("Debug info");  // Only in debug builds
 *   log::trace("Trace info");  // Only in debug builds
 */
#pragma once

#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>

#ifndef __EMSCRIPTEN__
#include <fmt/core.h>
#endif

namespace mlipcpp {
namespace log {

enum class Level {
  Trace = 0,
  Debug = 1,
  Info = 2,
  Warn = 3,
  Error = 4,
  Off = 5
};

#ifdef __EMSCRIPTEN__

// No-op logging for WASM builds
inline void set_level(Level) {}
inline Level get_level() { return Level::Off; }
inline void set_level_from_env() {}

template <typename... Args>
inline void error(const char*, Args&&...) {}
template <typename... Args>
inline void warn(const char*, Args&&...) {}
template <typename... Args>
inline void info(const char*, Args&&...) {}
template <typename... Args>
inline void debug(const char*, Args&&...) {}
template <typename... Args>
inline void trace(const char*, Args&&...) {}

inline void suppress_ggml_logging() {}

#else

namespace detail {

inline Level &current_level() {
  static Level level = Level::Info;
  return level;
}

inline void write(Level level, std::string_view msg) {
  if (level < current_level())
    return;
  fmt::print(stderr, "{}\n", msg);
}

} // namespace detail

// Configuration
inline void set_level(Level level) { detail::current_level() = level; }
inline Level get_level() { return detail::current_level(); }

inline void set_level_from_env() {
  if (const char *env = std::getenv("MLIP_LOG_LEVEL")) {
    std::string s(env);
    if (s == "TRACE" || s == "trace")
      set_level(Level::Trace);
    else if (s == "DEBUG" || s == "debug")
      set_level(Level::Debug);
    else if (s == "INFO" || s == "info")
      set_level(Level::Info);
    else if (s == "WARN" || s == "warn")
      set_level(Level::Warn);
    else if (s == "ERROR" || s == "error")
      set_level(Level::Error);
    else if (s == "OFF" || s == "off")
      set_level(Level::Off);
  }
}

// Logging functions
template <typename... Args>
inline void error(fmt::format_string<Args...> format, Args &&...args) {
  detail::write(Level::Error, fmt::format(format, std::forward<Args>(args)...));
}

template <typename... Args>
inline void warn(fmt::format_string<Args...> format, Args &&...args) {
  detail::write(Level::Warn, fmt::format(format, std::forward<Args>(args)...));
}

template <typename... Args>
inline void info(fmt::format_string<Args...> format, Args &&...args) {
  detail::write(Level::Info, fmt::format(format, std::forward<Args>(args)...));
}

template <typename... Args>
inline void debug(fmt::format_string<Args...> format, Args &&...args) {
#ifndef NDEBUG
  detail::write(Level::Debug, fmt::format(format, std::forward<Args>(args)...));
#else
  (void)format;
  ((void)args, ...);
#endif
}

template <typename... Args>
inline void trace(fmt::format_string<Args...> format, Args &&...args) {
#ifndef NDEBUG
  detail::write(Level::Trace, fmt::format(format, std::forward<Args>(args)...));
#else
  (void)format;
  ((void)args, ...);
#endif
}

// Suppress verbose GGML logging - implemented in backend.cpp
void suppress_ggml_logging();

#endif // __EMSCRIPTEN__

} // namespace log
} // namespace mlipcpp
