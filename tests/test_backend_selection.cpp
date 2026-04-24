// Verifies that the public device-selection API actually reflects which
// backend is in use, and that asking for a backend that isn't compiled in
// (or has no device) surfaces as MLIPCPP_ERROR_BACKEND rather than a
// generic IO/internal error or a silent CPU fallback.
//
// The last test needs a GGUF fixture (fetched by CI into
// build/tests/gguf/pet-auto.gguf); without it we still exercise the
// introspection helpers, which do not require a model.

#include <catch2/catch_test_macros.hpp>
#include <mlipcpp/mlipcpp.h>

#include <array>
#include <cstdlib>
#include <filesystem>
#include <string>

#ifndef MLIPCPP_TEST_MODEL_DEFAULT
#define MLIPCPP_TEST_MODEL_DEFAULT ""
#endif

namespace {

std::string model_fixture_path() {
  if (const char *env = std::getenv("MLIPCPP_TEST_MODEL")) {
    return env;
  }
  return MLIPCPP_TEST_MODEL_DEFAULT;
}

} // namespace

TEST_CASE("CPU and Auto are always available", "[backend]") {
  REQUIRE(mlipcpp_is_backend_available(MLIPCPP_BACKEND_CPU));
  REQUIRE(mlipcpp_is_backend_available(MLIPCPP_BACKEND_AUTO));
}

TEST_CASE("Selecting CPU yields a CPU backend", "[backend]") {
  mlipcpp_suppress_logging();
  mlipcpp_set_backend(MLIPCPP_BACKEND_CPU);

  REQUIRE_FALSE(mlipcpp_backend_is_gpu());

  std::string name(mlipcpp_get_backend_name());
  REQUIRE_FALSE(name.empty());
  REQUIRE(name.find("CPU") != std::string::npos);

  // Leave global state clean for the next test.
  mlipcpp_set_backend(MLIPCPP_BACKEND_AUTO);
}

TEST_CASE("Requesting an unavailable backend returns MLIPCPP_ERROR_BACKEND",
          "[backend]") {
  const auto path = model_fixture_path();
  if (path.empty() || !std::filesystem::exists(path)) {
    SKIP("Model fixture not found (set MLIPCPP_TEST_MODEL or run from CI)");
  }

  mlipcpp_suppress_logging();

  // Find any GPU backend that isn't compiled in / has no matching device.
  const std::array<mlipcpp_backend_t, 7> gpu_candidates{
      MLIPCPP_BACKEND_CUDA,   MLIPCPP_BACKEND_HIP,    MLIPCPP_BACKEND_METAL,
      MLIPCPP_BACKEND_VULKAN, MLIPCPP_BACKEND_WEBGPU, MLIPCPP_BACKEND_SYCL,
      MLIPCPP_BACKEND_CANN,
  };
  mlipcpp_backend_t unavailable = MLIPCPP_BACKEND_AUTO;
  for (auto b : gpu_candidates) {
    if (!mlipcpp_is_backend_available(b)) {
      unavailable = b;
      break;
    }
  }
  if (unavailable == MLIPCPP_BACKEND_AUTO) {
    SKIP("Every GPU backend is available on this host — nothing to probe");
  }

  mlipcpp_model_options_t opts;
  mlipcpp_model_options_default(&opts);
  opts.backend = unavailable;

  mlipcpp_model_t model = mlipcpp_model_create(&opts);
  REQUIRE(model != nullptr);

  const auto err = mlipcpp_model_load(model, path.c_str());
  CHECK(err == MLIPCPP_ERROR_BACKEND);
  // The error message should come from BackendUnavailableError, not a
  // generic IO failure — otherwise users can't distinguish "missing GPU"
  // from "bad GGUF file".
  const char *msg = mlipcpp_get_last_error();
  REQUIRE(msg != nullptr);
  CHECK(std::string(msg).find("not available") != std::string::npos);

  mlipcpp_model_free(model);
  mlipcpp_set_backend(MLIPCPP_BACKEND_AUTO);
}
