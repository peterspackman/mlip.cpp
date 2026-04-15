#include "mlipcpp/model.h"
#include "core/gguf_loader.h"
#include "pet/pet.h"
#include "runtime/graph_model.h"
#include <stdexcept>

namespace mlipcpp {

std::unique_ptr<Model> load_model(const std::string &path) {
  // Load metadata to determine model type
  GGUFLoader loader(path);
  std::string arch = loader.get_string("general.architecture", "");

  if (arch == "pet") {
    auto model = std::make_unique<pet::PETModel>(pet::PETHypers{});
    if (!model->load_from_gguf(path)) {
      throw std::runtime_error("Failed to load PET model");
    }
    return model;
  } else if (arch == "pet-graph") {
    auto model = std::make_unique<runtime::GraphModel>();
    if (!model->load_from_gguf(path)) {
      throw std::runtime_error("Failed to load graph model");
    }
    return model;
  }

  throw std::runtime_error("Unsupported model architecture: " + arch);
}

} // namespace mlipcpp
