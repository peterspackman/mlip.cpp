// mlip.cpp as a metatomic plugin: any engine that loads metatomic plugins
// (mta_load_plugin) gets a graph-format GGUF model's energy, forces and virial.
//
// The engine hands over each system with a full pair list at the model cutoff,
// and those pairs are the model's neighbor list (the adaptive cutoff prunes
// them as usual). With selected atoms the energy is theirs, and the gradients
// are with respect to every atom, as a domain decomposition with ghost atoms
// needs.
//
// mlip.cpp computes in float32, so the model asks for float32 systems. The
// systems are read on the host; ggml runs the model on whichever backend it
// was asked for (the "backend" option), moving the data itself.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <metatomic.hpp>

#include "core/backend.h"
#include "core/gguf_loader.h"
#include "mlipcpp/neighbor_list.h"
#include "mlipcpp/system.h"
#include "runtime/graph_model.h"

namespace {

// --- options -----------------------------------------------------------------

struct MlipOptions {
  mlipcpp::BackendPreference backend = mlipcpp::BackendPreference::Auto;
  int threads = 0; // 0: ggml's default
};

MlipOptions parse_options(const std::map<std::string, std::string> &options) {
  MlipOptions o;
  for (const auto &[key, value] : options) {
    if (key == "backend") {
      try {
        o.backend = mlipcpp::parse_backend_preference(value);
      } catch (const std::exception &e) {
        throw metatomic::Error(std::string("mlip.cpp option 'backend': ") + e.what());
      }
    } else if (key == "threads") {
      std::size_t used = 0;
      try {
        o.threads = std::stoi(value, &used);
      } catch (const std::exception &) {
        used = 0;
      }
      if (used == 0 || used != value.size() || o.threads < 1)
        throw metatomic::Error("mlip.cpp option 'threads' must be a positive integer, got '" + value + "'");
    } else {
      throw metatomic::Error("unknown mlip.cpp option '" + key + "' (known: backend, threads)");
    }
  }
  return o;
}

// --- reading the engine's arrays -----------------------------------------------

// A DLPack tensor on the host, copied out in row-major order whatever its
// strides, as T. Floats may be float32 or float64; integers int32.
template <typename T>
std::vector<T> host_values(const metatomic::DLPackTensor &tensor, const char *what) {
  const DLTensor &t = tensor->dl_tensor;
  if (t.device.device_type != kDLCPU)
    throw metatomic::Error(std::string("mlip.cpp: the system's ") + what + " must be on the CPU");
  const bool f32 = t.dtype.code == kDLFloat && t.dtype.bits == 32;
  const bool f64 = t.dtype.code == kDLFloat && t.dtype.bits == 64;
  const bool i32 = t.dtype.code == kDLInt && t.dtype.bits == 32;
  const bool b8 = (t.dtype.code == kDLBool || t.dtype.code == kDLUInt) && t.dtype.bits == 8;
  bool ok = false;
  if constexpr (std::is_floating_point_v<T>) ok = f32 || f64;
  else if constexpr (std::is_same_v<T, int32_t>) ok = i32;
  else if constexpr (std::is_same_v<T, bool>) ok = b8;
  if (!ok || t.dtype.lanes != 1)
    throw metatomic::Error(std::string("mlip.cpp: unexpected dtype for the system's ") + what);

  std::size_t n = 1;
  for (int d = 0; d < t.ndim; ++d) n *= std::size_t(t.shape[d]);
  std::vector<T> out(n);
  const auto *base = reinterpret_cast<const char *>(t.data) + t.byte_offset;
  const int64_t size = t.dtype.bits / 8;
  for (std::size_t flat = 0; flat < n; ++flat) {
    // Row-major index -> element offset through the strides (none: contiguous).
    int64_t offset = 0, rest = int64_t(flat), stride = 1;
    for (int d = t.ndim - 1; d >= 0; --d) {
      const int64_t i = rest % t.shape[d];
      rest /= t.shape[d];
      offset += i * (t.strides ? t.strides[d] : stride);
      stride *= t.shape[d];
    }
    const char *p = base + offset * size;
    if constexpr (std::is_same_v<T, bool>)
      out[flat] = *reinterpret_cast<const uint8_t *>(p) != 0;
    else if constexpr (std::is_floating_point_v<T>)
      out[flat] = f32 ? T(*reinterpret_cast<const float *>(p)) : T(*reinterpret_cast<const double *>(p));
    else
      out[flat] = *reinterpret_cast<const T *>(p);
  }
  return out;
}

// --- building outputs --------------------------------------------------------

bool has_names(const metatensor::Labels &labels, const std::vector<std::string> &names) {
  const auto got = labels.names();
  return std::equal(got.begin(), got.end(), names.begin(), names.end(),
                    [](const char *a, const std::string &b) { return b == a; });
}

metatensor::Labels labels(const std::vector<std::string> &names, const std::vector<int32_t> &values) {
  return metatensor::Labels(names, values.data(), values.size() / names.size(), metatensor::assume_unique{});
}

std::unique_ptr<metatensor::SimpleDataArray<float>> array(std::vector<uintptr_t> shape, std::vector<float> data) {
  return std::make_unique<metatensor::SimpleDataArray<float>>(std::move(shape), std::move(data));
}

metatensor::TensorMap single_block_map(metatensor::TensorBlock block) {
  std::vector<metatensor::TensorBlock> blocks;
  blocks.emplace_back(std::move(block));
  return metatensor::TensorMap(labels({"_"}, {0}), std::move(blocks));
}

// --- the model ---------------------------------------------------------------

// One system's answer, in the engine's atom order.
struct SystemResult {
  std::vector<int32_t> selected; // the atoms whose energy counts, in selection order
  int n_atoms = 0;
  float energy = 0;
  std::vector<float> atomic_energies; // [selected]
  std::vector<float> gradient;        // dE/dr [n_atoms * 3]
  std::array<float, 9> strain{};      // dE/d(eps), row-major
  bool has_strain = false;
};

class MlipModel final : public metatomic::BaseModel {
public:
  MlipModel(std::string path, const MlipOptions &options) : path_(std::move(path)) {
    model_.set_backend_preference(options.backend);
    model_.load_from_gguf(path_);
    if (options.threads > 0) model_.set_cpu_threads(options.threads);
    pairs_ = metatomic::PairListOptions::builder()
                 .cutoff(model_.cutoff())
                 .full_list(true)
                 .strict(false)
                 .add_requestor("mlip.cpp")
                 .build();
  }

  metatomic::ModelCapabilities capabilities() const override {
    auto energy = metatomic::Quantity::builder()
                      .name("energy")
                      .unit("eV")
                      .sample_kind(metatomic::SampleKind::System)
                      .add_gradient(metatomic::Gradients::Positions)
                      .add_gradient(metatomic::Gradients::Strain)
                      .build();
    auto atomic_energy =
        metatomic::Quantity::builder().name("energy").unit("eV").sample_kind(metatomic::SampleKind::Atom).build();
    return metatomic::ModelCapabilities::builder()
        .atomic_types(model_.atomic_types())
        .interaction_range(model_.interaction_range())
        .length_unit("angstrom")
        .supported_devices({metatomic::ModelCapabilities::Device::CPU})
        .dtype(metatomic::ModelCapabilities::DType::Float32)
        .add_output(energy)
        .add_output(atomic_energy)
        .build();
  }

  metatomic::ModelMetadata metadata() const override {
    return metatomic::ModelMetadata::builder()
        .name(path_)
        .description("PET interatomic potential exported to a GGUF graph, evaluated by mlip.cpp (ggml)")
        .add_reference("architecture", "https://arxiv.org/abs/2305.19302")
        .add_reference("implementation", "https://github.com/peterspackman/mlip.cpp")
        .build();
  }

  std::vector<metatomic::PairListOptions> requested_pair_lists() const override { return {pairs_}; }

  std::vector<metatomic::Quantity> requested_inputs() const override { return {}; }

  std::vector<metatensor::TensorMap> execute_inner(const std::vector<metatomic::System> &systems,
                                                   std::optional<metatensor::Labels> selected_atoms,
                                                   const std::vector<metatomic::Quantity> &requested_outputs) override {
    bool gradients = false;
    for (const auto &output : requested_outputs) {
      if (output.name() != "energy") throw metatomic::Error("mlip.cpp: unsupported output '" + output.name() + "'");
      const auto kind = output.sample_kind();
      if (kind != metatomic::SampleKind::System && kind != metatomic::SampleKind::Atom)
        throw metatomic::Error("mlip.cpp: energy must have system or atom samples");
      if (!output.gradients().empty()) {
        if (kind != metatomic::SampleKind::System)
          throw metatomic::Error("mlip.cpp: gradients are only available for the per-system energy");
        gradients = true;
      }
    }

    // Each system's selected atoms, in the order the selection lists them.
    std::vector<std::vector<int32_t>> selection(systems.size());
    if (selected_atoms) {
      if (!has_names(*selected_atoms, {"system", "atom"}))
        throw metatomic::Error("mlip.cpp: selected_atoms must have the dimensions 'system' and 'atom'");
      auto values = selected_atoms->values();
      for (std::size_t r = 0; r < selected_atoms->count(); ++r) {
        const int32_t s = values(r, 0), a = values(r, 1);
        if (s < 0 || std::size_t(s) >= systems.size() || a < 0 || std::size_t(a) >= systems[s].size())
          throw metatomic::Error("mlip.cpp: selected_atoms refers to an atom that does not exist");
        selection[s].push_back(a);
      }
    } else {
      for (std::size_t s = 0; s < systems.size(); ++s) {
        selection[s].resize(systems[s].size());
        for (std::size_t a = 0; a < systems[s].size(); ++a) selection[s][a] = int32_t(a);
      }
    }

    std::vector<SystemResult> results;
    results.reserve(systems.size());
    for (std::size_t s = 0; s < systems.size(); ++s)
      results.push_back(evaluate(systems[s], std::move(selection[s]), gradients));

    std::vector<metatensor::TensorMap> outputs;
    for (const auto &output : requested_outputs)
      outputs.push_back(output.sample_kind() == metatomic::SampleKind::System
                            ? system_energy(results, selected_atoms.has_value(), output)
                            : atom_energy(results));
    return outputs;
  }

private:
  SystemResult evaluate(const metatomic::System &system, std::vector<int32_t> selected, bool gradients) {
    SystemResult out;
    const int n = int(system.size());
    out.n_atoms = n;
    out.selected = std::move(selected);
    if (gradients) out.gradient.assign(std::size_t(n) * 3, 0.0f);
    if (out.selected.empty()) return out; // nothing counts: zero energy, zero gradient

    const auto types = host_values<int32_t>(system.types(), "types");
    const auto positions = host_values<float>(system.positions(), "positions");
    const auto cell = host_values<double>(system.cell(), "cell");
    const auto pbc = host_values<bool>(system.pbc(), "pbc");

    std::optional<mlipcpp::Cell> mcell;
    if (pbc[0] || pbc[1] || pbc[2]) {
      float lattice[3][3];
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) lattice[i][j] = float(cell[3 * i + j]);
      mcell.emplace(lattice, pbc[0], pbc[1], pbc[2]);
    }
    mlipcpp::AtomicSystem atoms(n, positions.data(), types.data(), mcell ? &*mcell : nullptr);

    // The engine's pairs are the neighbor list. It may include pairs a little
    // beyond the cutoff (the request is not strict); those are dropped.
    auto pairs = system.pairs(pairs_);
    auto samples = pairs.samples();
    if (!has_names(samples, {"first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"}))
      throw metatomic::Error("mlip.cpp: unexpected pair list samples");
    const std::size_t P = samples.count();
    mlipcpp::NeighborList list;
    list.n_atoms = n;
    if (P > 0) {
      auto indices = samples.values();
      auto vectors = pairs.values<float>();
      const int32_t *idx = indices.data();
      const float *vec = vectors.data();
      const int64_t si = indices.strides()[0], sc = indices.strides()[1];
      const int64_t sv = vectors.strides()[0], sx = vectors.strides()[1];
      const float cutoff = model_.cutoff();
      list.centers.reserve(P), list.neighbors.reserve(P), list.cell_shifts.reserve(P);
      list.edge_vectors.reserve(P), list.distances.reserve(P);
      for (std::size_t k = 0; k < P; ++k) {
        const int32_t *row = idx + int64_t(k) * si;
        const float *v = vec + int64_t(k) * sv;
        const std::array<float, 3> r = {v[0], v[sx], v[2 * sx]};
        const float d = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
        if (d > cutoff) continue;
        list.centers.push_back(row[0]);
        list.neighbors.push_back(row[sc]);
        list.cell_shifts.push_back({row[2 * sc], row[3 * sc], row[4 * sc]});
        list.edge_vectors.push_back(r);
        list.distances.push_back(d);
      }
    }

    const bool everything = int(out.selected.size()) == n;
    mlipcpp::ModelResult r;
    try {
      r = model_.predict(atoms, list, gradients, everything ? nullptr : &out.selected);
    } catch (const metatomic::Error &) {
      throw;
    } catch (const std::exception &e) {
      throw metatomic::Error(std::string("mlip.cpp: ") + e.what());
    }

    out.energy = r.energy;
    out.atomic_energies.reserve(out.selected.size());
    for (int32_t a : out.selected) out.atomic_energies.push_back(r.atomic_energies.at(a));
    if (gradients) {
      for (std::size_t i = 0; i < out.gradient.size(); ++i) out.gradient[i] = -r.forces.at(i);
      if (r.has_stress) {
        // The virial W = dE/d(eps) = V sigma, from Voigt [xx yy zz yz xz xy].
        const auto &h = cell;
        const double volume = std::abs(h[0] * (h[4] * h[8] - h[5] * h[7]) - h[1] * (h[3] * h[8] - h[5] * h[6]) +
                                       h[2] * (h[3] * h[7] - h[4] * h[6]));
        constexpr int voigt[9] = {0, 5, 4, 5, 1, 3, 4, 3, 2};
        for (int k = 0; k < 9; ++k) out.strain[k] = float(volume * r.stress.at(voigt[k]));
      }
      out.has_strain = r.has_stress;
    }
    return out;
  }

  // One row per system; with a selection, only the systems it has atoms in.
  // A gradient's "sample" is the row.
  metatensor::TensorMap system_energy(const std::vector<SystemResult> &results, bool selection,
                                      const metatomic::Quantity &request) const {
    std::vector<int32_t> ids, rows;
    for (std::size_t s = 0; s < results.size(); ++s) {
      if (selection && results[s].selected.empty()) continue;
      ids.push_back(int32_t(s)), rows.push_back(int32_t(rows.size()));
    }
    const std::size_t R = ids.size();
    std::vector<float> energy;
    for (int32_t s : ids) energy.push_back(results[s].energy);
    const auto property = labels({"energy"}, {0});
    metatensor::TensorBlock block(array({R, 1}, std::move(energy)), labels({"system"}, ids), {}, property);

    for (auto g : request.gradients()) {
      if (g == metatomic::Gradients::Positions) {
        std::vector<float> values;
        std::vector<int32_t> samples;
        for (std::size_t r = 0; r < R; ++r) {
          const auto &res = results[ids[r]];
          values.insert(values.end(), res.gradient.begin(), res.gradient.end());
          for (int a = 0; a < res.n_atoms; ++a) samples.insert(samples.end(), {int32_t(r), ids[r], a});
        }
        const std::size_t total = values.size() / 3;
        block.add_gradient("positions", metatensor::TensorBlock(array({total, 3, 1}, std::move(values)),
                                                                labels({"sample", "system", "atom"}, samples),
                                                                {labels({"xyz"}, {0, 1, 2})}, property));
      } else {
        std::vector<float> values;
        for (std::size_t r = 0; r < R; ++r) {
          const auto &res = results[ids[r]];
          if (!res.has_strain && !res.selected.empty())
            throw metatomic::Error("mlip.cpp: strain gradients need a periodic system");
          values.insert(values.end(), res.strain.begin(), res.strain.end());
        }
        block.add_gradient("strain", metatensor::TensorBlock(
                                         array({R, 3, 3, 1}, std::move(values)), labels({"sample"}, rows),
                                         {labels({"xyz_1"}, {0, 1, 2}), labels({"xyz_2"}, {0, 1, 2})}, property));
      }
    }
    return single_block_map(std::move(block));
  }

  metatensor::TensorMap atom_energy(const std::vector<SystemResult> &results) const {
    std::vector<float> values;
    std::vector<int32_t> samples;
    for (std::size_t s = 0; s < results.size(); ++s) {
      const auto &r = results[s];
      values.insert(values.end(), r.atomic_energies.begin(), r.atomic_energies.end());
      for (int32_t a : r.selected) samples.insert(samples.end(), {int32_t(s), a});
    }
    const std::size_t rows = values.size();
    return single_block_map(metatensor::TensorBlock(array({rows, 1}, std::move(values)),
                                                    labels({"system", "atom"}, samples), {}, labels({"energy"}, {0})));
  }

  std::string path_;
  mlipcpp::runtime::GraphModel model_;
  metatomic::PairListOptions pairs_ = metatomic::PairListOptions::builder().cutoff(1.0).full_list(true).build();
};

// A graph-format GGUF file (general.architecture = "pet-graph") is ours.
// Anything else is not: nullptr, so metatomic can try another plugin. A GGUF
// in the older hand-written PET format says how to get the graph one.
std::unique_ptr<metatomic::BaseModel> load_model(const std::string &path,
                                                 const std::map<std::string, std::string> &options) {
  std::string arch;
  try {
    arch = mlipcpp::GGUFLoader(path).get_string("general.architecture", "");
  } catch (const std::exception &) {
    return nullptr;
  }
  if (arch == "pet")
    throw metatomic::Error("mlip.cpp: '" + path +
                           "' is a GGUF in the older PET format; the plugin takes graph-format "
                           "(pet-graph) models, such as those at huggingface.co/peterspackman/mlip-gguf");
  if (arch != "pet-graph") return nullptr;
  return std::make_unique<MlipModel>(path, parse_options(options));
}

} // namespace

MTA_REGISTER_CXX_PLUGIN("mlip.cpp", load_model);
