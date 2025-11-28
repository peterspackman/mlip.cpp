#include <mlipcpp/mlipcpp.hpp>
#include "core/log.h"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace nb::literals;

// Suppress all logging by default when loading the module
static bool logs_initialized = []() {
  mlipcpp::log::suppress_ggml_logging();
  mlipcpp::log::set_level(mlipcpp::log::Level::Off);
  return true;
}();

NB_MODULE(_mlipcpp, m) {
  (void)logs_initialized;
  m.doc() = "mlipcpp - Fast C++ Machine Learning Interatomic Potentials";

  // LogLevel enum
  nb::enum_<mlipcpp::log::Level>(m, "LogLevel", "Logging verbosity level")
      .value("Trace", mlipcpp::log::Level::Trace, "Most verbose")
      .value("Debug", mlipcpp::log::Level::Debug, "Debug messages")
      .value("Info", mlipcpp::log::Level::Info, "Informational messages")
      .value("Warn", mlipcpp::log::Level::Warn, "Warnings only")
      .value("Error", mlipcpp::log::Level::Error, "Errors only")
      .value("Off", mlipcpp::log::Level::Off, "No logging (default)");

  // Log control functions
  m.def(
      "set_log_level",
      [](mlipcpp::log::Level level) { mlipcpp::log::set_level(level); },
      "level"_a, "Set the logging verbosity level");

  m.def(
      "enable_logging",
      []() {
        mlipcpp::log::set_level(mlipcpp::log::Level::Info);
      },
      "Enable informational logging");

  // Backend enum
  nb::enum_<mlipcpp::Backend>(m, "Backend", "Compute backend selection")
      .value("Auto", mlipcpp::Backend::Auto,
             "Automatically select best available GPU, fallback to CPU")
      .value("CPU", mlipcpp::Backend::CPU, "CPU only")
      .value("CUDA", mlipcpp::Backend::CUDA, "NVIDIA CUDA GPU")
      .value("HIP", mlipcpp::Backend::HIP, "AMD HIP/ROCm GPU")
      .value("Metal", mlipcpp::Backend::Metal, "Apple Metal GPU (macOS/iOS)")
      .value("Vulkan", mlipcpp::Backend::Vulkan, "Vulkan GPU (cross-platform)")
      .value("SYCL", mlipcpp::Backend::SYCL, "Intel SYCL (oneAPI)")
      .value("CANN", mlipcpp::Backend::CANN, "Huawei Ascend NPU");

  // ModelOptions
  nb::class_<mlipcpp::ModelOptions>(m, "ModelOptions", "Model configuration options")
      .def(nb::init<>())
      .def_rw("backend", &mlipcpp::ModelOptions::backend, "Compute backend")
      .def_rw("cutoff_override", &mlipcpp::ModelOptions::cutoff_override,
              "Override model cutoff (0 = use default)");

  // PredictOptions
  nb::class_<mlipcpp::PredictOptions>(m, "PredictOptions", "Prediction options")
      .def(nb::init<>())
      .def_rw("compute_forces", &mlipcpp::PredictOptions::compute_forces,
              "Whether to compute forces (default: True)")
      .def_rw("compute_stress", &mlipcpp::PredictOptions::compute_stress,
              "Whether to compute stress tensor (default: False)")
      .def_rw("use_nc_forces", &mlipcpp::PredictOptions::use_nc_forces,
              "Use non-conservative forces from forward pass heads (faster, not energy-conserving)");

  // Result struct
  nb::class_<mlipcpp::Result>(m, "Result", "Prediction results")
      .def_ro("energy", &mlipcpp::Result::energy, "Total energy in eV")
      .def_prop_ro(
          "forces",
          [](const mlipcpp::Result &r) {
            if (r.forces.empty())
              return nb::ndarray<nb::numpy, const float>();
            size_t n_atoms = r.forces.size() / 3;
            return nb::ndarray<nb::numpy, const float>(
                r.forces.data(), {n_atoms, 3});
          },
          "Forces array [n_atoms, 3] in eV/Angstrom")
      .def_prop_ro(
          "stress",
          [](const mlipcpp::Result &r) {
            if (r.stress.empty())
              return nb::ndarray<nb::numpy, const float>();
            return nb::ndarray<nb::numpy, const float>(r.stress.data(), {6});
          },
          "Stress tensor [6] in Voigt notation (eV/Angstrom^3)")
      .def("has_forces", &mlipcpp::Result::has_forces)
      .def("has_stress", &mlipcpp::Result::has_stress);

  // Predictor class
  nb::class_<mlipcpp::Predictor>(m, "Predictor",
                                  "MLIP model for energy/force predictions")
      .def(nb::init<const std::string &, const mlipcpp::ModelOptions &>(),
           "path"_a, "options"_a = mlipcpp::ModelOptions{},
           "Load a model from a GGUF file")
      .def_prop_ro("cutoff", &mlipcpp::Predictor::cutoff,
                   "Model cutoff radius in Angstroms")
      .def_prop_ro("model_type",
                   [](const mlipcpp::Predictor &p) {
                     return std::string(p.model_type());
                   },
                   "Model type string (e.g., 'PET')")
      .def(
          "predict",
          [](mlipcpp::Predictor &self,
             nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> positions,
             nb::ndarray<const int32_t, nb::shape<-1>, nb::c_contig>
                 atomic_numbers,
             std::optional<nb::ndarray<const float, nb::shape<3, 3>, nb::c_contig>> cell,
             std::optional<std::array<bool, 3>> pbc,
             bool compute_forces, bool compute_stress) {
            size_t n_atoms = positions.shape(0);
            if (atomic_numbers.shape(0) != n_atoms) {
              throw std::runtime_error(
                  "positions and atomic_numbers must have same length");
            }

            const float *cell_ptr = nullptr;
            const bool *pbc_ptr = nullptr;
            std::array<bool, 3> pbc_arr = {true, true, true};

            if (cell.has_value()) {
              cell_ptr = cell->data();
              if (pbc.has_value()) {
                pbc_arr = *pbc;
              }
              pbc_ptr = pbc_arr.data();
            }

            return self.predict(static_cast<int32_t>(n_atoms), positions.data(),
                                atomic_numbers.data(), cell_ptr, pbc_ptr,
                                compute_forces);
          },
          "positions"_a, "atomic_numbers"_a, "cell"_a = nb::none(),
          "pbc"_a = nb::none(), "compute_forces"_a = true,
          "compute_stress"_a = false,
          R"doc(
Predict energy and forces for an atomic system.

Parameters
----------
positions : ndarray[float32, (n_atoms, 3)]
    Atomic positions in Angstroms
atomic_numbers : ndarray[int32, (n_atoms,)]
    Atomic numbers (e.g., 1 for H, 6 for C)
cell : ndarray[float32, (3, 3)], optional
    Lattice vectors as rows for periodic systems
pbc : tuple[bool, bool, bool], optional
    Periodic boundary conditions (default: all True if cell provided)
compute_forces : bool
    Whether to compute forces (default: True)
compute_stress : bool
    Whether to compute stress tensor (default: False)

Returns
-------
Result
    Object containing energy, forces, and stress
)doc")
      .def(
          "predict_with_options",
          [](mlipcpp::Predictor &self,
             nb::ndarray<const float, nb::shape<-1, 3>, nb::c_contig> positions,
             nb::ndarray<const int32_t, nb::shape<-1>, nb::c_contig>
                 atomic_numbers,
             std::optional<nb::ndarray<const float, nb::shape<3, 3>, nb::c_contig>> cell,
             std::optional<std::array<bool, 3>> pbc,
             const mlipcpp::PredictOptions &options) {
            size_t n_atoms = positions.shape(0);
            if (atomic_numbers.shape(0) != n_atoms) {
              throw std::runtime_error(
                  "positions and atomic_numbers must have same length");
            }

            const float *cell_ptr = nullptr;
            const bool *pbc_ptr = nullptr;
            std::array<bool, 3> pbc_arr = {true, true, true};

            if (cell.has_value()) {
              cell_ptr = cell->data();
              if (pbc.has_value()) {
                pbc_arr = *pbc;
              }
              pbc_ptr = pbc_arr.data();
            }

            return self.predict(static_cast<int32_t>(n_atoms), positions.data(),
                                atomic_numbers.data(), cell_ptr, pbc_ptr,
                                options);
          },
          "positions"_a, "atomic_numbers"_a, "cell"_a = nb::none(),
          "pbc"_a = nb::none(), "options"_a = mlipcpp::PredictOptions{},
          R"doc(
Predict energy and forces with full options control.

Parameters
----------
positions : ndarray[float32, (n_atoms, 3)]
    Atomic positions in Angstroms
atomic_numbers : ndarray[int32, (n_atoms,)]
    Atomic numbers (e.g., 1 for H, 6 for C)
cell : ndarray[float32, (3, 3)], optional
    Lattice vectors as rows for periodic systems
pbc : tuple[bool, bool, bool], optional
    Periodic boundary conditions (default: all True if cell provided)
options : PredictOptions
    Prediction options (compute_forces, compute_stress, use_nc_forces)

Returns
-------
Result
    Object containing energy, forces, and stress

Notes
-----
When use_nc_forces is True, forces are computed from the model's forward pass
force heads instead of as gradients of energy. This is faster but the forces
are not energy-conserving.
)doc");

  // Module-level functions
  m.def("version", &mlipcpp::version, "Get library version string");

  m.def("set_backend", &mlipcpp::set_backend, "backend"_a,
        R"doc(
Set the global backend for all models.

This affects all subsequently loaded models. Call this before loading
any models to ensure they use the desired backend.

Parameters
----------
backend : Backend
    Backend to use (e.g., Backend.Metal, Backend.CPU, Backend.CUDA)

Example
-------
>>> import mlipcpp
>>> mlipcpp.set_backend(mlipcpp.Backend.Metal)
>>> model = mlipcpp.Predictor("model.gguf")  # Uses Metal
)doc");

  m.def("get_backend_name", &mlipcpp::get_backend_name,
        "Get the name of the current backend (e.g., 'Metal', 'CPU', 'CUDA')");

  // Version attribute
  m.attr("__version__") = mlipcpp::version();
}
