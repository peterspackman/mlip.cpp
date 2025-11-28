// WASM/Emscripten bindings for mlipcpp
// Provides JavaScript API for machine learning interatomic potentials

#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <mlipcpp/mlipcpp.hpp>
#include <mlipcpp/io.h>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <memory>
#include <fstream>

using namespace emscripten;

// Wrapper class for easier JS interaction with atomic systems
class AtomicSystemWrapper {
public:
    AtomicSystemWrapper() = default;

    // Create from arrays
    static AtomicSystemWrapper create(
        const val& positions,      // Float64Array [x0,y0,z0, x1,y1,z1, ...]
        const val& atomicNumbers,  // Int32Array [Z0, Z1, ...]
        const val& cell,           // Float64Array [3x3] or null for non-periodic
        bool periodic
    ) {
        AtomicSystemWrapper wrapper;

        const size_t n_atoms = atomicNumbers["length"].as<size_t>();
        wrapper.n_atoms_ = n_atoms;

        // Parse positions
        wrapper.positions_.resize(n_atoms * 3);
        for (size_t i = 0; i < n_atoms * 3; ++i) {
            wrapper.positions_[i] = positions[i].as<float>();
        }

        // Parse atomic numbers
        wrapper.atomic_numbers_.resize(n_atoms);
        for (size_t i = 0; i < n_atoms; ++i) {
            wrapper.atomic_numbers_[i] = atomicNumbers[i].as<int32_t>();
        }

        // Parse cell if periodic
        wrapper.periodic_ = periodic;
        if (periodic && !cell.isNull() && !cell.isUndefined()) {
            wrapper.cell_.resize(9);
            for (int i = 0; i < 9; ++i) {
                wrapper.cell_[i] = cell[i].as<float>();
            }
            wrapper.pbc_ = {true, true, true};
        }

        return wrapper;
    }

    // Create from XYZ string
    static AtomicSystemWrapper fromXyzString(const std::string& xyz_content) {
        AtomicSystemWrapper wrapper;

        std::istringstream stream(xyz_content);
        auto system = mlipcpp::io::read_xyz(stream);

        wrapper.n_atoms_ = system.num_atoms();
        wrapper.positions_.resize(wrapper.n_atoms_ * 3);
        const float* pos = system.positions();
        for (size_t i = 0; i < wrapper.n_atoms_ * 3; ++i) {
            wrapper.positions_[i] = pos[i];
        }

        const int32_t* nums = system.atomic_numbers();
        wrapper.atomic_numbers_.assign(nums, nums + wrapper.n_atoms_);
        wrapper.periodic_ = system.is_periodic();

        if (system.is_periodic() && system.cell()) {
            wrapper.cell_.resize(9);
            const auto* cell = system.cell();
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    wrapper.cell_[i * 3 + j] = cell->matrix[i][j];
                }
            }
            wrapper.pbc_ = {true, true, true};
        }

        return wrapper;
    }

    size_t numAtoms() const { return n_atoms_; }
    bool isPeriodic() const { return periodic_; }

    val getPositions() const {
        val result = val::global("Float64Array").new_(positions_.size());
        for (size_t i = 0; i < positions_.size(); ++i) {
            result.set(i, static_cast<double>(positions_[i]));
        }
        return result;
    }

    val getAtomicNumbers() const {
        val result = val::global("Int32Array").new_(atomic_numbers_.size());
        for (size_t i = 0; i < atomic_numbers_.size(); ++i) {
            result.set(i, atomic_numbers_[i]);
        }
        return result;
    }

    val getCell() const {
        if (!periodic_ || cell_.empty()) {
            return val::null();
        }
        val result = val::global("Float64Array").new_(9);
        for (int i = 0; i < 9; ++i) {
            result.set(i, static_cast<double>(cell_[i]));
        }
        return result;
    }

    // Internal accessors for prediction
    const float* positionsPtr() const { return positions_.data(); }
    const int32_t* atomicNumbersPtr() const { return atomic_numbers_.data(); }
    const float* cellPtr() const { return periodic_ && !cell_.empty() ? cell_.data() : nullptr; }
    const bool* pbcPtr() const { return periodic_ ? pbc_.data() : nullptr; }

private:
    size_t n_atoms_ = 0;
    std::vector<float> positions_;
    std::vector<int32_t> atomic_numbers_;
    std::vector<float> cell_;
    std::array<bool, 3> pbc_ = {false, false, false};
    bool periodic_ = false;
};

// Wrapper class for Predictor - uses shared_ptr for Embind compatibility
class PredictorWrapper {
public:
    PredictorWrapper() = default;
    PredictorWrapper(std::shared_ptr<mlipcpp::Predictor> p) : predictor_(std::move(p)) {}

    // Load model from file path (Emscripten VFS)
    static PredictorWrapper load(const std::string& path) {
        return PredictorWrapper(std::make_shared<mlipcpp::Predictor>(path));
    }

    // Load model from ArrayBuffer
    static PredictorWrapper loadFromBuffer(const val& buffer) {
        // Get data from ArrayBuffer
        val uint8Array = val::global("Uint8Array").new_(buffer);
        const size_t length = uint8Array["length"].as<size_t>();

        // Copy data to C++ vector
        std::vector<uint8_t> data(length);
        for (size_t i = 0; i < length; ++i) {
            data[i] = uint8Array[i].as<uint8_t>();
        }

        // Write to temp file using C++ fstream (works with Emscripten VFS)
        const std::string temp_path = "/tmp/model.gguf";
        {
            std::ofstream ofs(temp_path, std::ios::binary);
            if (!ofs) {
                throw std::runtime_error("Failed to create temp file");
            }
            ofs.write(reinterpret_cast<const char*>(data.data()), data.size());
        }

        // Load from temp file
        auto predictor = std::make_shared<mlipcpp::Predictor>(temp_path);

        // Clean up temp file
        std::remove(temp_path.c_str());

        return PredictorWrapper(std::move(predictor));
    }

    std::string modelType() const {
        return predictor_ ? std::string(predictor_->model_type()) : "";
    }

    double cutoff() const {
        return predictor_ ? predictor_->cutoff() : 0.0;
    }

    // Predict energy only
    double predictEnergy(const AtomicSystemWrapper& system) {
        if (!predictor_) {
            throw std::runtime_error("Model not loaded");
        }

        auto result = predictor_->predict(
            system.numAtoms(),
            system.positionsPtr(),
            system.atomicNumbersPtr(),
            system.cellPtr(),
            system.pbcPtr(),
            false  // compute_forces = false
        );

        return result.energy;
    }

    // Predict energy and forces (conservative, via gradient)
    val predict(const AtomicSystemWrapper& system) {
        return predictWithOptions(system, false);
    }

    // Predict with options - useNCForces uses forward pass heads instead of gradients
    val predictWithOptions(const AtomicSystemWrapper& system, bool useNCForces) {
        if (!predictor_) {
            throw std::runtime_error("Model not loaded");
        }

        mlipcpp::PredictOptions options;
        options.compute_forces = true;
        options.use_nc_forces = useNCForces;

        auto result = predictor_->predict(
            system.numAtoms(),
            system.positionsPtr(),
            system.atomicNumbersPtr(),
            system.cellPtr(),
            system.pbcPtr(),
            options
        );

        val output = val::object();
        output.set("energy", static_cast<double>(result.energy));

        // Convert forces to Float64Array
        val forces = val::global("Float64Array").new_(result.forces.size());
        for (size_t i = 0; i < result.forces.size(); ++i) {
            forces.set(i, static_cast<double>(result.forces[i]));
        }
        output.set("forces", forces);

        // Include stress if available
        if (result.has_stress()) {
            val stress = val::global("Float64Array").new_(6);
            for (int i = 0; i < 6; ++i) {
                stress.set(i, static_cast<double>(result.stress[i]));
            }
            output.set("stress", stress);
        }

        return output;
    }

    bool isLoaded() const { return predictor_ != nullptr; }

private:
    std::shared_ptr<mlipcpp::Predictor> predictor_;
};

// Utility functions
std::string getVersion() {
    return mlipcpp::version();
}

// Emscripten bindings
EMSCRIPTEN_BINDINGS(mlipcpp) {
    // AtomicSystem wrapper
    class_<AtomicSystemWrapper>("AtomicSystem")
        .constructor<>()
        .class_function("create", &AtomicSystemWrapper::create)
        .class_function("fromXyzString", &AtomicSystemWrapper::fromXyzString)
        .function("numAtoms", &AtomicSystemWrapper::numAtoms)
        .function("isPeriodic", &AtomicSystemWrapper::isPeriodic)
        .function("getPositions", &AtomicSystemWrapper::getPositions)
        .function("getAtomicNumbers", &AtomicSystemWrapper::getAtomicNumbers)
        .function("getCell", &AtomicSystemWrapper::getCell);

    // Predictor wrapper (named Model for JS API simplicity)
    class_<PredictorWrapper>("Model")
        .constructor<>()
        .class_function("load", &PredictorWrapper::load)
        .class_function("loadFromBuffer", &PredictorWrapper::loadFromBuffer)
        .function("modelType", &PredictorWrapper::modelType)
        .function("cutoff", &PredictorWrapper::cutoff)
        .function("predictEnergy", &PredictorWrapper::predictEnergy)
        .function("predict", &PredictorWrapper::predict)
        .function("predictWithOptions", &PredictorWrapper::predictWithOptions)
        .function("isLoaded", &PredictorWrapper::isLoaded);

    // Utility functions
    function("getVersion", &getVersion);
}
