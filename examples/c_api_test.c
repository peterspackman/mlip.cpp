/**
 * @file c_api_test.c
 * @brief C API test for mlipcpp
 *
 * Demonstrates how to use the mlipcpp C API to load a model
 * and run predictions on atomic systems.
 */

#include "mlipcpp/mlipcpp.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.gguf>\n", argv[0]);
        return 1;
    }

    const char* model_path = argv[1];

    // Suppress verbose logging
    mlipcpp_suppress_logging();

    printf("mlipcpp C API test\n");
    printf("Version: %s\n", mlipcpp_version());
    printf("Backend: %s\n\n", mlipcpp_get_backend_name());

    // Create model with default options
    printf("Loading model: %s\n", model_path);
    mlipcpp_model_t model = mlipcpp_model_create(NULL);
    if (!model) {
        fprintf(stderr, "Failed to create model: %s\n", mlipcpp_get_last_error());
        return 1;
    }

    // Load weights
    mlipcpp_error_t err = mlipcpp_model_load(model, model_path);
    if (err != MLIPCPP_OK) {
        fprintf(stderr, "Failed to load model: %s\n", mlipcpp_get_last_error());
        mlipcpp_model_free(model);
        return 1;
    }

    // Get cutoff
    float cutoff;
    err = mlipcpp_model_get_cutoff(model, &cutoff);
    if (err == MLIPCPP_OK) {
        printf("Model cutoff: %.2f Angstroms\n\n", cutoff);
    }

    // Test 1: Non-periodic water molecule
    printf("=== Test 1: Non-periodic water molecule ===\n");
    {
        // Water molecule geometry (O-H bond ~0.96 A, H-O-H angle ~104.5deg)
        float positions[] = {
            0.000f,  0.000f,  0.117f,   // O
           -0.756f,  0.000f, -0.468f,   // H
            0.756f,  0.000f, -0.468f    // H
        };
        int32_t atomic_numbers[] = {8, 1, 1};  // O, H, H

        // Energy only
        mlipcpp_result_t result;
        err = mlipcpp_predict_ptr(model, 3, positions, atomic_numbers,
                                   NULL, NULL, false, &result);
        if (err != MLIPCPP_OK) {
            fprintf(stderr, "Prediction failed: %s\n", mlipcpp_get_last_error());
            mlipcpp_model_free(model);
            return 1;
        }

        float energy;
        mlipcpp_result_get_energy(result, &energy);
        printf("Energy (no forces): %.6f eV\n", energy);

        // With forces
        err = mlipcpp_predict_ptr(model, 3, positions, atomic_numbers,
                                   NULL, NULL, true, &result);
        if (err != MLIPCPP_OK) {
            fprintf(stderr, "Prediction failed: %s\n", mlipcpp_get_last_error());
            mlipcpp_model_free(model);
            return 1;
        }

        mlipcpp_result_get_energy(result, &energy);
        printf("Energy: %.6f eV\n", energy);

        bool has_forces;
        mlipcpp_result_has_forces(result, &has_forces);
        if (has_forces) {
            float forces[9];  // 3 atoms * 3 components
            mlipcpp_result_get_forces(result, forces, 3);
            printf("Forces (eV/A):\n");
            printf("  O:  [%10.6f, %10.6f, %10.6f]\n", forces[0], forces[1], forces[2]);
            printf("  H1: [%10.6f, %10.6f, %10.6f]\n", forces[3], forces[4], forces[5]);
            printf("  H2: [%10.6f, %10.6f, %10.6f]\n", forces[6], forces[7], forces[8]);
        }
        printf("\n");
    }

    // Test 2: Periodic Si crystal
    printf("=== Test 2: Periodic Si crystal ===\n");
    {
        const float a = 5.43f;  // Lattice constant

        float positions[] = {
            0.0f,        0.0f,        0.0f,
            a * 0.25f,   a * 0.25f,   a * 0.25f
        };
        int32_t atomic_numbers[] = {14, 14};  // Silicon

        // FCC lattice vectors (row-major)
        float cell[] = {
            a * 0.5f, a * 0.5f, 0.0f,
            0.0f,     a * 0.5f, a * 0.5f,
            a * 0.5f, 0.0f,     a * 0.5f
        };
        bool pbc[] = {true, true, true};

        mlipcpp_system_t system = {
            .n_atoms = 2,
            .positions = positions,
            .atomic_numbers = atomic_numbers,
            .cell = cell,
            .pbc = pbc
        };

        mlipcpp_result_t result;
        err = mlipcpp_predict(model, &system, true, &result);
        if (err != MLIPCPP_OK) {
            fprintf(stderr, "Prediction failed: %s\n", mlipcpp_get_last_error());
            mlipcpp_model_free(model);
            return 1;
        }

        float energy;
        mlipcpp_result_get_energy(result, &energy);
        printf("Energy: %.6f eV\n", energy);
        printf("Energy per atom: %.6f eV/atom\n", energy / 2.0f);

        bool has_forces;
        mlipcpp_result_has_forces(result, &has_forces);
        if (has_forces) {
            float forces[6];  // 2 atoms * 3 components
            mlipcpp_result_get_forces(result, forces, 2);
            printf("Forces (eV/A):\n");
            printf("  Si1: [%10.6f, %10.6f, %10.6f]\n", forces[0], forces[1], forces[2]);
            printf("  Si2: [%10.6f, %10.6f, %10.6f]\n", forces[3], forces[4], forces[5]);
        }

        bool has_stress;
        mlipcpp_result_has_stress(result, &has_stress);
        if (has_stress) {
            float stress[6];
            mlipcpp_result_get_stress(result, stress);
            printf("Stress (Voigt): [%.4f, %.4f, %.4f, %.4f, %.4f, %.4f] eV/A^3\n",
                   stress[0], stress[1], stress[2], stress[3], stress[4], stress[5]);
        }
        printf("\n");
    }

    // Clean up
    printf("Cleaning up...\n");
    mlipcpp_model_free(model);

    printf("All tests passed!\n");
    return 0;
}
