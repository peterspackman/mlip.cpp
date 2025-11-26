!> @file fortran_test.f90
!> @brief Fortran API test for mlipcpp
!>
!> Demonstrates the modern Fortran interface with object-oriented syntax.

program fortran_test
    use mlipcpp
    use, intrinsic :: iso_c_binding
    implicit none

    character(len=256) :: model_path
    type(mlipcpp_model) :: model
    type(mlipcpp_options) :: opts
    integer(c_int) :: ierr
    real(c_float) :: cutoff

    ! Get command line argument
    if (command_argument_count() < 1) then
        print '(A)', 'Usage: fortran_test <model.gguf>'
        stop 1
    end if
    call get_command_argument(1, model_path)

    ! Suppress verbose logging
    call mlipcpp_suppress_logging()

    print '(A)', 'mlipcpp Fortran API test'
    print '(A,A)', 'Version: ', mlipcpp_version()
    print '(A,A)', 'Backend: ', mlipcpp_get_backend_name()
    print '(A)', ''

    ! Load model
    print '(A,A)', 'Loading model: ', trim(model_path)
    ierr = model%load(trim(model_path))
    if (ierr /= MLIPCPP_OK) then
        print '(A,A)', 'Error: ', mlipcpp_error_string(ierr)
        stop 1
    end if

    ierr = model%cutoff(cutoff)
    print '(A,F5.2,A)', 'Model cutoff: ', cutoff, ' Angstroms'
    print '(A)', ''

    ! Test 1: Non-periodic water molecule
    call test_water(model)

    ! Test 2: Periodic Si crystal
    call test_silicon(model)

    ! Clean up
    print '(A)', 'Cleaning up...'
    call model%free()

    print '(A)', 'All tests passed!'

contains

    subroutine test_water(model)
        type(mlipcpp_model), intent(inout) :: model

        ! Water molecule geometry [3, n_atoms] column-major
        real(c_float) :: positions(3, 3)
        integer(c_int) :: atomic_numbers(3)
        real(c_float) :: energy
        real(c_float) :: forces(3, 3)
        real(c_float) :: force_sum(3)
        integer(c_int) :: ierr
        integer :: i

        print '(A)', '=== Test 1: Non-periodic water molecule ==='

        ! O-H bond ~0.96 A, H-O-H angle ~104.5 degrees
        positions(:, 1) = [0.000,  0.000,  0.117]   ! O
        positions(:, 2) = [-0.756, 0.000, -0.468]   ! H
        positions(:, 3) = [0.756,  0.000, -0.468]   ! H

        atomic_numbers = [8, 1, 1]  ! O, H, H

        ! Energy only
        ierr = model%predict(positions, atomic_numbers, energy)
        if (ierr /= MLIPCPP_OK) then
            print '(A,A)', 'Error: ', mlipcpp_error_string(ierr)
            return
        end if
        print '(A,F12.6,A)', 'Energy (no forces): ', energy, ' eV'

        ! With forces
        ierr = model%predict(positions, atomic_numbers, energy, forces)
        if (ierr /= MLIPCPP_OK) then
            print '(A,A)', 'Error: ', mlipcpp_error_string(ierr)
            return
        end if
        print '(A,F12.6,A)', 'Energy: ', energy, ' eV'

        print '(A)', 'Forces (eV/A):'
        print '(A,3F12.6)', '  O:  ', forces(:, 1)
        print '(A,3F12.6)', '  H1: ', forces(:, 2)
        print '(A,3F12.6)', '  H2: ', forces(:, 3)

        ! Check force sum
        force_sum = 0.0
        do i = 1, 3
            force_sum = force_sum + forces(:, i)
        end do
        print '(A,3ES10.2,A)', 'Force sum: ', force_sum, ' (should be ~0)'
        print '(A)', ''

    end subroutine

    subroutine test_silicon(model)
        type(mlipcpp_model), intent(inout) :: model

        real(c_float), parameter :: a = 5.43  ! Lattice constant
        real(c_float) :: positions(3, 2)
        integer(c_int) :: atomic_numbers(2)
        real(c_float) :: cell(3, 3)
        logical :: pbc(3)
        real(c_float) :: energy
        real(c_float) :: forces(3, 2)
        real(c_float) :: stress(6)
        integer(c_int) :: ierr

        print '(A)', '=== Test 2: Periodic Si crystal ==='

        ! Diamond Si primitive cell
        positions(:, 1) = [0.0, 0.0, 0.0]
        positions(:, 2) = [a * 0.25, a * 0.25, a * 0.25]

        atomic_numbers = [14, 14]  ! Silicon

        ! FCC lattice vectors (columns)
        cell(:, 1) = [a * 0.5, a * 0.5, 0.0]      ! a vector
        cell(:, 2) = [0.0,     a * 0.5, a * 0.5]  ! b vector
        cell(:, 3) = [a * 0.5, 0.0,     a * 0.5]  ! c vector

        pbc = [.true., .true., .true.]

        ! Predict with forces and stress
        ierr = model%predict_periodic(positions, atomic_numbers, cell, pbc, &
                                       energy, forces, stress)
        if (ierr /= MLIPCPP_OK) then
            print '(A,A)', 'Error: ', mlipcpp_error_string(ierr)
            return
        end if

        print '(A,F12.6,A)', 'Energy: ', energy, ' eV'
        print '(A,F12.6,A)', 'Energy per atom: ', energy / 2.0, ' eV/atom'

        print '(A)', 'Forces (eV/A):'
        print '(A,3F12.6)', '  Si1: ', forces(:, 1)
        print '(A,3F12.6)', '  Si2: ', forces(:, 2)

        print '(A,6F8.4,A)', 'Stress (Voigt): ', stress, ' eV/A^3'
        print '(A)', ''

    end subroutine

end program fortran_test
