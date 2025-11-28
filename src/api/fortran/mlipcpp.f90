!> @file mlipcpp.f90
!> @brief Modern Fortran interface for mlipcpp
!>
!> This module provides Fortran 2003+ bindings to the mlipcpp C API
!> using iso_c_binding for interoperability.
!>
!> Example usage:
!> @code
!>   use mlipcpp
!>   type(mlipcpp_model) :: model
!>   real(c_float) :: energy
!>   integer :: ierr
!>
!>   ierr = model%load("model.gguf")
!>   ierr = model%predict(positions, atomic_numbers, energy)
!>   call model%free()
!> @endcode

module mlipcpp
    use, intrinsic :: iso_c_binding
    implicit none
    private

    ! Public types
    public :: mlipcpp_model
    public :: mlipcpp_options
    public :: mlipcpp_predict_options

    ! Public constants
    public :: MLIPCPP_OK
    public :: MLIPCPP_ERROR_INVALID_HANDLE
    public :: MLIPCPP_ERROR_NULL_POINTER
    public :: MLIPCPP_ERROR_MODEL_NOT_LOADED
    public :: MLIPCPP_ERROR_OUT_OF_MEMORY
    public :: MLIPCPP_ERROR_IO
    public :: MLIPCPP_ERROR_COMPUTATION
    public :: MLIPCPP_ERROR_INVALID_PARAMETER
    public :: MLIPCPP_ERROR_BACKEND
    public :: MLIPCPP_ERROR_UNSUPPORTED
    public :: MLIPCPP_ERROR_INTERNAL

    public :: MLIPCPP_BACKEND_AUTO
    public :: MLIPCPP_BACKEND_CPU
    public :: MLIPCPP_BACKEND_CUDA
    public :: MLIPCPP_BACKEND_HIP
    public :: MLIPCPP_BACKEND_METAL
    public :: MLIPCPP_BACKEND_VULKAN
    public :: MLIPCPP_BACKEND_SYCL
    public :: MLIPCPP_BACKEND_CANN

    ! Public procedures
    public :: mlipcpp_version
    public :: mlipcpp_error_string
    public :: mlipcpp_set_backend
    public :: mlipcpp_get_backend_name
    public :: mlipcpp_suppress_logging

    ! Error codes
    integer(c_int), parameter :: MLIPCPP_OK = 0
    integer(c_int), parameter :: MLIPCPP_ERROR_INVALID_HANDLE = 1
    integer(c_int), parameter :: MLIPCPP_ERROR_NULL_POINTER = 2
    integer(c_int), parameter :: MLIPCPP_ERROR_MODEL_NOT_LOADED = 3
    integer(c_int), parameter :: MLIPCPP_ERROR_OUT_OF_MEMORY = 4
    integer(c_int), parameter :: MLIPCPP_ERROR_IO = 5
    integer(c_int), parameter :: MLIPCPP_ERROR_COMPUTATION = 6
    integer(c_int), parameter :: MLIPCPP_ERROR_INVALID_PARAMETER = 7
    integer(c_int), parameter :: MLIPCPP_ERROR_BACKEND = 8
    integer(c_int), parameter :: MLIPCPP_ERROR_UNSUPPORTED = 9
    integer(c_int), parameter :: MLIPCPP_ERROR_INTERNAL = 255

    ! Backend types
    integer(c_int), parameter :: MLIPCPP_BACKEND_AUTO = 0
    integer(c_int), parameter :: MLIPCPP_BACKEND_CPU = 1
    integer(c_int), parameter :: MLIPCPP_BACKEND_CUDA = 2
    integer(c_int), parameter :: MLIPCPP_BACKEND_HIP = 3
    integer(c_int), parameter :: MLIPCPP_BACKEND_METAL = 4
    integer(c_int), parameter :: MLIPCPP_BACKEND_VULKAN = 5
    integer(c_int), parameter :: MLIPCPP_BACKEND_SYCL = 6
    integer(c_int), parameter :: MLIPCPP_BACKEND_CANN = 7

    !> Model options
    type, bind(c) :: mlipcpp_options
        integer(c_int) :: backend = MLIPCPP_BACKEND_AUTO
        integer(c_int) :: precision = 0  ! F32
        real(c_float) :: cutoff_override = 0.0
    end type

    !> Prediction options
    type, bind(c) :: mlipcpp_predict_options
        logical(c_bool) :: compute_forces = .true._c_bool  !< Compute forces
        logical(c_bool) :: compute_stress = .false._c_bool !< Compute stress tensor
        logical(c_bool) :: use_nc_forces = .false._c_bool  !< Use non-conservative forces
    end type

    !> Main model type with object-oriented interface
    type :: mlipcpp_model
        type(c_ptr), private :: handle = c_null_ptr
        type(c_ptr), private :: result_handle = c_null_ptr
    contains
        procedure :: load => model_load
        procedure :: free => model_free
        procedure :: cutoff => model_cutoff
        procedure :: predict => model_predict
        procedure :: predict_periodic => model_predict_periodic
        procedure :: predict_with_options => model_predict_with_options
        procedure :: predict_periodic_with_options => model_predict_periodic_with_options
        procedure :: is_loaded => model_is_loaded
    end type

    ! C API interface
    interface
        function c_mlipcpp_version() bind(c, name='mlipcpp_version')
            import :: c_ptr
            type(c_ptr) :: c_mlipcpp_version
        end function

        function c_mlipcpp_error_string(error) bind(c, name='mlipcpp_error_string')
            import :: c_ptr, c_int
            integer(c_int), value :: error
            type(c_ptr) :: c_mlipcpp_error_string
        end function

        function c_mlipcpp_get_last_error() bind(c, name='mlipcpp_get_last_error')
            import :: c_ptr
            type(c_ptr) :: c_mlipcpp_get_last_error
        end function

        subroutine c_mlipcpp_set_backend(backend) bind(c, name='mlipcpp_set_backend')
            import :: c_int
            integer(c_int), value :: backend
        end subroutine

        function c_mlipcpp_get_backend_name() bind(c, name='mlipcpp_get_backend_name')
            import :: c_ptr
            type(c_ptr) :: c_mlipcpp_get_backend_name
        end function

        subroutine c_mlipcpp_suppress_logging() bind(c, name='mlipcpp_suppress_logging')
        end subroutine

        function c_mlipcpp_model_options_default(options) bind(c, name='mlipcpp_model_options_default')
            import :: c_int, mlipcpp_options
            type(mlipcpp_options), intent(out) :: options
            integer(c_int) :: c_mlipcpp_model_options_default
        end function

        function c_mlipcpp_model_create(options) bind(c, name='mlipcpp_model_create')
            import :: c_ptr, mlipcpp_options
            type(mlipcpp_options), intent(in) :: options
            type(c_ptr) :: c_mlipcpp_model_create
        end function

        function c_mlipcpp_model_create_default() bind(c, name='mlipcpp_model_create')
            import :: c_ptr
            type(c_ptr) :: c_mlipcpp_model_create_default
        end function

        function c_mlipcpp_model_load(model, path) bind(c, name='mlipcpp_model_load')
            import :: c_ptr, c_int, c_char
            type(c_ptr), value :: model
            character(kind=c_char), intent(in) :: path(*)
            integer(c_int) :: c_mlipcpp_model_load
        end function

        subroutine c_mlipcpp_model_free(model) bind(c, name='mlipcpp_model_free')
            import :: c_ptr
            type(c_ptr), value :: model
        end subroutine

        function c_mlipcpp_model_get_cutoff(model, cutoff) bind(c, name='mlipcpp_model_get_cutoff')
            import :: c_ptr, c_int, c_float
            type(c_ptr), value :: model
            real(c_float), intent(out) :: cutoff
            integer(c_int) :: c_mlipcpp_model_get_cutoff
        end function

        function c_mlipcpp_predict_ptr(model, n_atoms, positions, atomic_numbers, &
                                        cell, pbc, compute_forces, result) &
                                        bind(c, name='mlipcpp_predict_ptr')
            import :: c_ptr, c_int, c_float, c_bool
            type(c_ptr), value :: model
            integer(c_int), value :: n_atoms
            real(c_float), intent(in) :: positions(*)
            integer(c_int), intent(in) :: atomic_numbers(*)
            type(c_ptr), value :: cell
            type(c_ptr), value :: pbc
            logical(c_bool), value :: compute_forces
            type(c_ptr), intent(out) :: result
            integer(c_int) :: c_mlipcpp_predict_ptr
        end function

        function c_mlipcpp_predict_options_default(options) &
                                        bind(c, name='mlipcpp_predict_options_default')
            import :: c_int, mlipcpp_predict_options
            type(mlipcpp_predict_options), intent(out) :: options
            integer(c_int) :: c_mlipcpp_predict_options_default
        end function

        function c_mlipcpp_predict_ptr_with_options(model, n_atoms, positions, atomic_numbers, &
                                        cell, pbc, options, result) &
                                        bind(c, name='mlipcpp_predict_ptr_with_options')
            import :: c_ptr, c_int, c_float, mlipcpp_predict_options
            type(c_ptr), value :: model
            integer(c_int), value :: n_atoms
            real(c_float), intent(in) :: positions(*)
            integer(c_int), intent(in) :: atomic_numbers(*)
            type(c_ptr), value :: cell
            type(c_ptr), value :: pbc
            type(mlipcpp_predict_options), intent(in) :: options
            type(c_ptr), intent(out) :: result
            integer(c_int) :: c_mlipcpp_predict_ptr_with_options
        end function

        function c_mlipcpp_result_get_energy(result, energy) bind(c, name='mlipcpp_result_get_energy')
            import :: c_ptr, c_int, c_float
            type(c_ptr), value :: result
            real(c_float), intent(out) :: energy
            integer(c_int) :: c_mlipcpp_result_get_energy
        end function

        function c_mlipcpp_result_get_forces(result, forces, n_atoms) bind(c, name='mlipcpp_result_get_forces')
            import :: c_ptr, c_int, c_float
            type(c_ptr), value :: result
            real(c_float), intent(out) :: forces(*)
            integer(c_int), value :: n_atoms
            integer(c_int) :: c_mlipcpp_result_get_forces
        end function

        function c_mlipcpp_result_get_stress(result, stress) bind(c, name='mlipcpp_result_get_stress')
            import :: c_ptr, c_int, c_float
            type(c_ptr), value :: result
            real(c_float), intent(out) :: stress(6)
            integer(c_int) :: c_mlipcpp_result_get_stress
        end function

        function c_mlipcpp_result_has_forces(result, has_forces) bind(c, name='mlipcpp_result_has_forces')
            import :: c_ptr, c_int, c_bool
            type(c_ptr), value :: result
            logical(c_bool), intent(out) :: has_forces
            integer(c_int) :: c_mlipcpp_result_has_forces
        end function

        function c_mlipcpp_result_has_stress(result, has_stress) bind(c, name='mlipcpp_result_has_stress')
            import :: c_ptr, c_int, c_bool
            type(c_ptr), value :: result
            logical(c_bool), intent(out) :: has_stress
            integer(c_int) :: c_mlipcpp_result_has_stress
        end function
    end interface

contains

    !> Get library version string
    function mlipcpp_version() result(version)
        character(len=:), allocatable :: version
        type(c_ptr) :: cptr
        cptr = c_mlipcpp_version()
        version = c_to_f_string(cptr)
    end function

    !> Get error description string
    function mlipcpp_error_string(error) result(str)
        integer(c_int), intent(in) :: error
        character(len=:), allocatable :: str
        type(c_ptr) :: cptr
        cptr = c_mlipcpp_error_string(error)
        str = c_to_f_string(cptr)
    end function

    !> Set the global backend for all subsequently loaded models
    subroutine mlipcpp_set_backend(backend)
        integer(c_int), intent(in) :: backend
        call c_mlipcpp_set_backend(backend)
    end subroutine

    !> Get the name of the current backend
    function mlipcpp_get_backend_name() result(name)
        character(len=:), allocatable :: name
        type(c_ptr) :: cptr
        cptr = c_mlipcpp_get_backend_name()
        name = c_to_f_string(cptr)
    end function

    !> Suppress verbose logging from mlipcpp and GGML
    subroutine mlipcpp_suppress_logging()
        call c_mlipcpp_suppress_logging()
    end subroutine

    !> Load model from GGUF file
    function model_load(self, path, options) result(ierr)
        class(mlipcpp_model), intent(inout) :: self
        character(len=*), intent(in) :: path
        type(mlipcpp_options), intent(in), optional :: options
        integer(c_int) :: ierr
        type(mlipcpp_options) :: opts

        ! Free existing model if any
        if (c_associated(self%handle)) then
            call c_mlipcpp_model_free(self%handle)
        end if

        ! Create model with options
        if (present(options)) then
            opts = options
        else
            ierr = c_mlipcpp_model_options_default(opts)
        end if

        self%handle = c_mlipcpp_model_create(opts)
        if (.not. c_associated(self%handle)) then
            ierr = MLIPCPP_ERROR_OUT_OF_MEMORY
            return
        end if

        ! Load weights
        ierr = c_mlipcpp_model_load(self%handle, f_to_c_string(path))
    end function

    !> Free model resources
    subroutine model_free(self)
        class(mlipcpp_model), intent(inout) :: self
        if (c_associated(self%handle)) then
            call c_mlipcpp_model_free(self%handle)
            self%handle = c_null_ptr
        end if
    end subroutine

    !> Check if model is loaded
    function model_is_loaded(self) result(loaded)
        class(mlipcpp_model), intent(in) :: self
        logical :: loaded
        loaded = c_associated(self%handle)
    end function

    !> Get model cutoff radius
    function model_cutoff(self, cutoff) result(ierr)
        class(mlipcpp_model), intent(in) :: self
        real(c_float), intent(out) :: cutoff
        integer(c_int) :: ierr
        ierr = c_mlipcpp_model_get_cutoff(self%handle, cutoff)
    end function

    !> Predict energy and forces for non-periodic system
    !> @param positions Atomic positions [3, n_atoms] (column-major)
    !> @param atomic_numbers Atomic numbers [n_atoms]
    !> @param energy Output energy in eV
    !> @param forces Output forces [3, n_atoms] in eV/Angstrom (optional)
    function model_predict(self, positions, atomic_numbers, energy, forces) result(ierr)
        class(mlipcpp_model), intent(inout) :: self
        real(c_float), intent(in), contiguous :: positions(:,:)
        integer(c_int), intent(in), contiguous :: atomic_numbers(:)
        real(c_float), intent(out) :: energy
        real(c_float), intent(out), optional, contiguous :: forces(:,:)
        integer(c_int) :: ierr

        integer(c_int) :: n_atoms
        logical(c_bool) :: compute_forces

        n_atoms = size(atomic_numbers)
        compute_forces = present(forces)

        ! Call C API (positions are already contiguous in memory)
        ierr = c_mlipcpp_predict_ptr(self%handle, n_atoms, positions, atomic_numbers, &
                                      c_null_ptr, c_null_ptr, compute_forces, self%result_handle)
        if (ierr /= MLIPCPP_OK) return

        ! Get energy
        ierr = c_mlipcpp_result_get_energy(self%result_handle, energy)
        if (ierr /= MLIPCPP_OK) return

        ! Get forces if requested
        if (present(forces)) then
            ierr = c_mlipcpp_result_get_forces(self%result_handle, forces, n_atoms)
        end if
    end function

    !> Predict energy and forces for periodic system
    !> @param positions Atomic positions [3, n_atoms] (column-major)
    !> @param atomic_numbers Atomic numbers [n_atoms]
    !> @param cell Lattice vectors [3, 3] as columns (a, b, c)
    !> @param pbc Periodic boundary conditions [3]
    !> @param energy Output energy in eV
    !> @param forces Output forces [3, n_atoms] in eV/Angstrom (optional)
    !> @param stress Output stress [6] in Voigt notation (optional)
    function model_predict_periodic(self, positions, atomic_numbers, cell, pbc, &
                                     energy, forces, stress) result(ierr)
        class(mlipcpp_model), intent(inout) :: self
        real(c_float), intent(in), contiguous, target :: positions(:,:)
        integer(c_int), intent(in), contiguous :: atomic_numbers(:)
        real(c_float), intent(in), target :: cell(3,3)
        logical, intent(in) :: pbc(3)
        real(c_float), intent(out) :: energy
        real(c_float), intent(out), optional, contiguous :: forces(:,:)
        real(c_float), intent(out), optional :: stress(6)
        integer(c_int) :: ierr

        integer(c_int) :: n_atoms
        logical(c_bool) :: compute_forces
        logical(c_bool), target :: c_pbc(3)
        real(c_float), target :: c_cell(9)
        integer :: i, j

        n_atoms = size(atomic_numbers)
        compute_forces = present(forces)

        ! Convert cell from column-major [3,3] to row-major [9]
        ! Fortran: cell(:,1) = a, cell(:,2) = b, cell(:,3) = c
        ! C expects: cell[0:2] = a, cell[3:5] = b, cell[6:8] = c (row-major)
        do i = 1, 3
            do j = 1, 3
                c_cell((i-1)*3 + j) = cell(j, i)
            end do
        end do

        ! Convert logical to c_bool
        c_pbc = pbc

        ! Call C API
        ierr = c_mlipcpp_predict_ptr(self%handle, n_atoms, positions, atomic_numbers, &
                                      c_loc(c_cell), c_loc(c_pbc), compute_forces, self%result_handle)
        if (ierr /= MLIPCPP_OK) return

        ! Get energy
        ierr = c_mlipcpp_result_get_energy(self%result_handle, energy)
        if (ierr /= MLIPCPP_OK) return

        ! Get forces if requested
        if (present(forces)) then
            ierr = c_mlipcpp_result_get_forces(self%result_handle, forces, n_atoms)
            if (ierr /= MLIPCPP_OK) return
        end if

        ! Get stress if requested
        if (present(stress)) then
            ierr = c_mlipcpp_result_get_stress(self%result_handle, stress)
        end if
    end function

    !> Predict energy and forces for non-periodic system with options
    !> @param positions Atomic positions [3, n_atoms] (column-major)
    !> @param atomic_numbers Atomic numbers [n_atoms]
    !> @param options Prediction options (use_nc_forces, etc.)
    !> @param energy Output energy in eV
    !> @param forces Output forces [3, n_atoms] in eV/Angstrom (optional)
    function model_predict_with_options(self, positions, atomic_numbers, options, &
                                         energy, forces) result(ierr)
        class(mlipcpp_model), intent(inout) :: self
        real(c_float), intent(in), contiguous :: positions(:,:)
        integer(c_int), intent(in), contiguous :: atomic_numbers(:)
        type(mlipcpp_predict_options), intent(in) :: options
        real(c_float), intent(out) :: energy
        real(c_float), intent(out), optional, contiguous :: forces(:,:)
        integer(c_int) :: ierr

        integer(c_int) :: n_atoms
        type(mlipcpp_predict_options) :: opts

        n_atoms = size(atomic_numbers)
        opts = options
        ! Override compute_forces based on whether forces array is present
        opts%compute_forces = present(forces)

        ! Call C API
        ierr = c_mlipcpp_predict_ptr_with_options(self%handle, n_atoms, positions, atomic_numbers, &
                                      c_null_ptr, c_null_ptr, opts, self%result_handle)
        if (ierr /= MLIPCPP_OK) return

        ! Get energy
        ierr = c_mlipcpp_result_get_energy(self%result_handle, energy)
        if (ierr /= MLIPCPP_OK) return

        ! Get forces if requested
        if (present(forces)) then
            ierr = c_mlipcpp_result_get_forces(self%result_handle, forces, n_atoms)
        end if
    end function

    !> Predict energy and forces for periodic system with options
    !> @param positions Atomic positions [3, n_atoms] (column-major)
    !> @param atomic_numbers Atomic numbers [n_atoms]
    !> @param cell Lattice vectors [3, 3] as columns (a, b, c)
    !> @param pbc Periodic boundary conditions [3]
    !> @param options Prediction options (use_nc_forces, etc.)
    !> @param energy Output energy in eV
    !> @param forces Output forces [3, n_atoms] in eV/Angstrom (optional)
    !> @param stress Output stress [6] in Voigt notation (optional)
    function model_predict_periodic_with_options(self, positions, atomic_numbers, cell, pbc, &
                                     options, energy, forces, stress) result(ierr)
        class(mlipcpp_model), intent(inout) :: self
        real(c_float), intent(in), contiguous, target :: positions(:,:)
        integer(c_int), intent(in), contiguous :: atomic_numbers(:)
        real(c_float), intent(in), target :: cell(3,3)
        logical, intent(in) :: pbc(3)
        type(mlipcpp_predict_options), intent(in) :: options
        real(c_float), intent(out) :: energy
        real(c_float), intent(out), optional, contiguous :: forces(:,:)
        real(c_float), intent(out), optional :: stress(6)
        integer(c_int) :: ierr

        integer(c_int) :: n_atoms
        type(mlipcpp_predict_options) :: opts
        logical(c_bool), target :: c_pbc(3)
        real(c_float), target :: c_cell(9)
        integer :: i, j

        n_atoms = size(atomic_numbers)
        opts = options
        ! Override compute_forces based on whether forces array is present
        opts%compute_forces = present(forces)

        ! Convert cell from column-major [3,3] to row-major [9]
        ! Fortran: cell(:,1) = a, cell(:,2) = b, cell(:,3) = c
        ! C expects: cell[0:2] = a, cell[3:5] = b, cell[6:8] = c (row-major)
        do i = 1, 3
            do j = 1, 3
                c_cell((i-1)*3 + j) = cell(j, i)
            end do
        end do

        ! Convert logical to c_bool
        c_pbc = pbc

        ! Call C API
        ierr = c_mlipcpp_predict_ptr_with_options(self%handle, n_atoms, positions, atomic_numbers, &
                                      c_loc(c_cell), c_loc(c_pbc), opts, self%result_handle)
        if (ierr /= MLIPCPP_OK) return

        ! Get energy
        ierr = c_mlipcpp_result_get_energy(self%result_handle, energy)
        if (ierr /= MLIPCPP_OK) return

        ! Get forces if requested
        if (present(forces)) then
            ierr = c_mlipcpp_result_get_forces(self%result_handle, forces, n_atoms)
            if (ierr /= MLIPCPP_OK) return
        end if

        ! Get stress if requested
        if (present(stress)) then
            ierr = c_mlipcpp_result_get_stress(self%result_handle, stress)
        end if
    end function

    ! Helper: Convert C string pointer to Fortran string
    function c_to_f_string(cptr) result(fstr)
        type(c_ptr), intent(in) :: cptr
        character(len=:), allocatable :: fstr
        character(kind=c_char), pointer :: cstr(:)
        integer :: i, length

        if (.not. c_associated(cptr)) then
            fstr = ""
            return
        end if

        ! Find string length
        length = 0
        call c_f_pointer(cptr, cstr, [1000])  ! Max length
        do i = 1, 1000
            if (cstr(i) == c_null_char) exit
            length = length + 1
        end do

        allocate(character(len=length) :: fstr)
        do i = 1, length
            fstr(i:i) = cstr(i)
        end do
    end function

    ! Helper: Convert Fortran string to C string (null-terminated)
    function f_to_c_string(fstr) result(cstr)
        character(len=*), intent(in) :: fstr
        character(kind=c_char) :: cstr(len_trim(fstr) + 1)
        integer :: i, n

        n = len_trim(fstr)
        do i = 1, n
            cstr(i) = fstr(i:i)
        end do
        cstr(n + 1) = c_null_char
    end function

end module mlipcpp
