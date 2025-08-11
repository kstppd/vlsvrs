MODULE vlsvrs
    use iso_c_binding, only : c_char, c_size_t, c_ptr, c_int
    IMPLICIT NONE

type, bind(C) :: Grid32
    integer(c_size_t) :: nx, ny, nz, nc
    real(8)           :: xmin, ymin, zmin, xmax, ymax, zmax
    type(c_ptr)       :: data
end type Grid32


type, bind(C) :: Grid64
    integer(c_size_t) :: nx, ny, nz, nc
    real(8)           :: xmin, ymin, zmin, xmax, ymax, zmax
    type(c_ptr)       :: data
end type Grid64

interface
    type(Grid32) function read_var_32(filename, varname, op) bind(C, name = "read_var_32")
        import :: Grid32, c_char, c_int
        character(len=1, kind=C_char), dimension(*), intent(in) :: filename, varname
        integer(c_int), value, intent(in) :: op
    end function read_var_32
        
    type(Grid64) function read_var_64(filename, varname, op) bind(C, name = "read_var_64")
        import :: Grid64, c_char, c_int
        character(len=1, kind=C_char), dimension(*), intent(in) :: filename, varname
        integer(c_int), value, intent(in) :: op
    end function read_var_64

    
    type(Grid32) function read_vdf_32(filename, population, cid) bind(C, name = "read_vdf_32")
        import :: Grid32, c_char, c_size_t
        character(len=1, kind=C_char), dimension(*), intent(in) :: filename, population
        integer(c_size_t), value, intent(in) :: cid
    end function read_vdf_32
        
    type(Grid64) function read_vdf_64(filename, population, cid) bind(C, name = "read_vdf_64")
        import :: Grid64, c_char, c_size_t
        character(len=1, kind=C_char), dimension(*), intent(in) :: filename, population
        integer(c_size_t), value, intent(in) :: cid
    end function read_vdf_64
end interface 

END MODULE vlsvrs

