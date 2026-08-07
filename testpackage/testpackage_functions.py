import vlsvrs
import numpy as np
import analysator as pt

# For testing basic function input and output stuff, any new functions added should ideally be added here

fn = "./testpackage/bulk_hermite_compressed.0000001.vlsv"
f = vlsvrs.VlsvFile(fn)
p = pt.vlsvfile.VlsvReader(fn)

testpackage_failed = False


def test_dictionary(py_dict, rust_dict, func_name, sort=False):
    keys_diff_rust = set(list(rust_dict.keys())) - set(list(py_dict.keys()))
    keys_diff_py = (
        set(list(py_dict.keys())) - set(list(rust_dict.keys()))
        # - set([(np.int64(0), np.int64(0), np.int64(0))])
        # some weird key is currenty default initialized in the dictionary, for now just skipping it. This should be removed once fixed
    )
    if keys_diff_py or keys_diff_rust:
        print(
            f"Keys in dictionaries returned from {func_name} differ!\n\
        following keys were unique to"
        )
        if keys_diff_rust:
            print(f"rust: {keys_diff_rust}\n")
        if keys_diff_py:
            print(f"python: {keys_diff_py}")
        return 1
    for key in py_dict.keys():
        # if np.all(key == (np.int64(0), np.int64(0), np.int64(0))):
        #     # skip the weird key
        #     continue
        py_val = py_dict[key]
        rust_val = rust_dict[key]
        if type(py_val) is type(set()) and type(rust_val) is not type(set()):
            sort = False
            rust_val = set(np.array(rust_val))
        if type(rust_val) is type(set()) and type(py_val) is not type(set()):
            sort = False
            py_val = set(
                py_val
            )  # cant make set of sets so this is fine, just an identity
        if sort:
            py_val = np.sort(np.array(py_val))
            rust_val = np.sort(np.array(rust_val))

        if key not in rust_dict:
            print(f"{key} not found in dictionary returned from rust for {func_name}")
            if np.any(py_val != rust_val):
                print(
                    f"{func_name}:\n\
                    key {key} has different values!\n\
                    rust: {rust_dict[key]}\n\
                    python: {py_dict[key]}"
                )
            return 1
    return 0


def print_array_diff(rust, py, func):
    ind = np.where(rust != py)
    print(
        f"Arrays returned from {func} have different values\n\
    indices that differ:\n\
    {ind}\n\
    At these indices:\n\
    rust: {rust[ind]} \n\
    python: {py[ind]}"
    )


cids = f.read_variable_raw("CellID")
coords = p.get_cell_coordinates(cids)

#################################
#   Test get_vertex_indices()   #
#################################

# Test returns outside the mesh and all coords from cell centers
coord_list = [np.array([-10e20, -10e20, 10e20]), coords]
for input in coord_list:
    rust_vert_inds = f.get_vertex_indices(input)
    py_vert_inds = p.get_vertex_indices(input)

    if np.any(rust_vert_inds != py_vert_inds):
        print_array_diff(rust_vert_inds, py_vert_inds, "get_vertex_indices()")
        testpackage_failed = True

#######################################
#   Test build_dual_from_vertices()   #
#######################################

vertices = p.get_vertex_indices(coords)

py_duals = p.build_dual_from_vertices(vertices)
rust_duals = f.build_dual_from_vertices(np.array(vertices))
if test_dictionary(py_duals, rust_duals, "build_dual_from_vertices()"):
    testpackage_failed = True

#######################################
#   Test build_cell_neighborhoods()   #
#######################################

rust_neighborhoods = f.build_cell_neighborhoods(cids)
py_neighborhoods = p.build_cell_neighborhoods(cids)

if test_dictionary(py_neighborhoods, rust_neighborhoods, "build_cell_neighborhoods()"):
    testpackage_failed = True

#########################
#   Test get_cellid()   #
#########################

coord_list = [
    np.array([[-10e10, 0, 0], [10e10, 10, 10], [0.00000001, 10.313, 14.03941]]),
    coords,
]
for input in coord_list:
    rust_vert_inds = f.get_cellid(input)
    py_vert_inds = p.get_cellid(input)
    if np.any(rust_vert_inds != py_vert_inds):
        print_array_diff(rust_vert_inds, py_vert_inds, "get_cellid()")
        testpackage_failed = True


#######################################
#   Test get_cell_corner_vertices()   #
#######################################

rust_corner_verts = f.get_cell_corner_vertices(cids)
py_corner_verst = p.get_cell_corner_vertices(cids)
if test_dictionary(py_corner_verst, rust_corner_verts, "get_cell_corner_vertices()"):
    testpackage_failed = True


###############################
#   Test get_cell_dx_base()   #
###############################

# Note that get_cell_dx takes AMR into account, get_cell_dx_base does not.
# It is more of an internal function in rust but I thought might as well bind it to python
if np.any(f.get_cell_dx_base() != p.get_cell_dx(1)):
    print(
        f"get_cell_dx_base and get_cell_dx differ:\n\
        rust get_cell_dx_base(): {f.get_cell_dx_base()}\n\
        python get_cell_dx(): {p.get_cell_dx()}\n\
        "
    )
    testpackage_failed = True


if testpackage_failed:
    print("Failed testpackage! Check the output")
    exit(1)
else:
    print("testpackage_functions.py succeeded")
