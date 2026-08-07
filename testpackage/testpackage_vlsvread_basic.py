import analysator as pt
import numpy as np
import vlsvrs
import os
import hashlib
import importlib

# This testpackage is basic hash comparison test for the file included in testpackage folder
# a more thorough comparison is done in Analysator's repo at https://github.com/fmihpc/analysator
datalocation = "./testpackage/"
files = ["bulk_hermite_compressed.0000001.vlsv"]


class Tester:
    def __init__(self, filename=None):
        self.filename = filename
        self.vlsvobj = None
        self.hashes_dict_rust = {}
        self.hashes_dict_python = {}

    def changeFile(self, filename):
        self.filename = filename

    def load(self, backend=None):
        if not backend or backend.lower() == "rust":
            self.vlsvobj_rust = vlsvrs.VlsvFile(self.filename)
        if not backend or backend.lower() == "python":
            self.vlsvobj_python = pt.vlsvfile.VlsvReader(self.filename)

    def setHashTarget(self, backend):
        if backend == "rust":
            self.vlsvobj = self.vlsvobj_rust
        elif backend == "python":
            self.vlsvobj = self.vlsvobj_python
        else:
            print("None set, give valid backend")

    def hash(
        self,
        func,
        args,
        op=None,
        opargs=None,
        both=False,
        loop=False,
        flatten=False,
        sort=False,
        argkey_name=None,
        novlsv=False,
    ):

        def update(vlsvobj, op, opargs, args, hashdict, loop=False):
            # If we want to repeat same function func with different arguments
            if loop:
                for arg in args:
                    update(vlsvobj, op, opargs, arg, hashdict)
                return 0
            if argkey_name:
                argkey = str(argkey_name + "_NOTARG")
            else:
                argkey = str(args)

            opsname = "_" + str(op) + "_" + str(opargs)
            # Get the method of the vlsvobj that matches the given func str
            if not novlsv:
                t = getattr(vlsvobj, func)
            else:
                t = func
            # Handle arguments and call the function with the given args to get return value
            if type(args) is dict:
                retval = t(**args)
            elif type(args) is list:
                retval = t(*args)
            else:
                raise IOError(f"Wrong args type: {type(args)} {args}")
            # If we want to do operations on the retval for example reshaping, type chaning or sorting
            if op and opargs:
                # Make into list for handling
                if type(op) is not list:
                    op = [op]
                    opargs = [opargs]

                for i, f in enumerate(op):
                    try:
                        fun = getattr(retval, f)
                    except AttributeError:
                        try:
                            # if given function is not method of retval we make retval the argument of function
                            if "." in f:
                                funcl = f.split(".")
                                # in case it is inside a module like numpy we need to get instance of the module
                                funcl[0] = importlib.import_module(funcl[0])

                                fun = getattr(funcl[0], funcl[1])
                            else:
                                fun = f
                            opargs[i] = [retval]
                        except AttributeError as e:
                            raise AttributeError(
                                f"Did not find func {func} to operate with: {e}"
                            )

                    retval = fun(*opargs[i])

            # save hash of the retval as array
            retval = np.array(retval)
            if sort:
                retval.sort()
            funname = func.__name__ if callable(func) else func
            if self.filename not in hashdict.keys():
                hashdict[self.filename] = {}
            if func not in hashdict[self.filename]:
                hashdict[self.filename][funname] = {}
            bytedata = retval.tobytes()
            if not flatten:
                bytedata += np.array(retval.shape).tobytes()

            hashdict[self.filename][funname][argkey] = [
                hashlib.sha256(bytedata).hexdigest(),
                opsname,
            ]

        if not both:
            if self.vlsvobj == self.vlsvobj_python:
                hashdict = self.hashes_dict_python
            elif self.vlsvobj == self.vlsvobj_rust:
                hashdict = self.hashes_dict_rust
            update(self.vlsvobj, op, opargs, args, hashdict, loop)
        else:
            update(self.vlsvobj_rust, op, opargs, args, self.hashes_dict_rust, loop)
            update(self.vlsvobj_python, op, opargs, args, self.hashes_dict_python, loop)

    def compare(self, funcpy, argspy, funcrust, argsrust):
        try:
            py = getattr(self.vlsvobj_python, funcpy)
            retval_py = py(**argspy)

            rust = getattr(self.vlsvobj_rust, funcrust)
            retval_rust = rust(**argsrust)

        except Exception as e:
            raise e

        if type(retval_py) is dict and type(retval_rust) is dict:
            print(
                "Checking dictionaries between vlsvrs and python from function call",
                "\n     (python):",
                str(funcpy),
                "\n     (rust):",
                str(funcrust),
                "\nThis may take a moment!",
            )
            stack = list(retval_rust.keys())
            if len(retval_py) != len(retval_rust):
                if len(list(retval_py.keys())) != 0:
                    raise SystemError(
                        "one or both of the dictionaries returned by the readers are empty"
                    )
                raise SystemError(
                    "Legnth of the dictionaries returned from vlsvrs and python do not match."
                )

            for key in retval_py.keys():
                if retval_rust[key] == retval_py[key]:
                    stack.remove(key)  # maybe a some ohter way to remove it is faster?
                else:
                    raise SystemError(
                        "returned dictionary values between vlsvreader and vlsvrs do not match"
                    )

            if len(stack) != 0:
                raise KeyError(
                    "returned dictionry from vlsvrs contains keys not present in dictonary returned by python."
                )

            # Make a hash of the returned value if they are the same, this is in case both vlsvrs and analysator read_velocity_cells changes to differ from reference
            self.hash(
                list,
                [retval_py.items()],
                novlsv=True,
                both=True,
                flatten=True,
                argkey_name="read_velocity_cells",
            )
            return True
        else:
            raise NotImplementedError

    def compareReaders(self, variable_map=None):
        print("comparing hashes between vlsvrs and vlsvreader")

        # function calls may not match, can be used to map from rust vlsvrs calls to py calls
        key_map_rust_to_py = {
            "read_variable_raw": "read_variable",
            "read_variable": "read_variable",
            "list": "list",
        }
        for file in self.hashes_dict_rust.keys():
            print(f"------{file}------")
            for key in self.hashes_dict_rust[file].keys():
                if key in key_map_rust_to_py:
                    py_key = key_map_rust_to_py[key]
                else:
                    py_key = key
                py_dict = self.hashes_dict_python[file][py_key]
                rust_dict = self.hashes_dict_rust[file][key]
                for argcall in rust_dict.keys():
                    py_argcall = argcall
                    if variable_map and argcall in variable_map:
                        py_argcall = variable_map[argcall]
                    if rust_dict[argcall][0] != py_dict[py_argcall][0]:
                        print(rust_dict[argcall][0], py_dict[py_argcall][0])
                        raise SystemError(f"Hashes do not match for call {argcall}!")
                    else:
                        continue
        return True


ciTester = Tester()
for file in files:
    # Load data
    filename = os.path.join(datalocation, file)

    ciTester.changeFile(filename)
    ciTester.load()

    # Test compare
    cid = 1
    ciTester.compare(
        "read_velocity_cells",
        {"cellid": cid, "pop": "proton"},
        "read_vdf_sparse",
        {"cid": cid, "pop": "proton"},
    )

    # Make hash rust
    ciTester.setHashTarget("rust")
    variables_to_test = [
        "CellID",
        "vg_rhom",
        "vg_v",
        "vg_rhoq",
        "proton/vg_rho",
        "proton/vg_v",
    ]  # fg_variable read issue with read_variable
    variables_to_test_nonraw = ["fg_b", "fg_v"]
    pylist = ciTester.vlsvobj_python.get_variables()
    rustlist = ciTester.vlsvobj_rust.list_variables()
    variables = [
        [var] for var in variables_to_test if (var in pylist and var in rustlist)
    ]
    nonraw_vars = [
        [var, 0]
        for var in variables_to_test_nonraw
        if (var in pylist and var in rustlist)
    ]
    nonraw_to_raw_map = {str(var): str([var[0]]) for var in nonraw_vars}
    ciTester.hash("read_variable_raw", variables, loop=True, flatten=True)
    ciTester.hash("read_variable", nonraw_vars, loop=True, flatten=True)

    # Make hash python
    ciTester.setHashTarget("python")
    variables.extend(
        [[var[0]] for var in nonraw_vars]
    )  # prob some prettier way than looping through it all but it's not a big list
    ciTester.hash("read_variable", variables, loop=True, flatten=True)
    ciTester.compareReaders(nonraw_to_raw_map)
