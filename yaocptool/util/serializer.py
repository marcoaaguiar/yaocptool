import concurrent.futures
import copy
import copyreg
import importlib
from multiprocessing import Lock
import os
import pickle
from typing import Union

from casadi import MX, SX, StringDeserializer, StringSerializer
import casadi


class CasadiPicklerDepickler:
    serializer = StringSerializer()
    deserializer = StringDeserializer(serializer.encode())

    def __init__(self):
        print("creating pickler/depickler")

    def dumps(self, var: Union[SX, MX]) -> str:
        self.serializer.pack(var)
        print(
            os.getpid(),
            id(self.serializer),
            "dumping",
            repr(var),
        )
        return self.serializer.encode()

    def loads(self, s: str, name: str = "no name") -> Union[SX, MX]:
        self.deserializer.decode(s)
        print(
            os.getpid(),
            id(self.deserializer),
            "loading",
            name,
        )
        return self.deserializer.unpack()


_pickler_depickler_dict = {"pid": os.getpid(), "obj": CasadiPicklerDepickler()}


def get_pickler_depickler():
    global _pickler_depickler_dict
    if _pickler_depickler_dict["pid"] != os.getpid():
        importlib.reload(casadi)
        _pickler_depickler_dict = {"pid": os.getpid(), "obj": CasadiPicklerDepickler()}
    return _pickler_depickler_dict["obj"]


def depickle(
    *args,
    **kwargs,
):
    return get_pickler_depickler().loads(*args, **kwargs)


def pickle_casadi(obj):
    _pickler_depickler = get_pickler_depickler()
    s = _pickler_depickler.dumps(obj)
    return depickle, (s, repr(obj))


for casadi_class in [SX, MX]:
    copyreg.pickle(casadi_class, pickle_casadi)


if __name__ == "__main__":
    x = SX.sym("x")
    y = MX.sym("y")

    s = StringSerializer()
    s.pack(x)
    data1 = s.encode()
    s.reset()
    s.pack(y)
    data2 = s.encode()

    #  def printer(data):
    #      print(data)
    #      ind, data1, data2 = data
    #      s = StringDeserializer(data1)
    #      a = s.unpack()
    #      s.decode(data2)
    #      b = s.unpack()
    #      print(a, b)
    #      return data

    def printer(data):
        print(data)
        ind, data1, data2 = data
        s = StringDeserializer(data[ind])
        a = s.unpack()
        #  s.decode(data2)
        #  b = s.unpack()
        print(a)
        return data

    with concurrent.futures.ProcessPoolExecutor(2) as executor:
        futs = [
            executor.submit(printer, t) for t in [[1, data1, data2], [2, data1, data2]]
        ]
        print([fut.result() for fut in futs])
    #  x = _pickler_depickler.loads(_pickler_depickler.dumps(SX.sym("x")))
    #  y = _pickler_depickler.loads(_pickler_depickler.dumps(MX.sym("y")))
    #  x = pickle.loads(pickle.dumps(SX.sym("x")))
    #  y = pickle.loads(pickle.dumps(MX.sym("y")))
    #  data1 = pickle.dumps(SX.sym("x"))
    #  data2 = pickle.dumps(MX.sym("y"))
    #  x = pickle.loads(data1)
    #  y = pickle.loads(data2)
