import concurrent.futures
import multiprocessing as mp
from typing import Union

from casadi import MX, SX, StringDeserializer, StringSerializer


class NewForkingPickler(mp.reduction.ForkingPickler):
    def __init__(self, *args, **kwargs):
        #  print("Initing", self.__class__, id(self))
        super().__init__(*args, **kwargs)

        _casadi_serializer = CasadiPicklerDepickler()

        def pickle_casadi(obj):
            #  print("pickling", obj)
            s = _casadi_serializer.dumps(obj)
            return _casadi_serializer.loads, (s, repr(obj))

        self.dispatch_table.update({SX: pickle_casadi, MX: pickle_casadi})


class CasadiPicklerDepickler:
    serializer = StringSerializer()
    deserializer = StringDeserializer(serializer.encode())

    def __init__(self):
        #  print("creating pickler/depickler", os.getpid())
        self.__class__.serializer = StringSerializer()
        self.__class__.deserializer = StringDeserializer(
            self.__class__.serializer.encode()
        )

    def dumps(self, var: Union[SX, MX]) -> str:
        #  print(os.getpid(), id(self.serializer), "starting to dump", repr(var))
        try:
            self.serializer.pack(var)
        except Exception as e:
            print("failed to deserialize", e)
            raise e
        #  print(os.getpid(), id(self.serializer), "dump succesful", repr(var))
        return self.serializer.encode()

    def loads(self, s: str, name: str = "no name") -> Union[SX, MX]:
        #  print(os.getpid(), id(self.deserializer), "starting to load", name)
        try:
            self.deserializer.decode(s)
            var = self.deserializer.unpack()
        except Exception as e:
            print("failed to deserialize", name, e)
            raise e
        #  print(os.getpid(), id(self.deserializer), "load succesful", name)
        return var


mp.reduction.ForkingPickler = NewForkingPickler


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
        return data

    with concurrent.futures.ProcessPoolExecutor(2) as executor:
        futs = [executor.submit(printer, t) for t in [[1, x, x], [2, x, x]]]
        print([fut.result() for fut in futs])
    #  x = _pickler_depickler.loads(_pickler_depickler.dumps(SX.sym("x")))
    #  y = _pickler_depickler.loads(_pickler_depickler.dumps(MX.sym("y")))
    #  x = pickle.loads(pickle.dumps(SX.sym("x")))
    #  y = pickle.loads(pickle.dumps(MX.sym("y")))
    #  data1 = pickle.dumps(SX.sym("x"))
    #  data2 = pickle.dumps(MX.sym("y"))
    #  x = pickle.loads(data1)
    #  y = pickle.loads(data2)
