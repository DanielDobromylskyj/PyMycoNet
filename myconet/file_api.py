import numpy as np
import lz4.frame
import struct


def compress(data):
    return lz4.frame.compress(data)

def decompress(data):
    return lz4.frame.decompress(data)


def encode_intx(v, length, file):
    file.write(v.to_bytes(length, byteorder="little"))

def decode_intx(length, file):
    return int.from_bytes(file.read(length), byteorder="little")


def encode_number(v: int, file):
    file.write(v.to_bytes(8, byteorder="little"))

def decode_int(file):
    return int.from_bytes(file.read(8), byteorder="little")

def encode_float(v: float, file):
    file.write(struct.pack("f", v))

def decode_float(file):
    return struct.unpack("f", file.read(4))[0]

def encode_bytes(v: bytes | bytearray, file):
    encode_number(len(v), file)
    file.write(v)
    return v

def decode_bytes(file) -> bytes:
    return file.read(decode_int(file))

def encode_str(v: str, file):
    encode_number(len(v), file)
    file.write(v.encode())

def decode_str(file):
    return file.read(decode_int(file)).decode()

def encode_none(v: None, file):
    file.write(b"N")

def decode_none(file):
    file.read(1)  # Assuming "N" is just one byte
    return None

def encode_bool(v: bool, file):
    file.write(int(v).to_bytes(1, byteorder="little"))

def decode_bool(file):
    return bool(int.from_bytes(file.read(1), byteorder="little"))

def encode_ndarray(v: np.ndarray, file, should_compress=False):
    encode_tuple(v.shape, file)
    encode_str(str(v.dtype), file)

    data = v.tobytes()

    if should_compress:
        data = compress(data)

    encode_number(len(data), file)
    file.write(data)

def decode_ndarray(file, is_compress=False):
    shape = decode_tuple(file)
    dtype = np.dtype(decode_str(file))

    raw = file.read(decode_int(file))

    if is_compress:
        raw = decompress(raw)

    return np.frombuffer(raw, dtype=dtype).reshape(shape)

def encode_list(v: list, file):
    values = {i: v[i] for i in range(len(v))}
    encode_dict(values, file)

def decode_list(file):
    return list(decode_dict(file).values())

def encode_tuple(v, file):
    values = {i: v[i] for i in range(len(v))}
    encode_dict(values, file)

def decode_tuple(file):
    return tuple(decode_dict(file).values())

type_lookup = {
    int: (0, encode_number, decode_int),
    str: (1, encode_str, decode_str),
    float: (2, encode_float, decode_float),
    bool: (3, encode_bool, decode_bool),
    type(None): (4, encode_none, decode_none),
    bytes: (5, encode_bytes, decode_bytes),
    dict: (6, None, None),
    np.ndarray: (7, encode_ndarray, decode_ndarray),
    list: (8, encode_list, decode_list),
    tuple: (9, encode_tuple, decode_tuple),
}


def get_type_from_int(v):
    for key, value in type_lookup.items():
        if value[0] == v:
            return key


def length_encode(data: bytes):
    return len(data).to_bytes(8, byteorder="little") + data

def length_decode(data: bytes):
    return int.from_bytes(data[:8], byteorder="little")

def decode_dict(file, is_compressed=False):
    pair_amount = decode_int(file)
    loaded_dict = {}

    for i in range(pair_amount):
        key_type = get_type_from_int(decode_int(file))
        value_type = get_type_from_int(decode_int(file))

        key = type_lookup[key_type][2](file)

        if value_type is dict:
            value = decode_dict(file)
        else:
            if value_type is np.ndarray and is_compressed:
                value = type_lookup[value_type][2](file, True)
            else:
                value = type_lookup[value_type][2](file)

        loaded_dict[key] = value

    return loaded_dict

def encode_dict(dictionary: dict, file, should_compress=False):
    encode_number(len(dictionary), file)

    for key, value in dictionary.items():
        if not type(key) in type_lookup or type(key) is dict:
            raise NotImplementedError(f"Cannot Encode Dict as key has an unsupported type: {type(key)}")

        if not type(value) in type_lookup:
            raise NotImplementedError(F"Cannot Encode Dict as value has an unsupported type: {key}: {type(value)}")

        encode_number(type_lookup[type(key)][0], file)
        encode_number(type_lookup[type(value)][0], file)
        type_lookup[type(key)][1](key, file)

        if type(value) is dict:
            encode_dict(value, file)
        else:
            if type(value) is np.ndarray and should_compress:
                type_lookup[type(value)][1](value, file, True)

            else:
                type_lookup[type(value)][1](value, file)