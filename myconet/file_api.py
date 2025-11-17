import numpy as np
import lz4.frame
import struct


class FileAPI:
    def __init__(self, file):
        self.f = file

        self.lookup = {
            int: (0, self.encode_uint64, self.decode_uint64, False),
            str: (1, self.encode_string, self.decode_string, False),
            float: (2, self.encode_float, self.decode_float, False),
            bool: (3, self.encode_bool, self.decode_bool, False),
            type(None): (4, self.encode_none, self.decode_none, False),
            bytes: (5, self.encode_bytes, self.decode_bytes, False),
            dict: (6, self.encode_dict, self.decode_dict, True),
            np.ndarray: (7, self.encode_ndarray, self.decode_ndarray, True),
            list: (8, self.encode_list, self.decode_list, False),
            tuple: (9, self.encode_tuple, self.decode_tuple, False),
        }

    # ========== ENCODE / DECODE METHODS ==========

    def encode_uint64(self, integer: int):
        self.f.write(struct.pack("<Q", integer))

    def decode_uint64(self) -> int:
        return struct.unpack("<Q", self.f.read(8))[0]

    def encode_float(self, float_value: float):
        self.f.write(struct.pack("f", float_value))

    def decode_float(self) -> float:
        return struct.unpack("f", self.f.read(4))[0]

    def encode_bytes(self, data: bytes):
        self.encode_uint64(len(data))
        self.f.write(data)

    def decode_bytes(self) -> bytes:
        return self.f.read(self.decode_uint64())

    def encode_string(self, string_data: str):
        self.encode_bytes(string_data.encode(encoding="utf-8"))

    def decode_string(self) -> str:
        return self.decode_bytes().decode(encoding="utf-8")

    def encode_none(self, none_value: None):
        self.f.write(b"N")

    def decode_none(self) -> None:
        byte = self.f.read(1)  # Assuming "N" is just one byte
        assert byte == b"N", "Failed to decode / validate NoneType."

    def encode_bool(self, boolean_value: bool):
        self.f.write(int(boolean_value).to_bytes(1, byteorder="little"))

    def decode_bool(self):
        return bool(int.from_bytes(self.f.read(1), byteorder="little"))

    def encode_ndarray(self, v: np.ndarray, should_compress=False):
        self.encode_tuple(v.shape)
        self.encode_string(str(v.dtype))

        data = v.tobytes()
        if should_compress:
            data = lz4.frame.compress(data)

        self.encode_bytes(data)

    def decode_ndarray(self, is_compress=False):
        shape = self.decode_tuple()
        dtype = np.dtype(self.decode_string())

        raw = self.decode_bytes()
        if is_compress:
            raw = lz4.frame.decompress(raw)

        return np.frombuffer(raw, dtype=dtype).reshape(shape)

    def encode_list(self, list_data: list):
        self.encode_dict({i: list_data[i] for i in range(len(list_data))}, False)

    def decode_list(self):
        return list(self.decode_dict(False).values())

    def encode_tuple(self, tuple_data):
        self.encode_dict({i: tuple_data[i] for i in range(len(tuple_data))}, False)

    def decode_tuple(self):
        return tuple(self.decode_dict(False).values())

    def encode_dict(self, dictionary: dict, use_compression):
        self.encode_uint64(len(dictionary))

        for key, value in dictionary.items():
            if not type(key) in self.lookup or type(key) is dict:
                raise NotImplementedError(f"Cannot Encode Dict as key has an unsupported type: {type(key)}")

            if not type(value) in self.lookup:
                raise NotImplementedError(F"Cannot Encode Dict as value has an unsupported type: {key}: {type(value)}")

            key_lookup = self.lookup[type(key)]
            key_encode_func = key_lookup[1]

            item_lookup = self.lookup[type(value)]
            item_encode_func = item_lookup[1]

            self.encode_uint64(key_lookup[0])
            self.encode_uint64(item_lookup[0])

            if key_lookup[3] is True:  # Can be compressed
                key_encode_func(key, use_compression)
            else:
                key_encode_func(key)

            if item_lookup[3] is True:  # Can be compressed
                item_encode_func(value, use_compression)
            else:
                item_encode_func(value)


    def decode_dict(self, used_compression) -> dict:
        pair_amount = self.decode_uint64()
        loaded_dict = {}

        for i in range(pair_amount):
            key_type = self.__get_key_from_lookup(self.decode_uint64())
            value_type = self.__get_key_from_lookup(self.decode_uint64())

            key_lookup = self.lookup[key_type]
            key_decode_func = key_lookup[2]

            item_lookup = self.lookup[value_type]
            item_decode_func = item_lookup[2]

            if key_lookup[3] is True:  # Can be compressed
                key = key_decode_func(used_compression)
            else:
                key = key_decode_func()

            if item_lookup[3] is True:  # Can be compressed
                item = item_decode_func(used_compression)
            else:
                item = item_decode_func()

            loaded_dict[key] = item

        return loaded_dict

    # ==========   HELPER METHODS   ==========

    def __get_key_from_lookup(self, search_id: int):
        for key, value in self.lookup.items():
            if value[0] == search_id:
                return key

        raise NotImplementedError("Unknown Search ID!")