# Owner(s): ["oncall: pt2"]

import torch
from torch._dynamo.test_case import TestCase, run_tests


class DynamoByteArrayTests(TestCase):
    def test_bytearray_construction_and_len(self):
        def fn():
            b = bytearray(b"abc")
            return len(b)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(), 3)

    def test_bytearray_getitem_and_setitem(self):
        def fn():
            b = bytearray(b"hello")
            b[0] = 72  # 'H'
            b[1] = 69  # 'E'
            return b[0], b[1], bytes(b)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        r0, r1, r_bytes = opt_fn()
        self.assertEqual(r0, 72)
        self.assertEqual(r1, 69)
        self.assertEqual(r_bytes, b"HEllo")

    def test_bytearray_append_and_extend(self):
        def fn():
            b = bytearray(b"ab")
            b.append(99)  # 'c'
            b.extend(b"de")
            return bytes(b)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(), b"abcde")

    def test_bytearray_pop_and_clear(self):
        def fn():
            b = bytearray(b"xyz")
            val = b.pop()
            b.clear()
            return val, len(b)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        val, length = opt_fn()
        self.assertEqual(val, 122)  # 'z'
        self.assertEqual(length, 0)

    def test_bytearray_reverse_and_copy(self):
        def fn():
            b = bytearray(b"123")
            b.reverse()
            c = b.copy()
            c.append(52)  # '4'
            return bytes(b), bytes(c)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        b_res, c_res = opt_fn()
        self.assertEqual(b_res, b"321")
        self.assertEqual(c_res, b"3214")

    def test_bytearray_decode_and_hex(self):
        def fn():
            b = bytearray(b"hello world")
            h = b.hex()
            s = b.decode("utf-8")
            return h, s

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        h, s = opt_fn()
        self.assertEqual(h, "68656c6c6f20776f726c64")
        self.assertEqual(s, "hello world")

    def test_bytearray_slice_assignment(self):
        def fn():
            b = bytearray(b"abcdef")
            b[1:4] = b"XYZ"
            return bytes(b)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(), b"aXYZef")

    def test_bytearray_equality_and_contains(self):
        def fn():
            b1 = bytearray(b"test")
            b2 = bytearray(b"test")
            b3 = bytearray(b"other")
            return (b1 == b2), (b1 == b3), (101 in b1)  # 101 is 'e'

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        eq1, eq2, inc = opt_fn()
        self.assertTrue(eq1)
        self.assertFalse(eq2)
        self.assertTrue(inc)


if __name__ == "__main__":
    run_tests()
