"""
ByteArrayVariable variable tracking implementation for PyTorch Dynamo.

Provides symbolic tracking and graph-break-free compilation support for
Python's mutable `bytearray` type, including sequence operations, in-place
mutations, string/bytes conversion methods, and protocol operations.
"""

from __future__ import annotations

import operator
from typing import Any, TYPE_CHECKING

from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import raise_observed_exception, raise_type_error
from ..utils import unpack_iterable
from .base import GetSet, Method, ValueMutationNew, VariableTracker
from .constant import ConstantVariable


if TYPE_CHECKING:
    from torch._dynamo.codegen import PyCodegen
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase


class ByteArrayVariable(VariableTracker):
    """
    Variable tracker for Python's mutable `bytearray` type.
    """

    _cpython_type = bytearray

    def __init__(
        self,
        items: list[VariableTracker],
        mutation_type: ValueMutationNew | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.items: list[VariableTracker] = list(items)
        if mutation_type is not None:
            self.mutation_type = mutation_type

    def python_type(self) -> type[bytearray]:
        return bytearray

    def as_python_constant(self) -> bytearray:
        return bytearray([x.as_python_constant() for x in self.items])

    def is_python_constant(self) -> bool:
        return all(x.is_python_constant() for x in self.items)

    def unpack_var_sequence(self, tx: InstructionTranslatorBase) -> list[VariableTracker]:
        return list(self.items)

    def reconstruct(self, codegen: PyCodegen) -> None:
        if self.is_python_constant():
            codegen.extend_output(
                [
                    codegen.create_load_const(self.as_python_constant()),
                ]
            )
        else:
            for item in self.items:
                codegen(item)
            codegen.extend_output(
                [
                    create_instruction("BUILD_LIST", arg=len(self.items)),
                    codegen.create_load_python_module(bytearray),
                    create_instruction("ROT_TWO"),
                    create_call_function(1, True),
                ]
            )

    # -------------------------------------------------------------------------
    # tp_methods implementation for bytearray methods
    # -------------------------------------------------------------------------

    def _bytearray_append(
        self,
        tx: InstructionTranslatorBase,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if len(args) != 1 or kwargs:
            raise_type_error(tx, "append() takes exactly one argument")
        val = args[0]
        if not val.is_python_constant():
            tx.output.side_effects.mutation(self)
            self.items.append(val)
            return ConstantVariable.create(None)

        int_val = val.as_python_constant()
        if not isinstance(int_val, int) or not (0 <= int_val <= 255):
            raise_observed_exception(ValueError, tx, args=["an integer is required (got type ...)"])
        tx.output.side_effects.mutation(self)
        self.items.append(ConstantVariable.create(int_val))
        return ConstantVariable.create(None)

    def _bytearray_extend(
        self,
        tx: InstructionTranslatorBase,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if len(args) != 1 or kwargs:
            raise_type_error(tx, "extend() takes exactly one argument")
        iterable = args[0]
        unpacked = unpack_iterable(tx, iterable)
        for item in unpacked:
            if item.is_python_constant():
                v = item.as_python_constant()
                if not isinstance(v, int) or not (0 <= v <= 255):
                    raise_observed_exception(ValueError, tx, args=["byte must be in range(0, 256)"])
                self.items.append(ConstantVariable.create(v))
            else:
                self.items.append(item)
        tx.output.side_effects.mutation(self)
        return ConstantVariable.create(None)

    def _bytearray_pop(
        self,
        tx: InstructionTranslatorBase,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if kwargs or len(args) > 1:
            raise_type_error(tx, "pop() takes at most 1 argument")
        idx = -1
        if args:
            if not args[0].is_python_constant():
                raise_type_error(tx, "pop index must be an integer")
            idx = args[0].as_python_constant()
        if not self.items:
            raise_observed_exception(IndexError, tx, args=["pop from empty bytearray"])
        try:
            val = self.items.pop(idx)
            tx.output.side_effects.mutation(self)
            return val
        except IndexError as e:
            raise_observed_exception(IndexError, tx, args=list(e.args))

    def _bytearray_clear(
        self,
        tx: InstructionTranslatorBase,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if args or kwargs:
            raise_type_error(tx, "clear() takes no arguments")
        self.items.clear()
        tx.output.side_effects.mutation(self)
        return ConstantVariable.create(None)

    def _bytearray_reverse(
        self,
        tx: InstructionTranslatorBase,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if args or kwargs:
            raise_type_error(tx, "reverse() takes no arguments")
        self.items.reverse()
        tx.output.side_effects.mutation(self)
        return ConstantVariable.create(None)

    def _bytearray_copy(
        self,
        tx: InstructionTranslatorBase,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if args or kwargs:
            raise_type_error(tx, "copy() takes no arguments")
        return ByteArrayVariable(list(self.items), mutation_type=ValueMutationNew())

    def _bytearray_decode(
        self,
        tx: InstructionTranslatorBase,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if not self.is_python_constant():
            raise_type_error(tx, "decode requires constant bytearray")
        py_bytes = self.as_python_constant()
        encoding = "utf-8"
        errors = "strict"
        if args:
            encoding = args[0].as_python_constant()
        if len(args) > 1:
            errors = args[1].as_python_constant()
        if "encoding" in kwargs:
            encoding = kwargs["encoding"].as_python_constant()
        if "errors" in kwargs:
            errors = kwargs["errors"].as_python_constant()

        try:
            res = py_bytes.decode(encoding, errors)
            return ConstantVariable.create(res)
        except Exception as e:
            raise_observed_exception(type(e), tx, args=list(e.args))

    def _bytearray_hex(
        self,
        tx: InstructionTranslatorBase,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if not self.is_python_constant():
            raise_type_error(tx, "hex requires constant bytearray")
        res = self.as_python_constant().hex()
        return ConstantVariable.create(res)

    tp_methods: dict[str, Method] = {
        "append": Method(_bytearray_append, pass_tx=True),
        "extend": Method(_bytearray_extend, pass_tx=True),
        "pop": Method(_bytearray_pop, pass_tx=True),
        "clear": Method(_bytearray_clear, pass_tx=True),
        "reverse": Method(_bytearray_reverse, pass_tx=True),
        "copy": Method(_bytearray_copy, pass_tx=True),
        "decode": Method(_bytearray_decode, pass_tx=True),
        "hex": Method(_bytearray_hex, pass_tx=True),
    }

    # -------------------------------------------------------------------------
    # Method / Operator dispatch
    # -------------------------------------------------------------------------

    def call_method(
        self,
        tx: InstructionTranslatorBase,
        name: str,
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        if name in self.tp_methods:
            return self.tp_methods[name](self, tx, args, kwargs)

        if name == "__getitem__":
            if len(args) != 1 or kwargs:
                raise_type_error(tx, "__getitem__ takes 1 argument")
            key = args[0]
            if key.is_python_constant():
                k = key.as_python_constant()
                if isinstance(k, int):
                    try:
                        return self.items[k]
                    except IndexError as e:
                        raise_observed_exception(IndexError, tx, args=list(e.args))
                elif isinstance(k, slice):
                    return ByteArrayVariable(self.items[k], mutation_type=ValueMutationNew())

        if name == "__setitem__":
            if len(args) != 2 or kwargs:
                raise_type_error(tx, "__setitem__ takes 2 arguments")
            key, val = args[0], args[1]
            if key.is_python_constant():
                k = key.as_python_constant()
                if isinstance(k, int):
                    if val.is_python_constant():
                        v = val.as_python_constant()
                        if not isinstance(v, int) or not (0 <= v <= 255):
                            raise_observed_exception(ValueError, tx, args=["byte must be in range(0, 256)"])
                        self.items[k] = ConstantVariable.create(v)
                    else:
                        self.items[k] = val
                    tx.output.side_effects.mutation(self)
                    return ConstantVariable.create(None)
                elif isinstance(k, slice):
                    unpacked = unpack_iterable(tx, val)
                    self.items[k] = unpacked
                    tx.output.side_effects.mutation(self)
                    return ConstantVariable.create(None)

        if name == "__len__":
            return ConstantVariable.create(len(self.items))

        if name == "__contains__":
            if len(args) != 1 or kwargs:
                raise_type_error(tx, "__contains__ takes 1 argument")
            target = args[0]
            if target.is_python_constant() and self.is_python_constant():
                res = target.as_python_constant() in self.as_python_constant()
                return ConstantVariable.create(res)

        if name in ("__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__"):
            if len(args) != 1 or kwargs:
                raise_type_error(tx, f"{name} takes 1 argument")
            other = args[0]
            if self.is_python_constant() and other.is_python_constant():
                s_val = self.as_python_constant()
                o_val = other.as_python_constant()
                op = getattr(operator, name.strip("_"))
                return ConstantVariable.create(op(s_val, o_val))

        return super().call_method(tx, name, args, kwargs)
