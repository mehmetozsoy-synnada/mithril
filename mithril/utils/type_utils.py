# Copyright 2022 Synnada, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from types import GenericAlias, UnionType
from typing import (
    Any,
    TypeGuard,
    Union,
    get_origin,
)


def is_int_tuple_tuple(
    data: tuple[int, int] | tuple[tuple[int, int], tuple[int, int]],
) -> TypeGuard[tuple[tuple[int, int], tuple[int, int]]]:
    return isinstance(data[0], tuple)


def is_tuple_int(t: Any) -> TypeGuard[tuple[int, ...]]:
    return isinstance(t, tuple) and all(isinstance(i, int) for i in t)


def is_list_int(t: Any) -> TypeGuard[list[int]]:
    return isinstance(t, list) and all(isinstance(i, int) for i in t)


def is_list_str(t: Any) -> TypeGuard[list[str]]:
    return isinstance(t, list) and all(isinstance(i, str) for i in t)


def is_list_int_or_none(t: Any) -> TypeGuard[list[int | None]]:
    return isinstance(t, list) and all(isinstance(i, int | None) for i in t)


def is_tuple_int_or_none(t: Any) -> TypeGuard[tuple[int | None, ...]]:
    return isinstance(t, tuple) and all(isinstance(i, int | None) for i in t)


def is_axis_reduce_type(
    axis: Any,
) -> TypeGuard[int | tuple[int, ...] | None]:
    is_int = isinstance(axis, int)
    is_int_tuple = is_tuple_int(axis)
    is_none = axis is None
    return is_int or is_int_tuple or is_none


def is_axis_reverse_type(
    axis: Any,
) -> TypeGuard[list[int] | tuple[int, ...] | None]:
    is_list = is_list_int(axis)
    is_tuple = is_tuple_int(axis)
    is_none = axis is None
    return is_list or is_none or is_tuple


def is_tuple_of_two_ints(obj: Any) -> TypeGuard[tuple[int, int]]:
    return (
        isinstance(obj, tuple)
        and len(obj) == 2
        and all(isinstance(i, int) for i in obj)
    )


def is_padding_type(
    padding: Any,
) -> TypeGuard[tuple[tuple[int, int], tuple[int, int]] | tuple[int, int]]:
    is_padding = False
    if isinstance(padding, tuple) and len(padding) == 2:
        is_padding = (
            is_tuple_of_two_ints(padding[0])
            and is_tuple_of_two_ints(padding[1])
            or is_tuple_of_two_ints(padding)
        )
    return is_padding


def is_union_type(
    type: Any,
) -> TypeGuard[UnionType]:
    return (get_origin(type)) in (Union, UnionType)


def is_generic_alias_type(
    typ: Any,
) -> TypeGuard[GenericAlias]:
    true_generic_alias = type(typ) is GenericAlias
    not_union = not is_union_type(typ)
    is_origin_single_type = type(get_origin(typ)) is type
    return true_generic_alias or (not_union and is_origin_single_type)
