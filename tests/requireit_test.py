import re
from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from requireit import ValidationError
from requireit import require_array
from requireit import require_between
from requireit import require_negative
from requireit import require_nonnegative
from requireit import require_nonpositive
from requireit import require_one_of
from requireit import require_positive


@pytest.mark.parametrize(
    "require",
    (
        pytest.param(partial(require_between, -1, 0, 1), id="between-below"),
        pytest.param(partial(require_between, 2, 0, 1), id="between-above"),
        pytest.param(partial(require_negative, 0), id="negative-0"),
        pytest.param(partial(require_nonnegative, -1), id="nonnegative--1"),
        pytest.param(partial(require_nonpositive, 1), id="nonpositive-1"),
        pytest.param(partial(require_one_of, "foo", allowed=("bar",)), id="one_of-foo"),
        pytest.param(partial(require_positive, 0), id="positive-0"),
    ),
)
def test_require_with_name(require):
    with pytest.raises(ValidationError, match="^foobar"):
        require(name="foobar")


@pytest.mark.parametrize(
    "value, allowed",
    (
        (0, (0, 1)),
        (1, ("0", 1)),
        ("foo", ("foo", "bar", "baz")),
        ("", ("", 0, True)),
        ("b", "foobar"),
        (0, ([1], [2], 0)),
        ([1], ([1], [2])),
    ),
)
@pytest.mark.parametrize("allowed_type", (list, tuple, set))
def test_require_one_of_ok(value, allowed, allowed_type):
    try:
        allowed_type(allowed)
    except TypeError:
        pytest.skip("allowed contains unhashable items")
    assert require_one_of(value, allowed=allowed_type(allowed)) == value


@pytest.mark.parametrize(
    "value, allowed",
    (
        (-1, (0, 1)),
        (0, ("0", 1)),
        ("fu", ("foo", "bar", "baz")),
        (1, ()),
        ("bar", "foobar"),
        (0, ([0],)),
        ([0], (0, 1)),
    ),
)
@pytest.mark.parametrize("allowed_type", (list, tuple, set))
def test_require_one_of_not_ok(value, allowed, allowed_type):
    try:
        allowed_type(allowed)
    except TypeError:
        pytest.skip("allowed contains unhashable items")
    with pytest.raises(ValidationError, match="^value must be one of"):
        require_one_of(value, allowed=allowed_type(allowed))


@pytest.mark.parametrize("value", (1.0, (-1, 1), np.asarray([-1, 1])))
@pytest.mark.parametrize(
    "kwds",
    (
        {"a_min": -2.0},
        {"a_min": -1.0, "inclusive_min": True},
        {"a_max": 2.0},
        {"a_max": 1.0, "inclusive_max": True},
        {"a_min": -2.0, "a_max": 2.0},
        {"a_min": -1.0, "a_max": 1.0, "inclusive_min": True, "inclusive_max": True},
    ),
)
def test_require_between_valid(value, kwds):
    assert require_between(value, **kwds) is value


@pytest.mark.parametrize("value", (1.0, (-1, 1), np.asarray([-1, 1])))
@pytest.mark.parametrize(
    "kwds",
    (
        {"a_min": 2.0},
        {"a_min": 1.0, "inclusive_min": False},
        {"a_max": 0.0},
        {"a_max": 1.0, "inclusive_max": False},
        {"a_min": -1.0, "a_max": 1.0, "inclusive_min": True, "inclusive_max": False},
        {"a_min": 1.0, "a_max": 1.0, "inclusive_min": False, "inclusive_max": True},
    ),
)
def test_require_between_is_invalid(value, kwds):
    with pytest.raises(ValidationError, match="value must be"):
        require_between(value, **kwds)


@pytest.mark.parametrize("value,ok", ((-1.0, True), (0.0, False), (1.0, False)))
def test_require_negative(value, ok):
    if ok:
        assert require_negative(value) is value
    else:
        with pytest.raises(ValidationError, match="value must be <"):
            require_negative(value)


@pytest.mark.parametrize("value,ok", ((1.0, True), (0.0, False), (-1.0, False)))
def test_require_positive(value, ok):
    if ok:
        assert require_positive(value) is value
    else:
        with pytest.raises(ValidationError, match="value must be >"):
            require_positive(value)


@pytest.mark.parametrize("value,ok", ((-1.0, False), (0.0, True), (1.0, True)))
def test_require_nonnegative(value, ok):
    if ok:
        assert require_nonnegative(value) is value
    else:
        with pytest.raises(ValidationError, match="value must be >="):
            require_nonnegative(value)


@pytest.mark.parametrize("value,ok", ((-1.0, True), (0.0, True), (1.0, False)))
def test_require_nonpositive(value, ok):
    if ok:
        assert require_nonpositive(value) is value
    else:
        with pytest.raises(ValidationError, match="value must be <="):
            require_nonpositive(value)


@pytest.mark.parametrize("array", ([1.0, 2.0], [[1, 2]], [[1, 2], [3, 4]], []))
def test_require_array_noop(array):
    x = np.asarray(array)
    actual = require_array(x)
    assert actual is x
    assert_array_equal(actual, array)


@pytest.mark.parametrize("array", ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], []))
def test_require_returns_same_object_when_valid(array):
    x = np.asarray(array)
    actual = require_array(x, dtype=x.dtype, shape=x.shape)
    assert actual is x
    assert_array_equal(actual, array)


@pytest.mark.parametrize(
    "dtype",
    [np.float32, "float32", np.dtype("float32")],
)
@pytest.mark.parametrize("array", ([1.0, 2.0, 3.0, 4.0], []))
def test_require_dtype_accepts_multiple_specifiers(array, dtype):
    x = np.asarray(array, dtype=np.float32)
    actual = require_array(x, dtype=dtype, shape=x.shape)
    assert actual is x
    assert_array_equal(actual, array)


@pytest.mark.parametrize(
    "array,dtype",
    (
        ([1.0, 2.0, 3.0], int),
        ([1.0, 2.0, 3.0], bool),
        ([1.0, 2.0, 3.0], complex),
        ([1, 2, 3], float),
        ([1, 2, 3], complex),
        ([1, 2, 3], bool),
    ),
)
@pytest.mark.parametrize("name", (None, "foobar"))
def test_require_dtype_mismatch_raises(array, dtype, name):
    prefix = re.escape(name or "array")
    with pytest.raises(ValidationError, match=f"^{prefix} must have dtype"):
        require_array(np.asarray(array), dtype=dtype, name=name)


@pytest.mark.parametrize(
    "array,shape",
    [
        ([0, 0, 0, 0, 0, 0], (2, 3)),
        ([[0, 0, 0], [0, 0, 0]], (3, 2)),
        ([[0, 0], [0, 0], [0, 0]], (6,)),
    ],
)
@pytest.mark.parametrize("name", (None, "foobar"))
def test_require_shape_mismatch(array, shape, name):
    prefix = re.escape(name or "array")
    with pytest.raises(ValidationError, match=f"^{prefix} must have shape"):
        require_array(np.asarray(array), shape=shape, name=name)


def test_require_requires_writable_passes_when_writable():
    x = np.zeros(5)
    actual = require_array(x, writable=True)
    assert actual is x


@pytest.mark.parametrize("name", (None, "foobar"))
def test_require_requires_writable_raises_when_readonly(name):
    x = np.arange(5)
    x.setflags(write=False)

    prefix = re.escape(name or "array")
    with pytest.raises(ValidationError, match=f"^{prefix} must be writable"):
        require_array(x, writable=True, name=name)


@pytest.mark.parametrize("array", ([1, 2, 3], [], [[1, 2, 3], [4, 5, 6]]))
def test_require_contiguous_requirement_passes_for_c_contiguous(array):
    x = np.asarray(array).copy(order="C")
    assert x.flags.c_contiguous
    actual = require_array(x, contiguous=True)
    assert_array_equal(actual, array)


@pytest.mark.parametrize(
    "array",
    (
        np.arange(12).reshape((3, 4)).T,
        np.arange(12).reshape((3, 4))[:, ::2],
    ),
)
@pytest.mark.parametrize("name", (None, "foobar"))
def test_require_contiguous_requirement_raises_for_noncontiguous(array, name):
    prefix = re.escape(name or "array")
    with pytest.raises(ValidationError, match=f"^{prefix} must be contiguous"):
        require_array(array, contiguous=True, name=name)
