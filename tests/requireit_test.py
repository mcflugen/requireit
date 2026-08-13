import re
from functools import partial

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from requireit import ValidationError
from requireit import add_note
from requireit import import_package
from requireit import raise_as
from requireit import require_array
from requireit import require_between
from requireit import require_contains
from requireit import require_dtype
from requireit import require_greater_than
from requireit import require_greater_than_or_equal
from requireit import require_instance
from requireit import require_length
from requireit import require_length_at_least
from requireit import require_length_at_most
from requireit import require_length_between
from requireit import require_less_than
from requireit import require_less_than_or_equal
from requireit import require_like
from requireit import require_ndim
from requireit import require_negative
from requireit import require_none
from requireit import require_nonnegative
from requireit import require_nonpositive
from requireit import require_not_none
from requireit import require_not_one_of
from requireit import require_one_of
from requireit import require_path_string
from requireit import require_positive
from requireit import require_shape
from requireit import require_sorted

CHECKS_THAT_FAIL = {
    ">": partial(require_greater_than, 0, 0),
    ">=": partial(require_greater_than_or_equal, 0, 1),
    "<": partial(require_less_than, 0, 0),
    "<=": partial(require_less_than_or_equal, 0, -1),
    "between-above": partial(require_between, 2, 0, 1),
    "between-below": partial(require_between, -1, 0, 1),
    "contains": partial(require_contains, {"foo", "bar"}, required=("baz",)),
    "dtype": partial(require_dtype, [0], "float"),
    "instance": partial(require_instance, 0, float),
    "length-2": partial(require_length, [1, 2, 3], 2),
    "like": partial(require_like, [1, 2, 3], 2),
    "length_at_least-4": partial(require_length_at_least, [1, 2, 3], 4),
    "length_at_most-2": partial(require_length_at_most, [1, 2, 3], 2),
    "length_between-long": partial(require_length_between, [1, 2, 3], 0, 2),
    "length_between-short": partial(require_length_between, [1, 2, 3], 4, 9),
    "ndim": partial(require_ndim, [1, 2, 3], 2),
    "negative-0": partial(require_negative, (0,)),
    "none": partial(require_none, True),
    "nonnegative--1": partial(require_nonnegative, -1),
    "nonpositive-1": partial(require_nonpositive, 1),
    "not_none": partial(require_not_none, None),
    "not_one_of-foo": partial(require_not_one_of, "foo", forbidden=("foo",)),
    "one_of-foo": partial(require_one_of, "foo", allowed=("bar",)),
    "positive-0": partial(require_positive, 0),
    "path-empty": partial(require_path_string, ""),
    "path-0": partial(require_path_string, 0),
    "path-null": partial(require_path_string, "foo\x00bar"),
    "sorted": partial(require_sorted, [0, 1, 0]),
}


CHECKS_THAT_PASS = {
    ">": (partial(require_greater_than, lower=0.0), 1.0),
    ">=": (partial(require_greater_than_or_equal, lower=0), 0),
    "<": (partial(require_less_than, upper=1.0), 0.0),
    "<=": (partial(require_less_than_or_equal, upper=1), 1),
    "require_array": (require_array, np.asarray(0.0)),
    "require_between": (partial(require_between, a_min=-1, a_max=1), (0.0,)),
    "require_contains": (partial(require_contains, required={"bar"}), {"foo", "bar"}),
    "require_dtype": (partial(require_dtype, dtype=float), [0.0]),
    "instance": (partial(require_instance, types=int), 0),
    "length": (partial(require_length, length=2), (1, 2)),
    "like": (partial(require_like, other=[1, 2]), [0, 0]),
    "length_at_least": (partial(require_length_at_least, length=1), (1, 2)),
    "length_at_most": (partial(require_length_at_most, length=4), (1, 2)),
    "length_between": (partial(require_length_between, minimum=0, maximum=3), (1, 2)),
    "ndim": (partial(require_ndim, ndim=1), (1, 2)),
    "none": (require_none, None),
    "not_none": (require_not_none, False),
    "not_one_of": (partial(require_not_one_of, forbidden=()), "foo"),
    "require_negative": (require_negative, -1.0),
    "require_nonnegative": (require_nonnegative, 0.0),
    "require_nonpositive": (require_nonpositive, 0.0),
    "require_one_of": (partial(require_one_of, allowed={"foo"}), "foo"),
    "require_path_string": (require_path_string, "/foo"),
    "require_positive": (require_positive, 1.0),
    "sorted": (require_sorted, [0, 1, 2]),
}


@pytest.mark.parametrize(
    "require", [pytest.param(f, id=id_) for id_, f in CHECKS_THAT_FAIL.items()]
)
def test_require_with_name(require):
    with pytest.raises(ValidationError, match="^foobar must"):
        require(name="foobar")


@pytest.mark.parametrize(
    "require, value",
    [pytest.param(f, value, id=id_) for id_, (f, value) in CHECKS_THAT_PASS.items()],
)
def test_require_returns_input(require, value):
    actual = require(value)
    assert actual is value


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
    "value, forbidden",
    (
        (0, (1, 2)),
        (1, ("1", 0)),
        ("foo", ("foobar", "bar", "baz")),
        ("", (0, True)),
        ("b", "foo"),
        (0, ([0], [1], 1)),
        ([1], ([0], [2])),
    ),
)
@pytest.mark.parametrize("forbidden_type", (list, tuple, set))
def test_require_not_one_of_ok(value, forbidden, forbidden_type):
    try:
        forbidden_type(forbidden)
    except TypeError:
        pytest.skip("forbidden contains unhashable items")
    assert require_not_one_of(value, forbidden=forbidden_type(forbidden)) == value


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


@pytest.mark.parametrize(
    "value, forbidden",
    (
        (-1, (0, 1, -1)),
        (0, (1, 0)),
        ("foo", ("foo", "bar", "baz")),
        ("b", "foobar"),
        (0, ([0], 0)),
        ([0], (0, 1, [0])),
    ),
)
@pytest.mark.parametrize("forbidden_type", (list, tuple, set))
def test_require_not_one_of_not_ok(value, forbidden, forbidden_type):
    try:
        forbidden_type(forbidden)
    except TypeError:
        pytest.skip("forbidden contains unhashable items")
    with pytest.raises(ValidationError, match="^value must not be one of"):
        require_not_one_of(value, forbidden=forbidden_type(forbidden))


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


@pytest.mark.parametrize(
    "func",
    (
        require_between,
        partial(require_greater_than, lower=0.0),
        partial(require_greater_than_or_equal, lower=0.0),
        partial(require_less_than, upper=0.0),
        partial(require_less_than_or_equal, upper=0.0),
        require_negative,
        require_nonnegative,
        require_nonpositive,
        require_positive,
    ),
)
def test_nan_is_invalid(func):
    with pytest.raises(ValidationError, match="^value must not be nan"):
        func(np.nan)


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


@pytest.mark.parametrize("values", ([1.0, 2.0], [[1, 2]], [[1, 2], [3, 4]], []))
def test_require_array_noop(values):
    array = np.array(values)
    actual = require_array(array)
    assert actual is array
    assert_array_equal(actual, values)


@pytest.mark.parametrize("array", ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], []))
def test_require_returns_same_object_when_valid(array):
    x = np.asarray(array)
    actual = require_array(x, dtype=x.dtype, shape=x.shape)
    assert actual is x
    assert_array_equal(actual, array)


@pytest.mark.parametrize(
    "dtype",
    [np.float32, "float32", np.dtype("float32"), np.floating],
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
        ([1.0, 2.0, 3.0], np.integer),
        ([1, 2, 3], float),
        ([1, 2, 3], complex),
        ([1, 2, 3], bool),
        ([1, 2, 3], np.floating),
    ),
)
@pytest.mark.parametrize("name", (None, "foobar"))
def test_require_dtype_mismatch_raises(array, dtype, name):
    prefix = re.escape(name or "array")
    with pytest.raises(ValidationError, match=f"^{prefix} must have dtype"):
        require_array(np.asarray(array), dtype=dtype, name=name)


def test_require_dtype_mismatch_message_does_not_contain_none():
    with pytest.raises(ValidationError, match="must have dtype") as exc_info:
        require_dtype(np.array([1, 2, 3]), np.floating)
    assert "None" not in str(exc_info.value)
    assert "floating" in str(exc_info.value)


@pytest.mark.parametrize(
    "array,shape",
    [
        ([[0, 0, 0], [0, 0, 0]], (3, 2)),
        ([[0, 0], [0, 0], [0, 0]], (2, 3)),
    ],
)
@pytest.mark.parametrize("name", (None, "foobar"))
def test_require_shape(array, shape, name):
    prefix = re.escape(name or "array")
    with pytest.raises(ValidationError, match=f"^{prefix} must have shape"):
        require_array(np.asarray(array), shape=shape, name=name)


@pytest.mark.parametrize(
    "array,shape",
    [
        ([0, 0, 0, 0, 0, 0], (2, 3)),
        ([[0, 0], [0, 0], [0, 0]], (6,)),
    ],
)
@pytest.mark.parametrize("name", (None, "foobar"))
def test_require_shape_dimension_mismatch(array, shape, name):
    prefix = re.escape(name or "array")
    with pytest.raises(
        ValidationError, match=f"^{prefix} must have {len(shape)} dimensions"
    ):
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


@pytest.mark.parametrize(
    "array, shape",
    (
        ([1, 2, 3], (3,)),
        ([1, 2, 3], ("n",)),
        ([1, 2, 3], (None,)),
        ([[1, 2, 3]], (1, 3)),
        ([[1, 2, 3]], ("n", 3)),
        ([[1, 2, 3]], ("n", "n")),
        ([[1, 2, 3]], (1, "n")),
        ([[1, 2, 3]], (None, "n")),
    ),
)
def test_require_shape_is_ok(array, shape):
    actual = require_shape(array, shape)
    assert actual is array


@pytest.mark.parametrize("shape", [(1,), (4,)])
def test_require_shape_with_wrong_shape(shape):
    with pytest.raises(ValidationError):
        require_shape([1, 2, 3], shape=shape)


@pytest.mark.parametrize("shape", [(1, "n"), ("n", 1), (1, 2)])
def test_require_shape_with_wrong_dimensionality(shape):
    with pytest.raises(ValidationError):
        require_shape([1, 2, 3], shape=shape)


@pytest.mark.parametrize("shape", [(True, 2), (2.5, 2), (object(), 2)])
def test_require_shape_raises_for_invalid_shape(shape):
    with pytest.raises(TypeError):
        require_shape(np.ones((3, 2)), shape)


def test_require_like_same_shape_ok():
    a = np.zeros((3, 4), dtype=int)
    b = np.zeros((3, 4), dtype=float)
    assert require_like(b, a, shape=True, dtype=False) is b


def test_require_like_same_dtype_ok():
    a = np.ones((3, 4), dtype=np.float32)
    b = np.zeros((2, 4), dtype=np.float32)
    assert require_like(b, a, shape=False, dtype=True) is b


def test_require_like_shape_and_dtype_ok():
    a = np.ones((3, 4), dtype=np.float32)
    b = np.zeros((3, 4), dtype=np.float32)
    assert require_like(b, a, shape=True, dtype=True) is b


def test_require_like_noop():
    a = np.ones((2, 3), dtype=int)
    b = np.zeros((3, 4), dtype=float)
    assert require_like(b, a, shape=False, dtype=False) is b


def test_require_like_shape_mismatch_raises():
    a = np.ones((3, 4))
    b = np.zeros((2, 4))
    with pytest.raises(ValidationError, match="^array must have the same shape as"):
        require_like(b, a, dtype=False)


def test_require_like_dtype_mismatch_raises():
    a = np.ones((3, 4), dtype=np.float32)
    b = np.zeros((3, 4), dtype=np.float64)
    with pytest.raises(ValidationError, match="^array must have the same dtype as"):
        require_like(b, a, shape=False)


@pytest.mark.parametrize("name", [None, "foo"])
def test_require_like_error_message_uses_other_name(name):
    name = "other" if name is None else name
    with pytest.raises(
        ValidationError,
        match=f"^array must have the same shape as {name}",
    ):
        require_like([1, 2, 3], [1, 2], other_name=name)


def test_require_like_dtype_not_checked_when_false():
    a = np.ones((3, 4), dtype=np.float32)
    b = np.zeros((3, 4), dtype=np.float64)
    assert require_like(b, a, dtype=False) is b


def test_require_like_shape_not_checked_when_false():
    a = np.ones((3, 4))
    b = np.zeros((2, 4))
    assert require_like(b, a, shape=False) is b


def test_require_like_note_contains_concrete_shape():
    with pytest.raises(ValidationError) as exc_info:
        require_like([1, 2], [1, 2, 3])
    assert exc_info.value.__notes__ == ["array must have shape (3,)"]


def test_require_like_note_contains_concrete_dtype():
    with pytest.raises(ValidationError) as exc_info:
        require_like([1, 2], np.array([1.0, 2.0], dtype=np.float64), dtype=True)
    assert exc_info.value.__notes__ == ["array must have dtype float64"]


@pytest.mark.parametrize(
    "array,ndim",
    [
        (np.array(0), 0),
        ([1, 2, 3], 1),
        (np.ones((3, 4)), 2),
        (np.ones((2, 3, 4)), 3),
    ],
)
def test_require_ndim_ok(array, ndim):
    actual = require_ndim(array, ndim)
    assert actual is array


@pytest.mark.parametrize(
    "array,ndim",
    [
        ([1, 2, 3], 2),
        (np.ones((3, 4)), 1),
        (np.ones((3, 4)), 3),
    ],
)
def test_require_ndim_mismatch_raises(array, ndim):
    with pytest.raises(ValidationError, match=f"^array must have {ndim}"):
        require_ndim(array, ndim)


def test_require_ndim_singular():
    with pytest.raises(ValidationError, match="must have 1 dimension$"):
        require_ndim(np.ones((3, 4)), 1)


def test_require_ndim_plural():
    with pytest.raises(ValidationError, match="must have 2 dimensions$"):
        require_ndim([1, 2, 3], 2)


def test_require_ndim_with_negative_ndim():
    with pytest.raises(ValidationError, match="^ndim must be >= 0"):
        require_ndim([1, 2, 3], -1)


@pytest.mark.parametrize(
    "path", ("a", "a/b", "https://foo/bar", "--foo", "*", "..", ".")
)
def test_require_path_string(path):
    actual = require_path_string(path)
    assert actual is path


@pytest.mark.parametrize(
    "path, match",
    (
        ("", "path must not be empty"),
        ("foo\x00bar", "path must not contain null characters"),
        (b"foo", "path must be a string"),
    ),
)
def test_require_path_string_bad_input(path, match):
    with pytest.raises(ValidationError, match=match):
        require_path_string(path)


@pytest.mark.parametrize(
    "value", (0.0, np.asarray(0.0), np.asarray([0.0, -1.0]), np.asarray([[0.0]]))
)
def test_require_less_than(value):
    actual = require_less_than(value, upper=1.0)
    assert actual is value

    with pytest.raises(ValidationError, match="^value must be < "):
        require_less_than(value, upper=np.asarray(value).max())


@pytest.mark.parametrize(
    "value", (0.0, np.asarray(0.0), np.asarray([0.0, -1.0]), np.asarray([[0.0]]))
)
def test_require_less_than_or_equal(value):
    actual = require_less_than_or_equal(value, upper=value)
    assert actual is value

    with pytest.raises(ValidationError, match="^value must be <= "):
        require_less_than_or_equal(value, upper=np.asarray(value).max() - 1e-6)


@pytest.mark.parametrize(
    "value", ({1, 2, 3}, [4, 5, 6], "foo", {"0": 0, "1": 1, "2": 2})
)
def test_require_length_ok(value):
    actual = require_length(value, 3)
    assert actual is value
    actual = require_length_at_least(value, 2)
    assert actual is value
    actual = require_length_at_most(value, 4)
    assert actual is value
    actual = require_length_between(value, minimum=2, maximum=4)


@pytest.mark.parametrize(
    "value", ({1, 2, 3}, [4, 5, 6], "foo", {"0": 0, "1": 1, "2": 2})
)
def test_require_length_not_ok(value):
    with pytest.raises(ValidationError, match="value must have length"):
        require_length(value, 2)
    with pytest.raises(ValidationError, match="value must have length >="):
        require_length_at_least(value, 4)
    with pytest.raises(ValidationError, match="value must have length <="):
        require_length_at_most(value, 2)
    with pytest.raises(ValidationError, match="value must have length >="):
        require_length_between(value, minimum=4, maximum=7)
    with pytest.raises(ValidationError, match="value must have length <="):
        require_length_between(value, minimum=0, maximum=2)


@pytest.mark.parametrize("value", (0, True, 3.14, 1 + 2j, None))
def test_require_length_without_len(value):
    with pytest.raises(ValidationError, match="value must have a length"):
        require_length(value, 2)
    with pytest.raises(ValidationError, match="value must have a length"):
        require_length_between(value, 0, 10)


@pytest.mark.parametrize("value", ({"foo", "bar"}, ("foo", "bar")))
@pytest.mark.parametrize("required", ({"foo"}, {"bar"}, {}, None))
def test_require_contains(value, required):
    actual = require_contains(value, required=required)
    assert actual is value


@pytest.mark.parametrize("value", ({"foo", "bar"}, ("foo",), (), {}))
def test_require_contains_empty_always_validates(value):
    actual = require_contains(value, required={})
    assert actual is value


def test_import_package_returns_imported_module():
    actual = import_package("requireit")
    assert actual.__name__ == "requireit"


def test_import_package_returns_same_module_as_importlib():
    import requireit

    actual = import_package("requireit")
    assert actual is requireit


def test_import_package_raises_for_missing_package():
    with pytest.raises(ValidationError, match="not_a_package must be installed"):
        import_package("not_a_package")


@pytest.mark.parametrize("values", ([], [1], [0.0, 1.0, 2.0], [-1, 2, 7, 8], [1]))
def test_require_sorted_strict(values):
    assert_array_equal(require_sorted(values, strict=True), values)


@pytest.mark.parametrize("values", ([], [1], [0.0, 1.0, 1.0], [-1, 7, 7, 8]))
def test_require_sorted_not_strict(values):
    assert_array_equal(require_sorted(values, strict=False), values)


@pytest.mark.parametrize(
    "values, strict",
    (
        ([0.0, 1.0, 1.0], True),
        ([-1, 7, 7, 8], True),
        ([0, -1, 2], False),
    ),
)
def test_require_sorted_not_sorted(values, strict):
    msg = "strictly increasing" if strict else "non-decreasing"
    with pytest.raises(ValidationError, match=f"array must be {msg}"):
        require_sorted(values, strict=strict)


@pytest.mark.parametrize(
    "allow_cast, dtype",
    (
        (False, np.int8),
        (False, np.integer),
        (True, np.int64),
        (True, np.int32),
        (True, np.int8),
        (True, float),
    ),
)
def test_require_dtype_ok(dtype, allow_cast):
    values = np.asarray([1, 2, 3], dtype=np.int8)
    actual = require_dtype(values, dtype=dtype, allow_cast=allow_cast)
    assert actual is values


@pytest.mark.parametrize(
    "allow_cast, dtype",
    (
        (False, np.int8),
        (False, np.int32),
        (False, np.integer),
        (True, np.int64),
        (True, np.int32),
        (True, np.int8),
    ),
)
def test_require_dtype_not_ok(dtype, allow_cast):
    values = np.asarray([1, 2, 3], dtype=np.float32)
    with pytest.raises(ValidationError, match="array must have dtype"):
        require_dtype(values, dtype=dtype, allow_cast=allow_cast)


@pytest.mark.parametrize("dtype", (np.generic, np.number, np.integer, np.floating))
def test_require_dtype_concrete_type_with_allow_cast(dtype):
    with pytest.raises(TypeError, match="dtype must be a concrete dtype"):
        require_dtype([1.0, 2.0], dtype=dtype, allow_cast=True)


def test_instance():
    with pytest.raises(ValidationError, match="foo must be an instance of int, str"):
        require_instance(1.0, (int, str), name="foo")
    with pytest.raises(ValidationError, match="foo must be an instance of str"):
        require_instance(1.0, (str,), name="foo")


def test_raise_as():
    with pytest.raises(ValueError, match="foobar!"), raise_as(ValueError):
        raise ValidationError("foobar!")


def test_raise_as_with_note():
    with (
        pytest.raises(ValueError, match="foobar!") as exc_info,
        raise_as(ValueError, note="extra context"),
    ):
        raise ValidationError("foobar!")
    assert exc_info.value.__notes__ == ["extra context"]


def test_raise_as_without_note_has_no_notes():
    with pytest.raises(ValueError) as exc_info, raise_as(ValueError):
        raise ValidationError("foobar!")
    assert not getattr(exc_info.value, "__notes__", [])


def test_add_note():
    with (
        pytest.raises(ValidationError, match="foobar!") as exc_info,
        add_note("extra context"),
    ):
        raise ValidationError("foobar!")
    assert exc_info.value.__notes__ == ["extra context"]


def test_none():
    value = None
    assert require_none(value) is value
    with pytest.raises(ValidationError, match="^value must not be None"):
        require_not_none(value)


@pytest.mark.parametrize("value", (True, False, 0, "", [1, 2], np.array([1, 2])))
def test_not_none(value):
    assert require_not_none(value) is value
    with pytest.raises(ValidationError, match="^value must be None"):
        require_none(value)
