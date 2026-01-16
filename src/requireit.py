from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import DTypeLike
from numpy.typing import NDArray

__version__ = "0.2.0"


class RequireItError(Exception):
    """Base class for exceptions raised from this module."""


class ValidationError(RequireItError):
    """Error to indicate that a validation has failed."""


def require_one_of(
    value: Any, *, allowed: Iterable[Any], name: str | None = None
) -> Any:
    """Validate that ``value`` is one of ``allowed``.

    Parameters
    ----------
    value : any
        A scalar value to validate.
    allowed : iterable of any
        Allowed values.
    name : str, optional
        Variable name used in error messages.

    Returns
    -------
    any
        The original value.

    Raises
    ------
    ValidationError
        If the value is not in ``allowed``.

    Examples
    --------
    >>> require_one_of("foo", allowed=("foo", "bar"))
    'foo'
    >>> require_one_of("baz", allowed=("foo", "bar"))
    Traceback (most recent call last):
    ...
    requireit.ValidationError: value must be one of 'bar', 'foo'
    >>> require_one_of("Foo", allowed=("foo", "bar"))
    Traceback (most recent call last):
    ...
    requireit.ValidationError: value must be one of 'bar', 'foo'
    """
    name = name or "value"

    try:
        collection_of_allowed: Collection = set(allowed)
        in_collection = value in collection_of_allowed
    except TypeError:
        collection_of_allowed = list(allowed)
        in_collection = value in collection_of_allowed

    if not in_collection:
        allowed_str = ", ".join(sorted(repr(x) for x in collection_of_allowed))
        raise ValidationError(f"{name} must be one of {allowed_str}")
    return value


def require_between(
    value: ArrayLike,
    a_min: float | None = None,
    a_max: float | None = None,
    *,
    inclusive_min: bool = True,
    inclusive_max: bool = True,
    name: str | None = None,
) -> ArrayLike:
    """Validate that a value lies within a specified interval.

    Parameters
    ----------
    value : scalar or array-like
        The input value(s) to be validated.
    a_min : float or None, optional
        Minimum allowable value. If ``None``, no lower bound is applied.
    a_max : float or None, optional
        Maximum allowable value. If ``None``, no upper bound is applied.
    inclusive_min : bool, optional
        If ``True`` (default), the lower bound is inclusive (``>=``).
        If ``False``, the lower bound is strict (``>``).
    inclusive_max : bool, optional
        If ``True`` (default), the upper bound is inclusive (``<=``).
        If ``False``, the upper bound is strict (``<``).
    name : str, optional
        Variable name used in error messages.

    Returns
    -------
    value : scalar or array-like
        The validated value.

    Raises
    ------
    ValidationError
        If any element of ``value`` violates the specified bounds.

    Examples
    --------
    >>> require_between([0, 1], a_min=0.0, inclusive_min=True)
    [0, 1]
    >>> require_between([0, 1], a_min=0.0, inclusive_min=False)
    Traceback (most recent call last):
    ...
    requireit.ValidationError: value must be > 0.0
    """
    name = name or "value"

    arr = np.asarray(value)

    if a_min is not None:
        cmp: Callable = np.less if inclusive_min else np.less_equal
        op = ">=" if inclusive_min else ">"
        if np.any(cmp(arr, a_min)):
            raise ValidationError(f"{name} must be {op} {a_min}")

    if a_max is not None:
        cmp = np.greater if inclusive_max else np.greater_equal
        op = "<=" if inclusive_max else "<"
        if np.any(cmp(arr, a_max)):
            raise ValidationError(f"{name} must be {op} {a_max}")

    return value


def require_positive(value: ArrayLike, name: str | None = None) -> ArrayLike:
    """Validate that a value is strictly greater than zero.

    Parameters
    ----------
    value : scalar or array-like
        The input value(s) to be validated.
    name : str, optional
        Variable name used in error messages.

    Returns
    -------
    value : scalar or array-like
        The validated value.

    Raises
    ------
    ValidationError
        If any element of ``value`` is less than or equal to zero.

    Examples
    --------
    >>> require_positive(1.0)
    1.0
    >>> require_positive(0.0)
    Traceback (most recent call last):
    ...
    requireit.ValidationError: value must be > 0.0
    """
    return require_between(value, a_min=0.0, a_max=None, inclusive_min=False, name=name)


def require_nonnegative(value: ArrayLike, name: str | None = None) -> ArrayLike:
    """Validate that a value is greater than or equal to zero.

    Parameters
    ----------
    value : scalar or array-like
        The input value(s) to be validated.
    name : str, optional
        Variable name used in error messages.

    Returns
    -------
    value : scalar or array-like
        The validated value.

    Raises
    ------
    ValidationError
        If any element of ``value`` is less than zero.

    Examples
    --------
    >>> require_nonnegative(-1.0)
    Traceback (most recent call last):
    ...
    requireit.ValidationError: value must be >= 0.0
    >>> require_nonnegative(0.0)
    0.0
    >>> require_nonnegative(1.0)
    1.0
    """
    return require_between(value, a_min=0.0, a_max=None, inclusive_min=True, name=name)


def require_negative(value: ArrayLike, name: str | None = None) -> ArrayLike:
    """Validate that a value is strictly less than zero.

    Parameters
    ----------
    value : scalar or array-like
        The input value(s) to be validated.
    name : str, optional
        Variable name used in error messages.

    Returns
    -------
    value : scalar or array-like
        The validated value.

    Raises
    ------
    ValidationError
        If any element of ``value`` is greater than or equal to zero.

    Examples
    --------
    >>> require_negative(-1.0)
    -1.0
    >>> require_negative(0.0)
    Traceback (most recent call last):
    ...
    requireit.ValidationError: value must be < 0.0
    """
    return require_between(value, a_min=None, a_max=0.0, inclusive_max=False, name=name)


def require_nonpositive(value: ArrayLike, name: str | None = None) -> ArrayLike:
    """Validate that a value is less than or equal to zero.

    Parameters
    ----------
    value : scalar or array-like
        The input value(s) to be validated.
    name : str, optional
        Variable name used in error messages.

    Returns
    -------
    value : scalar or array-like
        The validated value.

    Raises
    ------
    ValidationError
        If any element of ``value`` is greater than zero.

    Examples
    --------
    >>> require_nonpositive(-1.0)
    -1.0
    >>> require_nonpositive(0.0)
    0.0
    >>> require_nonpositive(1.0)
    Traceback (most recent call last):
    ...
    requireit.ValidationError: value must be <= 0.0
    """
    return require_between(value, a_min=None, a_max=0.0, inclusive_max=True, name=name)


def require_array(
    array: NDArray,
    *,
    dtype: DTypeLike | None = None,
    shape: tuple[int, ...] | None = None,
    writable: bool | None = None,
    contiguous: bool | None = None,
    name: str | None = None,
):
    """Validate an array to satisfy requirements.

    Parameters
    ----------
    array : ndarray
        The array to be validated.
    dtype : data-type, optional
        The required data type.
    shape : tuple of int, optional
        The required shape.
    writable : bool, optional
        Require the array to be writable.
    contiguous : bool, optional
        Require the array to be c-contiguous.
    name : str, optional
        Variable name used in error messages.

    Returns
    -------
    array : ndarray
        The validated array.

    Raises
    ------
    ValidationError
        If the array is invalid.

    Examples
    --------
    >>> import numpy as np
    >>> require_array(np.array([1, 2, 3, 4]), shape=(2, 2))
    Traceback (most recent call last):
    ...
    requireit.ValidationError: array must have shape (2, 2)
    """
    name = name or "array"

    if shape is not None and array.shape != shape:
        raise ValidationError(f"{name} must have shape {shape}")

    if dtype is not None and array.dtype != np.dtype(dtype):
        raise ValidationError(f"{name} must have dtype {dtype}")

    if writable and not array.flags.writeable:
        raise ValidationError(f"{name} must be writable")

    if contiguous and not array.flags.c_contiguous:
        raise ValidationError(f"{name} must be contiguous")

    return array


def require_path_string(path: Any, *, name: str | None = None) -> str:
    """Validate that a value is a string intended to be used as a path.

    This function verifies only that the provided value is a non-empty string
    that doesn't contain null characters. It does *not* check whether the
    path exists, is readable, is writable, or is valid on any particular
    operating system.

    Parameters
    ----------
    path : any
        Value to validate.
    name : str, optional
        Variable name used in error messages.

    Returns
    -------
    str
        The original value, guaranteed to be a string that could possibly
        be used as a file path.

    Raises
    ------
    ValidationError
        If ``path`` cannot be used as a file path.

    Examples
    --------
    >>> require_path_string("data/input.txt")
    'data/input.txt'
    >>> require_path_string("")
    Traceback (most recent call last):
    ...
    requireit.ValidationError: path must not be empty
    """
    name = name or "path"

    if not isinstance(path, str):
        raise ValidationError(f"{name} must be a string")
    if path == "":
        raise ValidationError(f"{name} must not be empty")
    if "\x00" in path:
        raise ValidationError(f"{name} must not contain null characters")

    return path


def require_less_than(
    value: Any,
    upper: Any,
    *,
    inclusive: bool = False,
    name: str | None = None,
) -> Any:
    """Require that a value is less than (or equal to) an upper bound.

    For arrays, this requirement fails if **any** element violates the
    constraint.

    Parameters
    ----------
    value : any
        Value to validate.
    upper : any
        Upper bound to compare against.
    inclusive : bool, optional
        If ``True``, require ``value <= upper``. Otherwise, require ``value < upper``.
    name : str, optional
        Variable name used in error messages.

    Returns
    -------
    any
        The original value.

    Raises
    ------
    ValidationError
        If the requirement is not satisfied.

    Examples
    --------
    >>> require_less_than(1.0, 2.0)
    1.0
    >>> require_less_than(2.0, 2.0)
    Traceback (most recent call last):
    ...
    requireit.ValidationError: value must be < 2.0
    >>> require_less_than(2.0, 2.0, inclusive=True)
    2.0
    """
    name = name or "value"

    value_array = np.asarray(value)
    upper = np.asarray(upper)

    if inclusive:
        violated = np.greater(value_array, upper)
        op = "<="
    else:
        violated = np.greater_equal(value_array, upper)
        op = "<"

    if np.any(violated):
        upper_str = upper.item() if upper.size == 1 else "upper"
        raise ValidationError(f"{name} must be {op} {upper_str}")

    return value
