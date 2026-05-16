# requireit

**Tiny, numpy-aware runtime validators for explicit precondition checks.**

`requireit` provides a small collection of lightweight helper functions such as
`require_positive`, `require_between`, and `require_array` for validating values
and arrays at runtime.

It is intentionally minimal and dependency-light (*numpy* only).

## Why `requireit`?

* **Explicit** – reads clearly
* **numpy-aware** – works correctly with scalars *and* arrays
* **Fail-fast** – raises immediately with clear error messages
* **Lightweight** – just a bunch of small functions
* **Reusable** – avoids copy-pasted validation code across projects

```python
from requireit import require_one_of
from requireit import require_positive

require_positive(dt)
require_one_of(method, allowed={"foo", "bar"})
```

## Design principles

* Prefer small, single-purpose functions
* Raise standard exceptions (`ValidationError`)
* Never coerce or "fix" invalid inputs
* Validate *all* elements for array-like inputs
* Keep the public API small

## Non-goals

`requireit` is **not**:

* a schema or data-modeling system
* a replacement for static typing
* a validation framework
* a substitute for unit tests
* a coercion or parsing library

If you need structured validation, transformations, or user-facing error
aggregation, you probably want something heavier.

## Installation

```bash
pip install requireit
```

## API Summary

All validators:

* validate the first argument
* return the original value/array on success
* raise `ValidationError` on failure

### Arrays

* `require_array`: Validate an array to satisfy requirements.
* `require_dtype`: Validate that an array has a required dtype or can be safely cast to it.
* `require_like`: Validate that an array has the same shape and/or dtype as another.
* `require_ndim`: Validate that an array has a specific number of dimensions.
* `require_shape`: Validate that an array has the specified shape.
* `require_sorted`: Validate that an array is sorted.

### General

* `require_contains`: Require `collection` contains required values.
+ `require_instance`: Require `value` is an instance of one or more types.
* `require_not_one_of`: Require `value` is not contained in `forbidden`
* `require_one_of`: Require `value` is contained in `allowed`

### Length

* `require_length`: Require `len(value) == length`
* `require_length_at_least`: Require `len(value) >= length`
* `require_length_at_most`: Require `len(value) <= length`
* `require_length_between`: Require `len(value)` falls within a specified range.

### Numeric

* `require_between`: Validate that a value lies within a specified interval.
* `require_greater_than`: Require `value > lower`
* `require_greater_than_or_equal`: Require `value >= lower`
* `require_less_than`: Require `value < upper`
* `require_less_than_or_equal`: Require `value <= upper`
* `require_negative`: Require `value < 0`
* `require_nonnegative`: Require `value >= 0`
* `require_nonpositive`: Require `value <= 0`
* `require_positive`: Require `value > 0`

### Paths

* `require_path_string`: Validate that a value is a string intended to be used as a path.

### Command-line integration

* `argparse_type`: Adapt a requireit validator for use as an argparse `type=` callable.

```python
import argparse
from requireit import argparse_type, require_positive


def parse_positive_int(value: str) -> int:
    return require_positive(int(value))


parser = argparse.ArgumentParser()
parser.add_argument("--count", type=argparse_type(parse_positive_int))
```

Converts `ValidationError` into `argparse.ArgumentTypeError`, allowing requireit
validators to produce clean command-line error messages.


## Errors

All validation failures raise:

```python
requireit.ValidationError
```

This allows callers to catch validation failures distinctly from other errors.
To adapt validation errors to another exception type, use `raise_as`:

```python
from requireit import raise_as
from requireit import require_positive

with raise_as(ValueError):
    require_positive(-1)
```

This raises:

```python
ValueError: value must be positive
```
Useful when integrating *requireit* into APIs that already expose a specific
exception type.


## Contributing

This project is intentionally small.

Contributions should preserve:

* minimal surface area
* explicit semantics
* no additional dependencies

If a proposed change needs much explanation, it probably doesn’t belong here.
