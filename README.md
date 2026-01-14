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

## API overview

All validators:

* return the original value/array on success
* raise `ValidationError` on failure

### Membership validation

```python
require_one_of(value, *, allowed)
```

Validate that a value is one of a set of allowed values.

```python
require_one_of("foo", allowed=("foo", "bar"))  # ok, returns "foo"
require_one_of("baz", allowed=("foo", "bar"))  # raises ValidationError
```

### Range validation

```python
require_between(
    value,
    a_min=None,
    a_max=None,
    *,
    inclusive_min=True,
    inclusive_max=True,
)
```

Validate that a scalar or array lies within specified bounds.

* Bounds may be inclusive or strict
* Validation fails if **any element** violates the constraint

```python
require_between([0, 1], a_min=0.0) # ok, returns [0, 1]
require_between([0, 1], a_min=0.0, inclusive_min=False) # raises
```

### Sign-based helpers

Convenience wrappers around `require_between`:

```python
require_positive(value)  # > 0
require_nonnegative(value)  # >= 0
require_negative(value)  # < 0
require_nonpositive(value)  # <= 0
```

All accept scalars or array-like inputs.


### Array validation

```python
require_array(
    array,
    *,
    dtype=None,
    shape=None,
    writable=None,
    contiguous=None,
)
```

Validate NumPy array properties without copying or modifying the array.

```python
require_array(x, dtype=np.float64, shape=(100,))
require_array(x, writable=True, contiguous=True)
```

Checks are applied only if the corresponding keyword is provided.


## Errors

All validation failures raise:

```python
requireit.ValidationError
```

This allows callers to catch validation failures distinctly from other errors.


## Contributing

This project is intentionally small.

Contributions should preserve:

* minimal surface area
* explicit semantics
* no additional dependencies

If a proposed change needs much explanation, it probably doesn’t belong here.
