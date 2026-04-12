# Release Notes


## 0.7.0 (unreleased)

### Features

* Added `require_sorted` validator that checks if values are sorted in
  ascending order. [#35](https://github.com/mcflugen/requireit/issues/35)
* Added `require_dtype` validator that checks if values are of a given
  dtype or, optionally, can be safely cast to that dtype.
  [#36](https://github.com/mcflugen/requireit/issues/36)

### Changes

* Dropped Python 3.10 support. [#37](https://github.com/mcflugen/requireit/issues/37)

## 0.6.0 (2026-04-01)

### Features

* Allow the `dtype` keyword of `require_array` to accept *nupy* dtype families such as
  `np.integer` and `np.floating` in addition to exact dtypes.

## 0.5.0 (2026-03-28)

### Features

* Extended ``require_array`` to allow flexible shape validation with support
  for wildcard dimensions (None or named axes).

## 0.4.0 (2026-03-27)

### Features

* Added `require_greater_than`, `require_greater_than_or_equal`, and
  `require_less_than_or_equal` validators.
  [#26](https://github.com/mcflugen/requireit/issues/26)

## 0.3.0 (2026-03-23)

### Features

* Added `require_not_one_of` validator to ensure a value is not in a forbidden set
  [#15](https://github.com/mcflugen/requireit/issues/15)
* Added length validators to check that an object’s length is exactly, at most, or at least
  a given value [#16](https://github.com/mcflugen/requireit/issues/16)
* Added `require_length_between` validator to check that an object’s length is within
  a specified range [#19](https://github.com/mcflugen/requireit/issues/19)
* Added `require_contains` validator to ensure a collection contains required values
  [#21](https://github.com/mcflugen/requireit/issues/21)
* Added `import_package` validator to check for and import a package
  [#22](https://github.com/mcflugen/requireit/issues/22)
* Added `argparse_type` to allow *requireit* validators to be used as
  `argparse` `type=` callables
  [#17](https://github.com/mcflugen/requireit/issues/17)

### Changes

* Renamed length validators for consistency:
  `require_length_is` → `require_length`,
  `require_length_is_at_least` → `require_length_at_least`,
  `require_length_is_at_most` → `require_length_at_most`
  [#20](https://github.com/mcflugen/requireit/issues/20)

### Tests

* Added unit tests to verify that validators return the input value (not a copy)
  on success [#18](https://github.com/mcflugen/requireit/issues/18)


## 0.2.0 (2026-01-16)

- Standardized validation error messages [#8](https://github.com/mcflugen/requireit/issues/8)
- Renamed ``validate_array`` to ``require_array`` [#9](https://github.com/mcflugen/requireit/issues/9)
- Added optional ``name`` keyword to require functions to make error messages
  easier to read [#10](https://github.com/mcflugen/requireit/issues/10)
- Added new validator, ``require_path_string``, that checks if a value could be used
  as a file path [#11](https://github.com/mcflugen/requireit/issues/11)
- Added new validator, ``require_less_than``, that checks if one value is less than
  another [#12](https://github.com/mcflugen/requireit/issues/12)

## 0.1.0 (2026-01-12)

- Added documentation to the README [#1](https://github.com/mcflugen/requireit/issues/1)
- Added project metadata files [#2](https://github.com/mcflugen/requireit/issues/2)
- Added the `requireit` module [#3](https://github.com/mcflugen/requireit/issues/3)
- Added `pyproject.toml` file [#4](https://github.com/mcflugen/requireit/issues/4)
- Added `noxfile.py` file and linters [#5](https://github.com/mcflugen/requireit/issues/5)
- Added unit tests for `requireit` [#6](https://github.com/mcflugen/requireit/issues/6)
- Added GitHub Actions for CI [#7](https://github.com/mcflugen/requireit/issues/7)
