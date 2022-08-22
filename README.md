# `acid_alloc`

[![CI](https://github.com/dataphract/acid_alloc/actions/workflows/ci.yaml/badge.svg)](https://github.com/dataphract/acid_alloc/actions)
[![crates-io](https://img.shields.io/crates/v/acid_alloc.svg)](https://crates.io/crates/acid_alloc)
[![api-docs](https://docs.rs/acid_alloc/badge.svg)](https://docs.rs/acid_alloc)

## Bare-metal allocators.

This crate provides allocators that are suitable for use on bare metal or with
OS allocation facilities like `mmap(2)`/`brk(2)`.

The following allocator types are available:

- **`Buddy`, a binary-buddy allocator**. O(log<sub>2</sub>_levels_) worst-case
  allocation and deallocation. Supports splitting and coalescing blocks by
  powers of 2. Good choice for periodic medium-to-large allocations.
- **`Bump`, a bump allocator**. O(1) allocation. Extremely fast to allocate and
  flexible in terms of allocation layout, but unable to deallocate individual
  items. Good choice for allocations that will never be deallocated or that will
  be deallocated en masse.
- **`Slab`, a slab allocator**. O(1) allocation and deallocation. All
  allocated blocks are the same size, making this allocator a good choice when
  allocating many similarly-sized objects.

## Features

All allocators provided by this crate are available in a `#![no_std]`,
`#![cfg(no_global_oom_handling)]` environment. Additional functionality is
available when enabling feature flags:

<table>
 <tr>
  <th>Flag</th>
  <th>Default?</th>
  <th>Requires nightly?</th>
  <th>Description</th>
 </tr>
 <tr><!-- sptr -->
  <td><code>sptr</code></td>
  <td>Yes</td>
  <td>No</td>
  <td>
   Uses the <a href="https://crates.io/crates/sptr"><code>sptr</code></a> polyfill for Strict Provenance.
  </td>
 </tr>
 <tr>
  <td><code>unstable</code></td>
  <td>No</td>
  <td>Yes</td>
  <td>
   Exposes constructors for allocators backed by implementors of the
   unstable <code>Allocator</code> trait, and enables the internal use of
   nightly-only Rust features. Obviates <code>sptr</code>.
  </td>
 </tr>
 <tr>
  <td><code>alloc</code></td>
  <td>No</td>
  <td>No</td>
  <td>
   Exposes constructors for allocators backed by the global allocator.
  </td>
 </tr>
</table>

[`sptr`]: https://crates.io/crates/sptr

## Acknowledgments

This crate includes stable-compatible polyfills for a number of unstable
standard-library APIs whose implementations are reproduced verbatim here. These
features are listed below along with their authors and/or maintainers:

- `alloc_layout_extra`, by [Amanieu d'Antras]
- `int_log`, by [Yoshua Wuyts]
- `strict_provenance`, by [Aria Beingessner (Gankra)]

This crate also depends on [`sptr`] (also authored by Gankra) to reproduce
strict provenance for normal pointers on stable Rust.

_If I've misattributed any of this work, or a contributor to these features is
missing, please open an issue!_

[amanieu d'antras]: https://github.com/Amanieu
[yoshua wuyts]: https://github.com/yoshuawuyts
[aria beingessner (gankra)]: https://github.com/Gankra

## License

Licensed under either of

- Apache License, Version 2.0
  ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license
  ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
