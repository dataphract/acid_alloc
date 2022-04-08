# `acid_alloc`

## Bare metal-friendly allocators.

This crate provides allocators that can function without the backing of another
allocator. This makes them suitable for use on bare metal or with OS allocation
facilities like `memmap(2)`/`brk(2)`.

The following allocator types are available:

- `Buddy`, a binary-buddy allocator

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
  <td>`sptr`</td>
  <td>Yes</td>
  <td>No</td>
  <td>
   Uses the [`sptr`] polyfill for Strict Provenance.
  </td>
 </tr>
 <tr>
  <td>`unstable`</td>
  <td>No</td>
  <td>Yes</td>
  <td>
   Exposes `Allocator` implementations for this crate's allocators, and enables
   the internal use of nightly-only Rust features. Obviates `sptr`.
  </td>
 </tr>
 <tr><!-- Requires nightly? -->
  <td>`alloc`</td>
  <td>No</td>
  <td>No</td>
  <td>
   Exposes constructors for allocators backed by the global allocator.
  </td>
 </tr>
 <tr><!-- Description -->
  <td>`std`</td>
  <td>No</td>
  <td>No</td>
  <td>
   Exposes additional wrapper types for allocators using the standard `Mutex`
   and `RwLock`.
  </td>
 </tr>
</table>

[`sptr`]: https://crates.io/crates/sptr

## Acknowledgments

This crate includes stable-compatible polyfills for a number of unstable
standard-library APIs whose implementations are reproduced verbatim here. These
features are listed below along with their authors and/or maintainers:

- `alloc_layout_extra`, by [Amanieu d'Antras]
- `int_log`, by [yoshuawuyts]
- `strict_provenance` for `NonNull<T>`, by [Aria Beingessner (aka Gankra)]

This crate also depends on [`sptr`] (also authored by Gankra) to reproduce
strict provenance for normal pointers on stable Rust.

_If I've misattributed any of this work, or a contributor to these features is
missing, please open an issue!_

[library api team]: https://www.rust-lang.org/governance/teams/library#Library%20API%20team
[amanieu d'antras]: https://github.com/Amanieu
[yoshuawuyts]: https://github.com/yoshuawuyts
[aria beingessner (aka gankra)]: https://github.com/Gankra

## License

Licensed under either of

- Apache License, Version 2.0
  ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license
  ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
