# `acid_alloc`

### Bare metal-friendly allocators.

This crate provides allocators that can function without the backing of another
allocator. This makes them suitable for use on bare metal or with low-level OS
allocation facilities like `brk(2)`.

### Acknowledgments

This crate includes stable-compatible polyfills for a number of unstable APIs
whose implementations are reproduced verbatim here. These features are listed
below along with their authors and/or maintainers:

- `alloc_layout_extra`, by [Amanieu d'Antras]
- `int_log`, by [yoshuawuyts]
- `strict_provenance`, by [Aria Beingessner (aka Gankra)]

This crate also depends on [`sptr`] (also authored by Gankra) to reproduce
strict provenance on stable Rust.

_If I've misattributed any of this work, or a contributor to these features is missing, please open an issue!_

[library api team]: https://www.rust-lang.org/governance/teams/library#Library%20API%20team
[amanieu d'antras]: https://github.com/Amanieu
[yoshuawuyts]: https://github.com/yoshuawuyts
[aria beingessner (aka gankra)]: https://github.com/Gankra
[`sptr`]: https://crates.io/crates/sptr

### License

Licensed under either of

- Apache License, Version 2.0
  ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license
  ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
