#!/usr/bin/env sh

set -x

cargo t -q &&
  cargo t -q --features alloc &&
  cargo t -q --no-default-features --features alloc,unstable &&
  RUSTFLAGS="--cfg no_global_oom_handling" cargo t -q &&
  RUSTFLAGS="--cfg no_global_oom_handling" cargo t -q --features alloc &&
  RUSTFLAGS="--cfg no_global_oom_handling" cargo t -q --no-default-features --features alloc,unstable
