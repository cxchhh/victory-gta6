# Open bug: `v_writelane_b32` is dropped when its lane is outside EXEC

Status: reproduced, not fixed. Written so that a person or a coding agent can pick it up.

## What is wrong

On the console `v_writelane_b32 vN, sX, K` writes lane K whatever EXEC is. The translator
lowers it as an ordinary vector write, under the guard of the lane that runs it
(`crates/victory-shader/src/emit/lower.rs`, the `"v_writelane_b32"` arm, then
`write_placed`). So when lane K is outside EXEC, or does not reach the block, the write is
lost and a later `v_readlane_b32 sY, vN, K` gives the old value.

The compiler uses this pair to park scalars in VGPR lanes. `ir::unspill`
(`crates/victory-shader/src/ir.rs`) turns the pairs into scalar moves, but only for a VGPR
whose every use is such a pair with a constant lane. A VGPR with any other use keeps its
lane writes, and those are the ones affected.

## Reproduce

```
cargo test -p victory-shader --lib a_lane_write_reaches_its_lane_outside_exec -- --ignored --nocapture
```

The test is in `crates/victory-shader/src/emit/wave64_tests.rs`. It needs a hardware device
with shader model 6.6; without one it prints `not run` and passes, which is not a result.
On Windows 10 the in-box D3D12 stops at 6.5 (PR #69 adds the newer runtime).

Program: one console wave of 64, `if (t > s0) { v_writelane_b32 v9, s6, 0 }` with s6 = 5 and
v9 = 7 before, then every thread stores `v_readlane_b32` of lane 0.

| s0 | threads inside | expected | got (RTX 3090, 32-lane waves) |
|----|----------------|----------|-------------------------------|
| 10 | 11..63, lane 0 outside | 5 | 7 |
| 62 | 63, lane 0 outside | 5 | 7 |
| 63 | none (the wave skips) | 7 | 7 |

The same in all three forms: as is, `ir::without_skips(p, false)`, `ir::without_skips(p, true)`.
The control with lane 0 inside (`t < s0`) gives 5 and 7 as it should.

"Expected" is the instruction's documented behaviour (it ignores EXEC), which `unspill`
already relies on. It has not been measured on a console.

## What a fix must do

Lane K takes the scalar exactly when the console wave executes the instruction:

* some lane of the console wave is in EXEC there: write, even if lane K is not;
* no lane is (the console took the `s_cbranch_execz` skip): do not write. Making the write
  unconditional breaks the `s0 = 63` row in the forms without skips.

Two ways, cheapest first:

1. At translation time: let `unspill` take a parking VGPR that has other uses when the
   parked lanes' live ranges do not overlap those uses. No cost per frame, and the program
   may then not need 64-lane waves at all.
2. At run time: write under "some lane of the console wave is in EXEC" (a ballot; across
   two host waves an exchange, which costs group barriers). Only where 1 cannot apply.

## Done when

* the test passes without `#[ignore]` on a 32-lane card and on a 64-lane one;
* `cargo test --workspace` passes;
* `victory translate --validate` over a title's shaders translates no fewer programs than before.

## Not known yet

Whether this is what makes the 14 `CS_ShadowRenderTiled*` programs wrong in the "without
skips, scalars followed" form (issue #62). To check: `VICTORY_HOST_WAVE=32 VICTORY_WAVE_WHY=1
victory translate --list` and see whether those programs name `v_writelane` / `v_readlane`.

Ruled out for #62: a loop running to the loop limit inside a region whose skip was left
out. A conditional back edge leaves the loop at the limit and the program goes on.
