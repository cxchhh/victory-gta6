# The memory layer rewrite: title memory as the card's own

Status: in progress on branch `try-mirror` (integration) with `try-flat` (stage 2) and
`try-lazy` (stage 1). Every stage sits behind a switch; off is today's path, byte for byte.

## Why

GTA VI at 2K (Series X profile) on an RTX 4070, world, standing still: about 187 ms a
frame. The card works 40-60 ms of it. The rest is the layer emulating the console's one
memory on a PC's two:

* every title buffer has a host copy of its own (`Shadow`, about 4,400 copies, 4.2 GB, plus
  windows on a 4 MiB grid), found per bind through a global table;
* before every submission the copies are refreshed from title memory: 270,000 refreshes
  over 45,000 distinct 64 KiB pages (2.8 GB) a frame, 120-150 MB copied, 50-60 ms;
* after the card runs, what shaders wrote is compared, fetched and merged back (about
  55 MB, 27-30 ms), although the title reads under 0.5% of it (`VICTORY_BACK_READS`);
* 10,000-19,000 constant buffers a frame are copied one by one into upload scratch;
* descriptor views are made again per draw; the recording threads waited about 450 ms a
  frame together on our locks (`VICTORY_PROFILE`).

Measured census (`VICTORY_FLAT=census`, run trace2): 7 slots of 4 GiB used, no ask crosses
a 4 GiB line, a mirror would send 175 MB a frame (today 156 MB: the bytes are alike, the
cost is the 270,000 calls and the locks), 330 pages a frame written by both CPU and shaders
in one submission, 2.2 GB read and never written by shaders. The card has about 2.3 GB
of its budget free: a mirror must map tiles on first use and fall back when full.

## The design, as an engine would have it

| Engine practice | Here today | The rewrite | Expected (ms/frame) | Stage |
|---|---|---|---|---|
| Resources resident in video memory, residency by budget | a PC-side copy per buffer, refreshed per submission | one VRAM mirror of title GPU memory: reserved windows, 64 KiB tiles mapped on first use from a capped pool, address = window + offset, no lookup, no lock | 30-50 | 2 (`VICTORY_FLAT=1`) |
| GPU output not read back unless the CPU needs it | compare, fetch, merge after every submission | pages shaders wrote stay in the mirror; title pages protected; a CPU touch faults and downloads | 25-40 | 2b |
| Several frames in flight | many "wait for the card, then read" points | once reads back are gone, only present synchronises | 30-40 | after 2b |
| Persistently mapped upload ring | constant buffers filled one by one | constants read where the mirror holds them (dirty pages go up in one batch, ordered on the queue) | 10-20 | 3 (`VICTORY_FLAT_CBV`) |
| Bindless: descriptors made once, indices only | views made per draw under a lock | views made when the title writes them, buffer views pointing into the mirror; shaders address memory by window arithmetic | 10-20 | 4 (`VICTORY_FLAT_VIEWS`) |
| Async compute, a copy queue | one DIRECT queue for everything | title compute on a host compute queue; mirror and texture uploads on a copy queue | 5-15 (card) | later |
| Streaming on a budget, baked assets | whole textures detiled on the CPU on the submitting thread | detile on the card from the mirror; detiled textures cached by content | 10+ (more when moving) | 5 |
| Job system, no global lock on hot paths | recording threads ~45% waiting | per-slot descriptor locks (done), lookups by arithmetic (stage 2) | 10-30 | partly done |

Rule the stages keep: the mirror is written only by copies recorded on the queue ahead of
the submission's lists, never by the CPU into memory the card reads. Splitting constant
buffers in place over host copies (`VICTORY_CBV_SPLIT_IN_PLACE`, 5089807e) drew black
start screens, most likely because a later submission's CPU-side refresh overwrote
constants an earlier, not yet executed submission still had to read.

## Stages and their state

| Stage | Switch | Branch | State |
|---|---|---|---|
| 0 census | `VICTORY_FLAT=census` | try-flat 10b86ef4, merged | numbers above |
| 1 lazy copy-back (pieces kept, title pages `PAGE_NOACCESS`, merged on touch) | `VICTORY_LAZY_BACK=1`/`alt` | try-lazy, merged | works (faults: 0.3 MB a frame) but no gain: lazy1 186.9 ms off, 200.3 ms on; only a third of pieces qualify (view memory is excluded) and the fetch and compare stay. Its fault machinery is reused by 2b. Off. |
| 2 flat mirror, CPU to card | `VICTORY_FLAT=1` | try-flat | in progress (2 GiB windows: WARP fails mapping past ~2 GiB) |
| 2b GPU writes stay on the card | | | to do |
| 3 constants in the mirror | `VICTORY_FLAT_CBV=1` | | to do |
| 4 views made once, flat shader addressing | `VICTORY_FLAT_VIEWS=1` | | to do |

Also on `try-mirror`: per-slot locks for `HostHeap` (`made`, `kept`), lock-free
`HostHeap::mirror` for unchanged slots (`VICTORY_MIRROR_FAST`), the lookup cache
(`VICTORY_LOOKUP_CACHE`), range-only remap waits and parallel texture refreshes
(`VICTORY_REMAP_RANGES`, `VICTORY_WIDE_TEXTURES`). The last four measured 218 -> 189 ms in
one run (ab1); the hitches when moving were 875 of 919 ms waiting on the remap lock.

Tried and dropped (numbers in the commits): a copy's refresh skipping bytes it stored
itself (keep1, 201 ms vs 194), merge-run cap 65,536 (fill2, +26 ms on the card),
`CARD_STEPS` (steps1, 219 ms), thin LTO and `target-cpu=native` (replay: under 1%; the
CPU time is system calls, copies and locks, not code generation).

## How each stage is checked

1. Offline, no title: unit tests (WARP and plain write-watched memory), then the trace
   replay: `VICTORY_TRACE=<folder>` in one run records title buffers, asked and written
   ranges, submissions and the bytes the layer copied; `victory trace-replay <folder>
   --passes 2 --verify` replays them on the real card, times asks, submissions and copies,
   and checks the asked ranges against title memory (trace2: 430 submissions, 9.7 s a
   pass, 15,506 ranges checked, 0 wrong). `abreplay.sh` alternates two exes on one trace.
   The replay does not run the title's shaders: GPU writes are not covered.
2. In the title, one run of about 220 s with the switch set to `alt`: the switches in turn
   change together every 20 s (`crate::host::alternate`), and `altstats.py` splits the
   frame lines by phase, so both sides see the same scene at the same moment.
3. `VICTORY_PROFILE=1 VICTORY_PROFILE_AFTER=150 VICTORY_PROFILE_FOR=30` for 30 s of world
   frames when a stage does not move the frame time as expected.

Builds never run beside the title (it takes about 23 of 32 GB): a lock file guards both.
