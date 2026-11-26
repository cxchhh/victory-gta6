# 交接：stage 5 纹理在显卡上 detile（`VICTORY_GPU_DETILE`），分支 try-detile

状态：WIP，**尚未编译、尚未测试**。用户需关机，中途停下。

## 已完成（本提交）
1. `crates/victory-formats/src/texture.rs`：新增 `Layout::coordinate_bits()`（每轴每个坐标位落到块内哪个地址位，由私有 `pattern` 表推出）+ 单测 `coordinate_bits_spread_into_the_patterns`（spread 回去等于 pattern 表）。未编译。
2. `crates/victory-gpu/src/detile.rs`（新文件，**尚未在 lib.rs 里 `mod detile;`，所以现在不参与编译**）：
   - 开关：`VICTORY_GPU_DETILE=1` / `alt`（`crate::host::alternate(6, ...)`，索引 0-3 已被占用，try-flat 可能用 4/5，所以选 6）；`VICTORY_GPU_DETILE_MB`（显存池上限，默认 256）。
   - HLSL 计算着色器 `SOURCE`：一线程写线性输出的一个 32 位字，网格覆盖整个 footprint（`RowPitch/4` 字 × rows × Depth），mip 外的字节写 0（与 CPU 路径零初始化的 staging 一致）；地址算法逐条照抄 `Layout::row/column/split/walk`（tail 偏移、块内位用 4bit/位打包在 root constants 里，thick 3D 的 slab、linear 模式）；1/2 字节元素按对齐 dword 读后移位拼接。width=0 即"全零"（未 held 的 level、尚无内容的区域）。
   - `Plan::new(layout, footprints, rows, mips, total)`：对每个子资源复刻 `Layout::walk` 的检查，任何一个 CPU 会拒绝的子资源 → 整个纹理留在 CPU 路径（once-warn + 计数 FALLBACK）。>2 GiB、XOR/MSAA 也走 CPU。
   - 平铺字节上传缓冲 `Upload`（按 2 的幂分级池，≤64 MiB 进池，空闲上限 256 MB）；`Gpu::take_tiled` 只在 `Arc::strong_count==1`（无未完成 job 持有）时原地覆写，否则换一块 —— 满足 WAR 规则。
   - **显存池（回应协调者要求：不要每纹理常驻线性缓冲）**：`region(bytes)` 从 32 MiB slab 组成的池（上限 = `VICTORY_GPU_DETILE_MB`）首次适配分配 64 KiB 粒度区域；池满或大于 slab 时单独建缓冲（计数 REGIONS[3]）；建不出来 → 计数 put_off，该 list 不录 upload（dirty 保持，下个 list 重试）。区域在 **list reset 时** 归还（此时该 list 所有执行都已完成）。日志打印在用量、峰值（自上一行 / 自启动）。
   - Job（Zeros / Detile / Copy(alt off 时 CPU detile 到 staging 后拷进区域)），`before_lists(queue, fence, value)`：把本次提交排队的 job 录进我们自己的 DIRECT list（COMMON→COPY_DEST/UAV→COMMON 屏障），在本次提交的各 list **之前单独 ExecuteCommandLists**；job 持有的资源到 fence ≥ value 才释放（`reclaim`）。
   - `log()`：每 5 秒一行池/作业统计。

## 设计要点（为什么这样）
- upload() 是在录制时录 `CopyTextureRegion` 的，而 detile 发生在提交时；texture 改变后常常是**下一次提交**的 list 才录拷贝。所以每个录了 upload 的 list 在录制时拿一块显存区域，拷贝源指向它；在该 list 提交时（`defer_staged` 步骤，必须排在 `defer_texture` **之后**，即在本次 refresh 之后）按纹理"最新内容在哪"排一个 job 填满这块区域：Nothing→全零，Tiled→我们的 detile pass（读该纹理的平铺上传缓冲），Linear→从 staging 缓冲拷贝。
- 注意 upload() 里现有的 uploaded/dirty 闭包是故意放在 `defer_texture` 之前的；新的填充步骤必须放在之后。
- 显存占用 = 尚未 reset 的 list 里的上传量（站立 ~几十 MB，移动时爆发可上百 MB，超出走单独缓冲并计数）。

## 下一步（精确）
1. `lib.rs`：`#[cfg(windows)] mod detile;`。
2. `texture.rs`：
   - 提取 `Staging::detile` 里的 CPU 循环为自由函数 `detile_into(layout, footprints, rows, mips, held, tiled, data)`（行为不变），测试也用它。
   - `Staging` 加 `gpu: Option<crate::detile::Gpu>`，在 `staging()` 里 `Gpu::new(&layout, &footprints, &rows, mips, total)`。（可选：对 gpu 纹理 staging 上传缓冲懒创建，只在 alt off 时用到，省内存；否则先保持原样最简单。）
   - `detile()`：算完 held 后，若 `gpu` 且 `on_card_now()` 且 `gpu.take_tiled(tiled, held)` → 计数 card，返回；否则原 CPU 循环，若 `gpu` 则 `gpu.set_linear()`；计数 cpu。计时加到 framelog。
   - `upload()`：在 `ensure_state` 之前，若 gpu：`region(staging.bytes)`，None → `put_off()`、`list.defer_texture(staging)`、return；拷贝源改为 `region.buffer()`、footprint.Offset += region.offset（diagnostics 的 note_read 用 region.gpu()）；在 `defer_texture` 之后加 `list.defer_staged(&staging, move || gpu.queue_fill(&region, Some(&staging_buffer)))`（闭包捕获 `Arc<Staging>` 和 `Arc<Region>`；Region 在 list reset 时随闭包 drop 归还）。两个分支（on_demand 与否）都要加。
3. `host.rs`（小钩子，另一 agent 的文件，只加两行）：
   - `submit()` 里 `prep_part(4, part); let prepared = ...` 之后、`if late {` 之前：`crate::detile::before_lists(&self.queue, &self.fence, *submitted + 1);`
   - 每 5 秒日志块（`video_memory_line();` 附近）：`crate::detile::log();`
4. `framelog.rs`：T 增加 `DetileCardTex, DetileCardBytes, DetileCpuTex, DetileCpuBytes, DetileUs`，在 "textures: ... staging N made M reused" 后追加 "detiled on the card N (X MB), on the CPU N (Y MB), Z ms detiling"。
5. 测试（detile.rs 里 `#[cfg(test)]`，仿 indirect.rs 的 WarpHarness；再加一个枚举硬件适配器的版本，无卡则跳过）：对 LINEAR、256B_S/D、4KB_S/D、64KB_S/D × 元素 1/2/4/8/16 字节 × BC1(71)/BC3(77)/BC7(98) × 奇数尺寸满 mip 链 × 3 个数组切片 × thick 3D（4KB_S、64KB_S）× held 掩码（部分 level 置零）：用 `GetCopyableFootprints` 得 footprints，随机平铺字节，CPU `detile_into` 到零缓冲；GPU：上传平铺字节、`record` 到零初始化默认缓冲、回读，整缓冲逐字节相等。另测 `record_zeros`。
6. 构建/测试（遵守 BUILD.lock 协议，`CARGO_TARGET_DIR=D:\REA\victory\target-detile`，`-j 6`，`timeout`）：`cargo test --release -p victory-gpu --features diagnostics -j 6 -- --test-threads=1`；`cargo test -p victory-formats`。
7. exe：照 target-flat\flat1 的方式 release + diagnostics 构建，拷 victory.exe/pdb 到 `D:\REA\victory\target-detile\detile1\`。

## 开放问题 / 风险
- 预期收益有限：`detile_cost` 测得 detile 进 WC 内存 0.07–0.09 ms/MB，hash 0.19 ms/MB 不变；memcpy 进 WC 估计 ~0.04–0.05 ms/MB → 站立时 11–15 ms 的 detile 估计省 5–8 ms；移动帧按比例更多。更大收益需要让卡直接读平铺字节（flat 镜像 / OpenExistingHeapFromAddress），省掉 memcpy。
- watched 路径目前是 copy_mapped→线程读缓冲→再 memcpy 进上传缓冲；可改为 copy_mapped 直接进上传缓冲省一次拷贝。
- 显存池：站立用量应在几十 MB；移动爆发时超过池上限的区域单独建缓冲（会建/释放 committed 资源，~0.25 ms/个），需要看日志峰值再定默认值（协调者建议 128–256）。
- alt off 阶段：有区域的纹理多一次卡上拷贝（staging→区域），CPU 侧与今天相同；staging 缓冲本身的 WAR（ASYNC_SUBMIT 时 CPU 在飞行中改写）是今天已有的问题，未改变。
- root SRV/UAV 无越界检查：地址正确性靠 Plan 的检查与 CPU 相同的布局算法；单测是关键。
- `on_demand`（levels on demand）纹理：held 掩码随 Latest::Tiled 保存，未 held 的 level 写 0，与 CPU 一致。
