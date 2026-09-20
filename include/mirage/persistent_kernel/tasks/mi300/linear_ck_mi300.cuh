/* Copyright 2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm.hpp"
#include "ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v2.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp"
#include "ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp"
#include "tasks/common/common_header.cuh"

namespace kernel {

// ============================================================================
// Non-temporal store helpers for MALL cache optimization (NT=1 → MALL: no-allocate)
// Transient output data should not pollute MALL; weights should stay resident.
// On GFX942/GFX950: NT bit controls MALL allocation policy:
//   NT=0 (default): TCP/TCC=LRU, MALL=allocate
//   NT=1 (non-temporal): TCP/TCC=evict/stream, MALL=no-allocate
// ============================================================================
__device__ __forceinline__ void nt_store_u64(void* addr, uint64_t val) {
    __builtin_nontemporal_store(val, reinterpret_cast<uint64_t*>(addr));
}

__device__ __forceinline__ void nt_store_bf16(ck_tile::bf16_t* addr, ck_tile::bf16_t val) {
    __builtin_nontemporal_store(val, addr);
}

using bf16 = ck_tile::bf16_t;

// ============================================================================
// Layout-independent traversal of a CK distributed C tile.
// Sweeps every distributed element of the block accumulator and calls
//   fn(row_within_block, col_within_block, value)
// where (row_within_block, col_within_block) are coordinates inside the
// [MPerBlock, NPerBlock] block tile (NOT the global [M, N] position), and
// value is the float32 accumulator element held by the current thread.
// This is independent of the warp gemm C register layout (MFMA or MMAC), so
// atomic/scatter epilogues no longer need hand-coded lane/register mappings.
// ============================================================================
template <typename CBlockTile, typename F>
CK_TILE_DEVICE void for_each_c_element(CBlockTile& c_block_tile, F&& fn)
{
    using CBlockTileT = ck_tile::remove_cvref_t<CBlockTile>;
    using TileDstr    = typename CBlockTileT::StaticTileDistribution;

    constexpr auto tile_dstr = TileDstr{};

    // Traverse every distributed index of the 2D [M, N] block tile.
    ck_tile::sweep_tile<CBlockTileT>([&](auto idx) {
        constexpr auto distributed_indices = ck_tile::make_tuple(
            idx[ck_tile::number<0>{}], idx[ck_tile::number<1>{}]);
        const auto x_indices =
            ck_tile::get_x_indices_from_distributed_indices(tile_dstr, distributed_indices);
        const ck_tile::index_t row = x_indices[ck_tile::number<0>{}];
        const ck_tile::index_t col = x_indices[ck_tile::number<1>{}];
        fn(row, col, c_block_tile(distributed_indices));
    });
}


// Promote a device pointer to wave-uniform (SGPR) representation.
// On AMDGPU, buffer_load requires the base address in an SGPR buffer resource
// descriptor. When the compiler cannot prove a pointer is uniform across the
// wave, it emits a v_readfirstlane "waterfall" loop for every buffer_load —
// adding ~40% overhead to the GEMM K-loop. In the persistent kernel all threads
// in a workgroup execute the same task, so pointers are identical across lanes;
// this intrinsic makes that explicit.
__device__ __forceinline__ uintptr_t __uniform_addr(const void* ptr) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    uint32_t lo = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(addr));
    uint32_t hi = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(addr >> 32));
    return static_cast<uintptr_t>(hi) << 32 | static_cast<uintptr_t>(lo);
}

// ============================================================================
// Custom block gemm policy for small batch sizes using MMAC (gfx938/DCU)
// Uses MWarp=1, NWarp=4 configuration to allow MPerBlock=16, NPerBlock=64
// This minimizes padding overhead for small batches (batch_size=8 -> 2x padding)
// MMAC bf16 warp tile is fixed at kM=16 (16x16x32 for the small tile).
// ============================================================================
struct BlockGemmSmallM16Policy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        // Use 16x16x32 MMAC with transposed C distribution
        // MWarp=1 allows MPerBlock=16 (minimal padding for batch=8)
        // NWarp=4 to maintain parallelism across 4 warps (256 threads)
        return ck_tile::make_tuple(
            ck_tile::WarpGemmMmacBF16BF16F32_WT16x16x32_MR1NR1MI1NI1_TRANSC{},
            1,  // MWarp = 1 -> MPerBlock >= 16
            4   // NWarp = 4 -> NPerBlock >= 64
        );
    }
};

// Default block gemm policy using 16x32x64 MMAC
// MWarp=2, NWarp=2 -> MPerBlock=128, NPerBlock=64 (with MIter/NIter)
struct BlockGemmDefaultPolicy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return ck_tile::make_tuple(
            ck_tile::WarpGemmMmacBF16BF16F32_WT16x32x64_MR1NR2MI1NI1_TRANSC{},
            2,  // MWarp
            2   // NWarp
        );
    }
};

// 2x2 warp grid policy using 16x32x64 MMAC
// MWarp=2, NWarp=2 -> MPerBlock=64, NPerBlock=64 (MIter=2, NIter=1)
struct BlockGemm2x2Policy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return ck_tile::make_tuple(
            ck_tile::WarpGemmMmacBF16BF16F32_WT16x32x64_MR1NR2MI1NI1_TRANSC{},
            2,  // MWarp = 2 -> M = 2 * 16 * MIter
            2   // NWarp = 2 -> N = 2 * 32 * NIter
        );
    }
};

// Large tile policy for bigger batch sizes using 16x32x64 MMAC
// MWarp=1, NWarp=4 -> MPerBlock=32, NPerBlock=128
struct BlockGemmLargeM32Policy
{
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetWarpGemmMWarpNWarp()
    {
        return ck_tile::make_tuple(
            ck_tile::WarpGemmMmacBF16BF16F32_WT16x32x64_MR1NR2MI1NI1_TRANSC{},
            1,  // MWarp = 1 -> MPerBlock = 32 (MIter=2)
            4   // NWarp = 4 -> NPerBlock = 128 (NIter=1)
        );
    }
};

// ============================================================================
// Custom LDS descriptor for small M tiles (MPerBlock=16)
// Adapted from CK's default policy with smaller tile support
// ============================================================================
template <ck_tile::index_t MPerBlock_, ck_tile::index_t NPerBlock_, ck_tile::index_t KPerBlock_>
struct GemmPipelineSmallTilePolicy
{
    static constexpr ck_tile::index_t MPerBlock = MPerBlock_;
    static constexpr ck_tile::index_t NPerBlock = NPerBlock_;
    static constexpr ck_tile::index_t KPerBlock = KPerBlock_;

    // LDS descriptor for A matrix [M, K] with padding to avoid bank conflicts
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeALdsBlockDescriptor()
    {
        using namespace ck_tile;
        // 3D layout with padding: [K/8, M, 8] with stride (M+1)*8 on K/8 dimension
        constexpr auto a_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<KPerBlock / 8>{}, number<MPerBlock>{}, number<8>{}),
            make_tuple(number<(MPerBlock + 1) * 8>{}, number<8>{}, number<1>{}),
            number<8>{},
            number<1>{});

        constexpr auto a_lds_block_desc = transform_tensor_descriptor(
            a_lds_block_desc_0,
            make_tuple(make_pass_through_transform(MPerBlock),
                       make_merge_transform(make_tuple(KPerBlock / 8, 8))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return a_lds_block_desc;
    }

    // LDS descriptor for B matrix [N, K]
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBLdsBlockDescriptor()
    {
        using namespace ck_tile;
        constexpr auto b_lds_block_desc_0 = make_naive_tensor_descriptor(
            make_tuple(number<KPerBlock / 8>{}, number<NPerBlock>{}, number<8>{}),
            make_tuple(number<(NPerBlock + 1) * 8>{}, number<8>{}, number<1>{}),
            number<8>{},
            number<1>{});

        constexpr auto b_lds_block_desc = transform_tensor_descriptor(
            b_lds_block_desc_0,
            make_tuple(make_pass_through_transform(NPerBlock),
                       make_merge_transform(make_tuple(KPerBlock / 8, 8))),
            make_tuple(sequence<1>{}, sequence<0, 2>{}),
            make_tuple(sequence<0>{}, sequence<1>{}));

        return b_lds_block_desc;
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeA()
    {
        return sizeof(typename Problem::ADataType) *
               MakeALdsBlockDescriptor<Problem>().get_element_space_size();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSizeB()
    {
        return sizeof(typename Problem::BDataType) *
               MakeBLdsBlockDescriptor<Problem>().get_element_space_size();
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr ck_tile::index_t GetSmemSize()
    {
        return GetSmemSizeA<Problem>() + GetSmemSizeB<Problem>();
    }

    // DRAM tile distribution for A [M, K] - distributes loads across threads
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeADramTileDistribution()
    {
        using namespace ck_tile;
        using ADataType = remove_cvref_t<typename Problem::ADataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t K1 = 16 / sizeof(ADataType);  // Vector load size (8 bf16)
        constexpr index_t K0 = KPerBlock / K1;
        constexpr index_t M2 = get_warp_size() / K0;
        constexpr index_t M1 = kBlockSize / get_warp_size();
        constexpr index_t M0 = MPerBlock / (M2 * M1);

        // Handle case where MPerBlock is smaller than thread distribution
        if constexpr (M0 == 0) {
            // For very small M, all threads cooperate on same M rows
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<1, M1, M2>, sequence<K0, K1>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<2, 0>>,
                                           sequence<1, 2>,
                                           sequence<0, 1>>{});
        } else {
            return make_static_tile_distribution(
                tile_distribution_encoding<sequence<1>,
                                           tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                           tuple<sequence<1>, sequence<1, 2>>,
                                           tuple<sequence<1>, sequence<2, 0>>,
                                           sequence<1, 2>,
                                           sequence<0, 1>>{});
        }
    }

    // DRAM tile distribution for B [N, K]
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeBDramTileDistribution()
    {
        using namespace ck_tile;
        using BDataType = remove_cvref_t<typename Problem::BDataType>;

        constexpr index_t kBlockSize = Problem::kBlockSize;
        constexpr index_t K1 = 16 / sizeof(BDataType);
        constexpr index_t K0 = KPerBlock / K1;
        constexpr index_t N2 = get_warp_size() / K0;
        constexpr index_t N1 = kBlockSize / get_warp_size();
        constexpr index_t N0 = NPerBlock / (N2 * N1);

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<N0, N1, N2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    // Use small-batch optimized block gemm policy.
    // NOTE: use the MMAC-dedicated MmacBlockGemmASmemBSmemCRegV1 (self-contained:
    // loads A/B warp tiles from LDS inside operator(), no external LocalPrefetch
    // required). The generic BlockUniversalGemmAsBsCr Intrawave variant relies on
    // a LocalPrefetch that the v2 pipeline never calls, which left A/B warp tiles
    // all-zero on DCU (gfx936/gfx938) and produced all-zero GEMM outputs.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto GetBlockGemm()
    {
        if constexpr (MPerBlock == 16) {
            // Use 16x16 MMAC for small M tiles
            return ck_tile::MmacBlockGemmASmemBSmemCRegV1<Problem, BlockGemmSmallM16Policy>{};
        } else if constexpr (MPerBlock == 64) {
            // Use 16x32 MMAC with 2x2 warp grid for full-batch M=64 tiles
            return ck_tile::MmacBlockGemmASmemBSmemCRegV1<Problem, BlockGemm2x2Policy>{};
        } else if constexpr (MPerBlock == 32 && NPerBlock == 128) {
            // Use 16x32 MMAC for larger batch sizes
            return ck_tile::MmacBlockGemmASmemBSmemCRegV1<Problem, BlockGemmLargeM32Policy>{};
        } else {
            // Use default 16x32 MMAC for other configurations
            return ck_tile::MmacBlockGemmASmemBSmemCRegV1<Problem, BlockGemmDefaultPolicy>{};
        }
    }
};

// ============================================================================
// CK Pipeline-based Linear Kernel
// Tile size adapts to batch size:
//   BATCH_SIZE <= 16:  16x64x256 tiles with 16x16x16 MFMA (minimal padding)
//   16 < BATCH_SIZE <= 64: 32x128x128 tiles with 32x32x16 MFMA
//   BATCH_SIZE > 64:  128x128x64 tiles with 32x32x16 MFMA, MWarp=2 NWarp=2
//                     (MIterPerWarp=2, NIterPerWarp=2 — matches hipBLASLt strategy)
// ============================================================================
template <typename T,
          int BATCH_SIZE,
          int REDUCTION_SIZE,
          bool FORCE_SMALL_TILE = false>
__device__ __forceinline__ void linear_kernel_ck(void const *input_ptr,
                                                  void const *weight_ptr,
                                                  void const *residual_ptr,
                                                  void *output_ptr,
                                                  int num_active_tokens,
                                                  bool residual_add,
                                                  int output_size,
                                                  int o_stride) {
#ifdef MPK_DISABLE_LINEAR
    return;
#endif
    using namespace ck_tile;

    // Four-tier tile selection:
    //   Tier 0 (small):  16x64x256,  MMAC 16x16x32, MWarp=1 NWarp=4 (bs<=16)
    //   Tier 1 (medium): 64x64x128,  MMAC 16x32x64, MWarp=2 NWarp=2 (17<=bs<=64)
    //   Tier 2 (large):  128x128x64, MMAC 16x32x64, MWarp=2 NWarp=2 (bs>64)
    constexpr bool use_xlarge_tile = !FORCE_SMALL_TILE && (BATCH_SIZE > 64);
    constexpr bool use_medium_tile = !FORCE_SMALL_TILE && (BATCH_SIZE > 16) && !use_xlarge_tile;

    constexpr index_t MPerBlock = use_xlarge_tile ? 128 : (use_medium_tile ? 64 : 16);
    constexpr index_t NPerBlock = use_xlarge_tile ? 128 : (use_medium_tile ? 64 : 64);
    constexpr index_t KPerBlock = use_xlarge_tile ? 64  : (use_medium_tile ? 128 : 256);

    constexpr index_t LoopM = (BATCH_SIZE + MPerBlock - 1) / MPerBlock;
    index_t LoopN = (output_size + NPerBlock - 1) / NPerBlock;
    constexpr index_t NumLoopK = REDUCTION_SIZE / KPerBlock;

    // Warp grid: 2x2 for 64x64 and 128x128, 1x4 for 16x64
    constexpr index_t MWarp = (use_xlarge_tile || use_medium_tile) ? 2 : 1;
    constexpr index_t NWarp = (use_xlarge_tile || use_medium_tile) ? 2 : 4;

    using BlockTile = sequence<MPerBlock, NPerBlock, KPerBlock>;
    using BlockWarps = sequence<MWarp, NWarp>;
    // WarpTile = MMAC warp tile dimensions (16x16x32 or 16x32x64), NOT the iterated tile.
    // MIterPerWarp/NIterPerWarp are computed automatically by BlockUniversalGemmAsBsCr.
    constexpr index_t WarpM = 16;   // MMAC bf16 warp tile kM is fixed at 16
    constexpr index_t WarpN = (use_xlarge_tile || use_medium_tile) ? 32 : 16;
    constexpr index_t WarpK = (use_xlarge_tile || use_medium_tile) ? 64 : 32;
    using WarpTile = sequence<WarpM, WarpN, WarpK>;

    using GemmShape = TileGemmShape<BlockTile, BlockWarps, WarpTile>;

    // Define traits — TransposeC no longer affects the MMAC block gemm (store_tile epilogue
    // follows the CWarpDstr encoding directly), so keep it uniform for all tiles.
    using GemmTraits = TileGemmUniversalTraits<
        true,   // kPadM
        false,  // kPadN
        true,   // kPadK
        false,  // DoubleSmemBuffer
        tensor_layout::gemm::RowMajor,
        tensor_layout::gemm::ColumnMajor,
        tensor_layout::gemm::RowMajor,
        false   // TransposeC: irrelevant for MMAC path
    >;

    using Problem = GemmPipelineProblem<bf16, bf16, float, GemmShape, GemmTraits>;

    // All tiles use the custom MMAC-capable policy (default policy hard-codes MWarp=1/NWarp=1).
    using PipelinePolicy = GemmPipelineSmallTilePolicy<MPerBlock, NPerBlock, KPerBlock>;
    using Pipeline = GemmPipelineAGmemBGmemCRegV2<Problem, PipelinePolicy>;

    // Promote pointers and runtime dims to wave-uniform (SGPR) to eliminate
    // v_readfirstlane waterfall sequences in CK's buffer_load instructions.
    const bf16* d_input = reinterpret_cast<const bf16*>(__uniform_addr(input_ptr));
    const bf16* d_weight = reinterpret_cast<const bf16*>(__uniform_addr(weight_ptr));
    const bf16* d_residual = reinterpret_cast<const bf16*>(__uniform_addr(residual_ptr));
    bf16* d_output = reinterpret_cast<bf16*>(__uniform_addr(output_ptr));
    output_size = __builtin_amdgcn_readfirstlane(output_size);
    o_stride = __builtin_amdgcn_readfirstlane(o_stride);

    extern __shared__ char smem[];

    // Iterate over M tiles
    for (index_t mm = 0; mm < LoopM; mm++) {
        index_t m_offset = mm * MPerBlock;
        index_t m_size = (mm == LoopM - 1) ? (BATCH_SIZE - m_offset) : MPerBlock;
        if (m_size <= 0) continue;

        // Iterate over N tiles
        for (index_t nn = 0; nn < LoopN; nn++) {
            index_t n_offset = nn * NPerBlock;
            index_t n_size = (nn == LoopN - 1) ? (output_size - n_offset) : NPerBlock;
            if (n_size <= 0) continue;

            // Create tensor views for A (input) - offset by m_offset
            auto a_tensor_view = make_naive_tensor_view<address_space_enum::global>(
                d_input + m_offset * REDUCTION_SIZE,
                make_tuple(m_size, index_t(REDUCTION_SIZE)),
                make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
                number<8>{},
                number<1>{}
            );

            // B tensor view - weight matrix offset by n_offset
            // Use non-temporal loads for weights: they're streamed once per iteration
            // and never reused at BS=1. NT loads bypass L2, keeping it free for
            // KV cache and activations that DO benefit from L2 reuse.
#ifdef MPK_NT_WEIGHT_LOADS
            // Cache-stream (.cs) for weight loads on gfx950:
            // DEVICE_NT1 = sc1=1, nt=1 = value 18
            // L1: MISS_EVICT, L2: Cache_Stream (allocate, use, discard)
            // Weights get short-lived L2 residency for M-tile sharing,
            // then are immediately evictable to protect KV cache.
            auto b_tensor_view = make_naive_tensor_view<
                address_space_enum::global,
                memory_operation_enum::set,
                static_cast<amd_buffer_coherence_enum>(18)>(
#else
            auto b_tensor_view = make_naive_tensor_view<address_space_enum::global>(
#endif
                d_weight + n_offset * REDUCTION_SIZE,
                make_tuple(n_size, index_t(REDUCTION_SIZE)),
                make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
                number<8>{},
                number<1>{}
            );

            // Create tile windows
            auto a_tile_window = make_tile_window(
                a_tensor_view,
                make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                {0, 0}
            );

            auto b_tile_window = make_tile_window(
                b_tensor_view,
                make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                {0, 0}
            );

            // Run CK pipeline GEMM
            Pipeline pipeline;

            auto c_block_tile = pipeline(a_tile_window, b_tile_window, NumLoopK, smem);

            block_sync_lds();

            // Epilogue: write GEMM results to global memory via CK store_tile.
            // The MMAC C register layout (kCMLane=4/kCNLane=16, MR/NR repeats) is encoded in
            // CWarpDstrEncoding, so store_tile writes it out correctly regardless of warp tile.
            // Residual is accumulated into the float32 accumulator before the bf16 store.
            {
                auto out_view = make_naive_tensor_view<address_space_enum::global>(
                    d_output + m_offset * o_stride + n_offset,
                    make_tuple(m_size, n_size),
                    make_tuple(index_t(o_stride), index_t(1)),
                    number<8>{},
                    number<1>{});
                auto out_window = make_tile_window(
                    out_view,
                    make_tuple(number<MPerBlock>{}, number<NPerBlock>{}),
                    {0, 0});

                if (residual_add && d_residual != nullptr) {
                    auto res_view = make_naive_tensor_view<address_space_enum::global>(
                        d_residual + m_offset * o_stride + n_offset,
                        make_tuple(m_size, n_size),
                        make_tuple(index_t(o_stride), index_t(1)),
                        number<8>{},
                        number<1>{});
                    // Residual must be loaded with the SAME tile distribution as the
                    // accumulator so tile_elementwise_inout operates element-wise.
                    auto res_window = make_tile_window(
                        res_view,
                        make_tuple(number<MPerBlock>{}, number<NPerBlock>{}),
                        {0, 0},
                        c_block_tile.get_tile_distribution());
                    auto res_tile = load_tile(res_window);
                    tile_elementwise_inout(
                        [](auto& c, const auto& r) {
                            c += type_convert<float>(r);
                        },
                        c_block_tile, res_tile);
                }
                store_tile(out_window, cast_tile<bf16>(c_block_tile));
            }
        }
    }
}

// ============================================================================
// Split-K variant: writes partial GEMM results to float32 workspace.
// Each K-split writes to its own slice (no atomics needed).
// A separate reduce kernel sums all K-split partials + residual → bf16 output.
// ============================================================================

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int O_STRIDE = OUTPUT_SIZE>
__device__ __forceinline__ void linear_kernel_ck_splitk(void const *input_ptr,
                                                         void const *weight_ptr,
                                                         void *workspace_ptr,
                                                         int num_active_tokens) {
#ifdef MPK_DISABLE_LINEAR
    return;
#endif
    using namespace ck_tile;

    constexpr bool use_large_tile = (BATCH_SIZE > 16);
    constexpr index_t MPerBlock = use_large_tile ? 32 : 16;
    constexpr index_t NPerBlock = use_large_tile ? 128 : 64;
    constexpr index_t KPerBlock = use_large_tile ? 128 : 256;

    constexpr index_t LoopM = (BATCH_SIZE + MPerBlock - 1) / MPerBlock;
    constexpr index_t LoopN = (OUTPUT_SIZE + NPerBlock - 1) / NPerBlock;
    constexpr index_t NumLoopK = REDUCTION_SIZE / KPerBlock;

    constexpr index_t MWarp = 1;
    constexpr index_t NWarp = 4;

    using BlockTile = sequence<MPerBlock, NPerBlock, KPerBlock>;
    using BlockWarps = sequence<MWarp, NWarp>;
    // MMAC warp tile: 16x16x32 for small (16x64 block), 16x32x64 for large (32x128 block)
    using WarpTile = std::conditional_t<use_large_tile,
                                        sequence<16, 32, 64>,
                                        sequence<16, 16, 32>>;

    using GemmShape = TileGemmShape<BlockTile, BlockWarps, WarpTile>;
    using GemmTraits = TileGemmUniversalTraits<
        true, false, true, false,
        tensor_layout::gemm::RowMajor,
        tensor_layout::gemm::ColumnMajor,
        tensor_layout::gemm::RowMajor
    >;

    using Problem = GemmPipelineProblem<bf16, bf16, float, GemmShape, GemmTraits>;
    using PipelinePolicy = GemmPipelineSmallTilePolicy<MPerBlock, NPerBlock, KPerBlock>;
    using Pipeline = GemmPipelineAGmemBGmemCRegV2<Problem, PipelinePolicy>;

    // Promote pointers to wave-uniform to eliminate waterfall sequences.
    const bf16* d_input = reinterpret_cast<const bf16*>(__uniform_addr(input_ptr));
    const bf16* d_weight = reinterpret_cast<const bf16*>(__uniform_addr(weight_ptr));
    float* d_workspace = reinterpret_cast<float*>(__uniform_addr(workspace_ptr));

    extern __shared__ char smem[];

    for (index_t mm = 0; mm < LoopM; mm++) {
        index_t m_offset = mm * MPerBlock;
        index_t m_size = (mm == LoopM - 1) ? (BATCH_SIZE - m_offset) : MPerBlock;
        if (m_size <= 0) continue;

        for (index_t nn = 0; nn < LoopN; nn++) {
            index_t n_offset = nn * NPerBlock;
            index_t n_size = (nn == LoopN - 1) ? (OUTPUT_SIZE - n_offset) : NPerBlock;
            if (n_size <= 0) continue;

            auto a_tensor_view = make_naive_tensor_view<address_space_enum::global>(
                d_input + m_offset * REDUCTION_SIZE,
                make_tuple(m_size, index_t(REDUCTION_SIZE)),
                make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
                number<8>{}, number<1>{}
            );

            auto b_tensor_view = make_naive_tensor_view<address_space_enum::global>(
                d_weight + n_offset * REDUCTION_SIZE,
                make_tuple(n_size, index_t(REDUCTION_SIZE)),
                make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
                number<8>{}, number<1>{}
            );

            auto a_tile_window = make_tile_window(
                a_tensor_view,
                make_tuple(number<MPerBlock>{}, number<KPerBlock>{}),
                {0, 0}
            );

            auto b_tile_window = make_tile_window(
                b_tensor_view,
                make_tuple(number<NPerBlock>{}, number<KPerBlock>{}),
                {0, 0}
            );

            Pipeline pipeline;
            auto c_block_tile = pipeline(a_tile_window, b_tile_window, NumLoopK, smem);

            block_sync_lds();

            // Store f32 results to workspace using CK store_tile
            {
                auto ws_tensor_view = make_naive_tensor_view<address_space_enum::global>(
                    d_workspace + m_offset * O_STRIDE + n_offset,
                    make_tuple(number<MPerBlock>{}, number<NPerBlock>{}),
                    make_tuple(index_t(O_STRIDE), index_t(1)),
                    number<1>{},
                    number<1>{}
                );
                auto ws_tile_window = make_tile_window(
                    ws_tensor_view,
                    make_tuple(number<MPerBlock>{}, number<NPerBlock>{}),
                    {0, 0}
                );
                store_tile(ws_tile_window, c_block_tile);
            }
        }
    }
}

// ============================================================================
// Split-K reduce: sum K_SPLITS float32 workspace slices + bf16 residual → bf16
// ============================================================================
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int K_SPLITS,
          int WS_STRIDE,
          int O_STRIDE = OUTPUT_SIZE>
__device__ __forceinline__ void splitk_reduce_kernel(void const *workspace_ptr,
                                                      void const *residual_ptr,
                                                      void *output_ptr,
                                                      int num_active_tokens) {
    const float* ws = reinterpret_cast<const float*>(__uniform_addr(workspace_ptr));
    const bf16* res = reinterpret_cast<const bf16*>(__uniform_addr(residual_ptr));
    bf16* out = reinterpret_cast<bf16*>(__uniform_addr(output_ptr));

    constexpr int TOTAL = BATCH_SIZE * OUTPUT_SIZE;
    for (int idx = threadIdx.x; idx < TOTAL; idx += 256) {
        int row = idx / OUTPUT_SIZE;
        int col = idx % OUTPUT_SIZE;

        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < K_SPLITS; k++) {
            sum += ws[k * BATCH_SIZE * WS_STRIDE + row * WS_STRIDE + col];
        }
        sum += ck_tile::type_convert<float>(res[row * O_STRIDE + col]);
        out[row * O_STRIDE + col] = ck_tile::type_convert<bf16>(sum);
    }
}

// ============================================================================
// Split-K with float32 atomicAdd — SINGLE TASK (no separate reduce)
// ============================================================================
template <typename T,
          int BATCH_SIZE,
          int NPerBlock,
          int REDUCTION_SIZE,
          int K_SPLITS>
__device__ __forceinline__ void splitk_linear_res_atomic_kernel(
        void const *input_ptr,
        void const *weight_ptr,
        void const *residual_ptr,
        void *workspace_ptr,
        void *output_ptr,
        int  *done_counter_ptr,
        int   num_active_tokens,
        int   ws_stride,
        int   o_stride) {
#ifdef MPK_DISABLE_LINEAR
    return;
#endif
    using namespace ck_tile;

    // Split-K uses fixed small tiles (atomicAdd epilogue needs manual register mapping)
    constexpr index_t MPerBlock_CK = 16;
    constexpr index_t NPerBlock_CK = 64;
    constexpr index_t KPerBlock = 256;

    constexpr index_t K_per_split = REDUCTION_SIZE / K_SPLITS;
    constexpr index_t LoopM = (BATCH_SIZE + MPerBlock_CK - 1) / MPerBlock_CK;
    constexpr index_t LoopN = (NPerBlock + NPerBlock_CK - 1) / NPerBlock_CK;
    constexpr index_t NumLoopK = K_per_split / KPerBlock;

    constexpr index_t MWarp = 1;
    constexpr index_t NWarp = 4;

    using BlockTile_CK = sequence<MPerBlock_CK, NPerBlock_CK, KPerBlock>;
    using BlockWarps = sequence<MWarp, NWarp>;
    using WarpTile = sequence<16, 16, 32>;

    using GemmShape = TileGemmShape<BlockTile_CK, BlockWarps, WarpTile>;
    using GemmTraits = TileGemmUniversalTraits<
        true, false, true, false,
        tensor_layout::gemm::RowMajor,
        tensor_layout::gemm::ColumnMajor,
        tensor_layout::gemm::RowMajor
    >;

    using Problem = GemmPipelineProblem<bf16, bf16, float, GemmShape, GemmTraits>;
    using PipelinePolicy = GemmPipelineSmallTilePolicy<MPerBlock_CK, NPerBlock_CK, KPerBlock>;
    using Pipeline = GemmPipelineAGmemBGmemCRegV2<Problem, PipelinePolicy>;

    // Promote pointers and runtime dims to wave-uniform to eliminate waterfall.
    const bf16* d_input  = reinterpret_cast<const bf16*>(__uniform_addr(input_ptr));
    const bf16* d_weight = reinterpret_cast<const bf16*>(__uniform_addr(weight_ptr));
    float*      d_ws     = reinterpret_cast<float*>(__uniform_addr(workspace_ptr));
    ws_stride = __builtin_amdgcn_readfirstlane(ws_stride);
    o_stride  = __builtin_amdgcn_readfirstlane(o_stride);

    extern __shared__ char smem[];

    // Phase 1: GEMM + atomicAdd to workspace
    for (index_t mm = 0; mm < LoopM; mm++) {
        index_t m_offset = mm * MPerBlock_CK;
        index_t m_size = (mm == LoopM - 1) ? (BATCH_SIZE - m_offset) : MPerBlock_CK;
        if (m_size <= 0) continue;

        for (index_t nn = 0; nn < LoopN; nn++) {
            index_t n_offset = nn * NPerBlock_CK;
            index_t n_size = (nn == LoopN - 1) ? (NPerBlock - n_offset) : NPerBlock_CK;
            if (n_size <= 0) continue;

            auto a_view = make_naive_tensor_view<address_space_enum::global>(
                d_input + m_offset * REDUCTION_SIZE,
                make_tuple(m_size, index_t(K_per_split)),
                make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
                number<8>{}, number<1>{});

            auto b_view = make_naive_tensor_view<address_space_enum::global>(
                d_weight + n_offset * REDUCTION_SIZE,
                make_tuple(n_size, index_t(K_per_split)),
                make_tuple(index_t(REDUCTION_SIZE), index_t(1)),
                number<8>{}, number<1>{});

            auto a_win = make_tile_window(a_view,
                make_tuple(number<MPerBlock_CK>{}, number<KPerBlock>{}), {0, 0});
            auto b_win = make_tile_window(b_view,
                make_tuple(number<NPerBlock_CK>{}, number<KPerBlock>{}), {0, 0});

            Pipeline pipeline;
            auto c_tile = pipeline(a_win, b_win, NumLoopK, smem);
            block_sync_lds();

            // float32 atomicAdd epilogue (layout-independent: sweeps the C distribution)
            for_each_c_element(c_tile, [&](index_t row_in_block,
                                          index_t col_in_block,
                                          float val) {
                index_t global_m = m_offset + row_in_block;
                index_t global_n = n_offset + col_in_block;
                if (global_m < BATCH_SIZE && global_n < NPerBlock) {
                    atomicAdd(&d_ws[global_m * ws_stride + global_n], val);
                }
            });
        }
    }

    // Phase 2: Track completion with atomic counter
    // Agent-scope release fence — L2 NOT coherent across XCDs, need writeback
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "agent");
    int* d_done = reinterpret_cast<int*>(__uniform_addr(done_counter_ptr));
    __shared__ int is_last;
    if (threadIdx.x == 0) {
        int old = atomicAdd(d_done, 1);
        is_last = (old == K_SPLITS - 1) ? 1 : 0;
    }
    __syncthreads();

    // Phase 3: Last block — add residual, convert to bf16, zero workspace, reset counter
    if (is_last) {
        const bf16* res = residual_ptr ? reinterpret_cast<const bf16*>(__uniform_addr(residual_ptr)) : nullptr;
        bf16*       out = reinterpret_cast<bf16*>(__uniform_addr(output_ptr));

        constexpr int TOTAL = BATCH_SIZE * NPerBlock;
        for (int idx = threadIdx.x; idx < TOTAL; idx += 256) {
            int row = idx / NPerBlock;
            int col = idx % NPerBlock;
            index_t ws_idx = row * ws_stride + col;
            index_t out_idx = row * o_stride + col;

            float val = d_ws[ws_idx];
            if (res) val += type_convert<float>(res[out_idx]);
            out[out_idx] = type_convert<bf16>(val);
            d_ws[ws_idx] = 0.0f;
        }
        if (threadIdx.x == 0) {
            *d_done = 0;
        }
    }
}

} // namespace kernel
