// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_addrmod.h"
#include "ckernel_defs.h"
#include "sfpi.h"

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void mul_int32(const uint dst_offset) {
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 64;
        // operand A - int32
        TTI_SFPLOAD(p_sfpu::LREG0, INT32, ADDR_MOD_3, 0);
        // operand B - int32
        TT_SFPLOAD(p_sfpu::LREG1, INT32, ADDR_MOD_3, dst_offset * dst_tile_size);

        // INT32 split into 8-bit inputs
        // mask
        TTI_SFPLOADI(p_sfpu::LREG7, SFPLOADI_MOD0_USHORT, 0xFF);

        // Copy A
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0);
        TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG3, 0);
        // TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG4, 0);

        // Extract A = [A3:A2:A1:A0] where each is 8-bit. Ignore A3 since its computation will go beyond 32 bits
        TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG0, 0);               // LREG0 = A0 = A[7:0]
        TTI_SFPSHFT((-8) & 0xfff, p_sfpu::LREG2, p_sfpu::LREG2, 1);   // LREG2 = A1 = A[15:8]
        TTI_SFPSHFT((-16) & 0xfff, p_sfpu::LREG3, p_sfpu::LREG3, 1);  // LREG3 = A2 = A[23:16]

        // Copy B
        TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG4, 0);
        TTI_SFPMOV(0, p_sfpu::LREG1, p_sfpu::LREG5, 0);

        // Extract B = [B3:B2:B1:B0] where each is 8-bit
        TTI_SFPAND(0, p_sfpu::LREG7, p_sfpu::LREG1, 0);               // LREG1 = B0 = B[7:0]
        TTI_SFPSHFT((-8) & 0xfff, p_sfpu::LREG4, p_sfpu::LREG4, 1);   // LREG5 = B1 = B[15:8]
        TTI_SFPSHFT((-16) & 0xfff, p_sfpu::LREG5, p_sfpu::LREG5, 1);  // LREG6 = B2 = B[23:16]

        // Cast all 8-bit values to FP32
        TTI_SFPCAST(p_sfpu::LREG0, p_sfpu::LREG0, 0);
        TTI_SFPCAST(p_sfpu::LREG1, p_sfpu::LREG1, 0);
        TTI_SFPCAST(p_sfpu::LREG2, p_sfpu::LREG2, 0);
        TTI_SFPCAST(p_sfpu::LREG3, p_sfpu::LREG3, 0);
        TTI_SFPCAST(p_sfpu::LREG4, p_sfpu::LREG4, 0);
        TTI_SFPCAST(p_sfpu::LREG5, p_sfpu::LREG5, 0);

        // a0*b0 (bits 0-15)
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG7, 0);
        TTI_SFPNOP;
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG7, p_sfpu::LREG7, 7);

        // a0*b1 (bits 8-23)
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
        TTI_SFPNOP;
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG6, p_sfpu::LREG6, 7);
        TTI_SFPSHFT(8, p_sfpu::LREG6, p_sfpu::LREG6, 1);                     // Shift left by 8
        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG7, SFPIADD_MOD1_CC_NONE);  // Accumulate in LREG7

        // a0*b2 (bits 16-31)
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
        TTI_SFPNOP;
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG6, p_sfpu::LREG6, 7);
        TTI_SFPSHFT(16, p_sfpu::LREG6, p_sfpu::LREG6, 1);                    // Shift left by 16
        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG7, SFPIADD_MOD1_CC_NONE);  // Accumulate

        // a1*b0 (bits 8-23)
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
        TTI_SFPNOP;
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG6, p_sfpu::LREG6, 7);
        TTI_SFPSHFT(8, p_sfpu::LREG6, p_sfpu::LREG6, 1);                     // Shift left by 8
        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG7, SFPIADD_MOD1_CC_NONE);  // Accumulate

        // a1*b1 (bits 16-31)
        TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
        TTI_SFPNOP;
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG6, p_sfpu::LREG6, 7);
        TTI_SFPSHFT(16, p_sfpu::LREG6, p_sfpu::LREG6, 1);                    // Shift left by 16
        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG7, SFPIADD_MOD1_CC_NONE);  // Accumulate

        // a2*b0 (bits 16-31)
        TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
        TTI_SFPNOP;
        TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG6, p_sfpu::LREG6, 7);
        TTI_SFPSHFT(16, p_sfpu::LREG6, p_sfpu::LREG6, 1);                    // Shift left by 16
        TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG7, SFPIADD_MOD1_CC_NONE);  // Accumulate

        // // a1*b2 --> goes beyond 32-bits
        // TTI_SFPMUL(p_sfpu::LREG2, p_sfpu::LREG5, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
        // TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG6, p_sfpu::LREG6, 7);
        // TTI_SFPSHFT(24, p_sfpu::LREG6, p_sfpu::LREG6, 1);                               // Shift left by 24
        // TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG7, SFPIADD_MOD1_CC_NONE);             // Accumulate

        // // a2*b1 --> goes beyond 32-bits
        // TTI_SFPMUL(p_sfpu::LREG3, p_sfpu::LREG4, p_sfpu::LCONST_0, p_sfpu::LREG6, 0);
        // TTI_SFP_STOCH_RND(0, 0, 0, p_sfpu::LREG6, p_sfpu::LREG6, 7);
        // TTI_SFPSHFT(24, p_sfpu::LREG6, p_sfpu::LREG6, 1);                               // Shift left by 24
        // TTI_SFPIADD(0, p_sfpu::LREG6, p_sfpu::LREG7, SFPIADD_MOD1_CC_NONE);             // Accumulate

        TTI_SFPSTORE(p_sfpu::LREG7, 4, ADDR_MOD_3, 0);
        sfpi::dst_reg++;
    }
}

}  // namespace sfpu
}  // namespace ckernel
