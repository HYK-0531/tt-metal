// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"

#include "sfpi.h"
// #include "ckernel_sfpu_exp.h"
// #include "ckernel_sfpu_custom_exp.h"
#include "sfpu/ckernel_sfpu_converter.h"

using namespace sfpi;

namespace ckernel {
namespace sfpu {

sfpi_inline sfpi::vFloat _sfpu_exp_21f_(sfpi::vFloat val) {
    sfpi::vInt z = sfpu::_float_to_int32_(val * sfpi::vFloat(0x00b8aa3b) + sfpi::vFloat(0x3f800000));
    sfpi::vInt zii = z & 0x7f800000;
    sfpi::vInt zif = z & sfpi::vInt(0x007fffff);  // extra mantissa

    sfpi::vFloat d1 = sfpi::vFloat(0.40196114e-7);
    sfpi::vFloat d2 = sfpi::int32_to_float(sfpi::vInt(0xf94ee7) + zif);
    sfpi::vFloat d3 = sfpi::int32_to_float(sfpi::vInt(0x560) + zif);
    d2 = d1 * d2;
    zif = sfpu::_float_to_int32_(d2 * d3);

    zii |= zif;  // restore exponent

    sfpi::vFloat y = sfpi::reinterpret<sfpi::vFloat>(zii);

    return y;
}

template <bool APPROXIMATION_MODE>
inline void calculate_elu(uint slope) {
    // SFPU microcode
    vFloat s = Converter::as_float(slope);

#pragma GCC unroll 0
    for (int d = 0; d < 8; d++) {
        vFloat v = dst_reg[0];

        v_if(v < 0.0f) {
            vFloat v_exp = _sfpu_exp_21f_(v);
            v = s * (v_exp - 1.0f);
        }
        v_endif;

        dst_reg[0] = v;

        dst_reg++;
    }
}

template <bool APPROXIMATION_MODE>
void elu_init() {
    if constexpr (APPROXIMATION_MODE) {
        vConstFloatPrgm0 = 1.442695f;  // ln2_recip
        vConstFloatPrgm1 = s2vFloat16b(p_exp::C23_73);
        vConstFloatPrgm2 = s2vFloat16b(p_exp::ADJ_EXP);
    } else {
        vConstFloatPrgm0 = 1.442695f;  // ln2_recip
        vConstFloatPrgm1 = 2.0f;
        vConstFloatPrgm2 = 0.863281f;
    }
}

}  // namespace sfpu
}  // namespace ckernel
