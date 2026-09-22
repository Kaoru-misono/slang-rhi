#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace rhi::testing::flat {

struct alignas(16) Nested
{
    uint32_t tag;
    float weights[3];
};

struct alignas(16) Params
{
    float direction[3];
    float scale;
    float samples[2][4];
    float matrix[4][4];
    Nested nested;
};

static_assert(std::is_trivially_copyable_v<Params> && std::is_standard_layout_v<Params>);
static_assert(sizeof(Params) == 128 && alignof(Params) == 16);
static_assert(offsetof(Params, direction) == 0 && offsetof(Params, scale) == 12);
static_assert(offsetof(Params, samples) == 16 && offsetof(Params, matrix) == 48);
static_assert(offsetof(Params, nested) == 112);
static_assert(sizeof(Nested) == 16 && alignof(Nested) == 16 && offsetof(Nested, weights) == 4);

} // namespace rhi::testing::flat
