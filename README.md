<!--
SPDX-FileCopyrightText: The Khronos Group, Inc.
SPDX-License-Identifier: CC-BY-4.0
-->

[![CI](https://github.com/shader-slang/slang-rhi/actions/workflows/ci.yml/badge.svg)](https://github.com/shader-slang/slang-rhi/actions/workflows/ci.yml)
[![Coverage Status](https://coveralls.io/repos/github/shader-slang/slang-rhi/badge.svg?branch=main)](https://coveralls.io/github/shader-slang/slang-rhi?branch=main)

# slang-rhi

This April2 fork supports **Vulkan and D3D12 only**. It is based on upstream
`82c03494bed8e2d42d65555e33b30a18d3d8f071` (2026-09-18) and uses Slang
`2026.17.1`. Compute/transfer queues, queue ownership transfers, Vulkan VMA
allocation, and April binding caches are maintained here. Other backend enum
values remain reserved for API compatibility, but cannot create devices.

## Introduction

The `slang-rhi` library provides a render hardware interface for the Slang shading language.
It is based on the "gfx" layer originally developed in the Slang repository.
This library is under active refactoring and development, and is not yet ready for general use.

## License

`slang-rhi` is released under the Apache 2.0 with LLVM Exception license. See the file  [LICENSE](LICENSE) for more information.

`slang-rhi` depends on the following third-party libraries, which have their own license:

- [doctest](https://github.com/doctest/doctest) (MIT)
- [RenderDoc API](https://github.com/baldurk/renderdoc) (MIT)
- [stb](https://github.com/nothings/stb) (Public Domain)
- [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) (MIT)
- [OffsetAllocator](https://github.com/sebbbi/OffsetAllocator) (MIT)
- [WinPixEventRuntime](https://www.nuget.org/packages/WinPixEventRuntime) (MIT)
- [D3D12 Memory Allocator](https://gpuopen.com/d3d12-memory-allocator) (MIT)
- [Vulkan Memory Allocator](https://gpuopen.com/vulkan-memory-allocator) (MIT)
