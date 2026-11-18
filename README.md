# PS4GL_GNM

**OpenGL ES-style graphics for jailbroken PS4, running directly on the GPU through GNM. No Sony SDK.**

[Русская версия](README.ru.md)

PS4GL_GNM is a small graphics driver for PS4 homebrew. You write ordinary `glXxx` calls, and the driver turns them into GPU command packets (PM4). It comes with its own GLSL-to-GCN shader compiler, which runs on your PC and is built on stock LLVM.

Why not Piglet, the system OpenGL library? Piglet only starts in system-category apps (`gde`), which get about 768 MiB of memory, and it has no shader compiler on a retail console. A normal game package (`gd`) gets about 4.5 GiB, and PS4GL_GNM works there.

## Status

- The driver has rendered complex 3D scenes at 60 FPS on a PS4 Pro with GoldHEN.
- The cube sample in this repository builds into a `.pkg`, and its GPU commands have been checked on a PC test stand. **It has not been run on a console yet.**
- This is a young project with a limited feature set. See [Limitations](#limitations).

## Requirements

**On the console:** a jailbroken PS4 with [GoldHEN](https://github.com/GoldHEN/GoldHEN), which is needed to install and run fake-signed packages.

**On the PC:**

- [OpenOrbis PS4 Toolchain](https://github.com/OpenOrbis/OpenOrbis-PS4-Toolchain), with the `OO_PS4_TOOLCHAIN` environment variable pointing at it
- [LLVM](https://llvm.org/) on `PATH` with the AMDGPU target: `clang`, `clang++`, `ld.lld`, `llc`, `opt`, `llvm-link`, `llvm-objcopy`, `llvm-mc` (tested with LLVM 19.1)
- Python 3.8 or newer
- .NET runtime, for the toolchain's `PkgTool.Core`

## Build the sample

The sample draws a coloured triangle and a textured spinning cube.

PowerShell:

```powershell
$env:OO_PS4_TOOLCHAIN = "C:\OpenOrbis\PS4Toolchain"
python build.py
```

bash:

```bash
export OO_PS4_TOOLCHAIN=~/OpenOrbis/PS4Toolchain
python3 build.py
```

The package appears at `build/cube/IV0000-PSGL00001_00-PS4GLCUBE0000000.pkg`. Copy it to the console and install it with GoldHEN's Package Installer.

## Using the driver

Add `src/ps4gl.cpp` to your project (and optionally `src/ps4gl_heap.c`), put `include/` and `src/` on the include path, and link `-lSceVideoOut -lSceGnmDriver -lSceSystemService`.

```cpp
#include "ps4gl.h"

int main()
{
    ps4glInit( 0 );                               // 1920x1080, 1 GiB of video memory

    GLuint vs = glCreateShader( GL_VERTEX_SHADER );
    glShaderBinary( 1, &vs, 0, vsData, vsSize );  // compiled on the PC by tools/shadersbC.py
    // ... the usual OpenGL ES 2 setup ...

    for ( ;; )
    {
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
        glDrawElements( GL_TRIANGLES, count, GL_UNSIGNED_SHORT, 0 );
        ps4glSwapBuffers();
    }
}
```

Compile shaders ahead of time:

```bash
python tools/shadersbC.py shader.vert -o shader.vert.sb
```

The driver writes its log to `/data/ps4gl/log.txt`. `ps4glSetLogCallback` sends it elsewhere.

## What works

| Area | Supported |
|---|---|
| Buffers | Vertex and index buffers, `glBufferSubData`, `glMapBufferRange`, 8/16/32-bit indices |
| Textures | RGBA8, RGB, BGRA, luminance/alpha, 4444/5551/565, 16-bit and float formats, mipmaps |
| Compressed textures | DXT1, DXT3, DXT5 (unpacked to RGBA8 on upload) |
| Shaders | GLSL vertex and fragment shaders, up to 16 textures per shader, 16 vertex attributes |
| Depth | 32-bit float depth, all compare functions, polygon offset |
| Blending | All factors and equations, separate alpha |
| Render targets | Framebuffer objects with colour and depth, `glBlitFramebuffer` |
| Other state | Scissor, culling, colour mask, sampler objects |
| Presentation | Triple buffering, 60 FPS vsync |

## GNM problems and how the driver handles them

These cost the most time to find, so they are written down here for anyone else working with GNM directly.

| Problem | Workaround |
|---|---|
| `DRAW_INDEX_2` hangs the GPU on some draws | Draw with `DRAW_INDEX_AUTO`. The fetch shader reads the index from the index buffer itself |
| BC/DXT textures in linear tiling hang the GPU | Unpack to RGBA8 on upload |
| Reading a render target right after drawing gives torn stripes | Flush the colour and depth caches when the render target changes |
| Clearing 1080p through DMA costs about 20 ms per frame | Clear with a full-screen quad drawn by the GPU |
| With register shadowing on, a clear only reached part of the target | Forget the shadow state just before clearing |
| Every context register write can start a new hardware context | Keep a shadow of context registers and write only what changed |
| `sceGnmSubmitAndFlipCommandBuffers` fails without a marker | End the command buffer with a 64-dword NOP carrying the PrepareFlip marker |
| Triangle winding differs between the screen and framebuffer objects | Flip the front-face setting for targets drawn without a Y flip |
| Flexible memory runs out at about 300 MiB | `ps4gl_heap.c` puts `malloc` on direct memory |
| `malloc` with 8-byte alignment crashes SSE code, and pthread mutexes crash this early | 16-byte alignment and a spin lock |

## Limitations

- The screen is always 1920x1080.
- Shaders must be compiled on the PC. `glShaderSource` and `glCompileShader` do nothing.
- The shader compiler translates a practical subset of GLSL. `gl_FragCoord`, `gl_FragDepth` and shaders that need scratch memory are not supported.
- No cube maps, 3D textures or texture arrays.
- No stencil, MSAA, instancing or multiple colour attachments.
- Mipmaps need RGBA8 textures with power-of-two sides, at least 64 pixels wide.
- Vertex array objects, queries and sync objects are accepted but do nothing.
- `glReadPixels` returns zeros, and `glGetError` always reports no error.
- An indexed draw with no vertex attributes needs sequential indices.

## Project layout

```text
include/ps4gl.h        public API
src/ps4gl.cpp          the driver
src/ps4gl_heap.c       optional malloc on direct memory (generated)
src/gen/               generated fetch shaders and tables
shaders/               GLSL support library and the driver's own shaders
tools/shadersbC.py        GLSL -> PS4 shader binary
tools/glsl2cl.py       GLSL -> OpenCL C (first compiler stage)
tools/sbformat.py      shader binary format: read and write
tools/sbdump.py        inspect shader binaries
tools/gen_*.py         generators for the files in src/
tools/mkpkg.py         ELF -> installable .pkg
samples/cube/          triangle and spinning cube
```

## Thanks

- [OpenOrbis](https://github.com/OpenOrbis/OpenOrbis-PS4-Toolchain), for the toolchain that makes PS4 homebrew possible without Sony's SDK
- [LLVM](https://llvm.org/), whose AMDGPU backend generates the shader code
- [shadPS4](https://github.com/shadps4-emu/shadPS4), for public information about texture and tiling layouts
- [GoldHEN](https://github.com/GoldHEN/GoldHEN) and the PS4 homebrew community
- Doug Lea, for dlmalloc

## License

[MIT](LICENSE). Third-party notices are in [THIRD_PARTY.md](THIRD_PARTY.md).

This project is not affiliated with or endorsed by Sony Interactive Entertainment. "PlayStation" and "PS4" are trademarks of their owners. The repository contains no code from the Sony SDK.
