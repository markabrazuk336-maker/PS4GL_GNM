# Third-party notices

## dlmalloc (`src/ps4gl_heap.c`)

`src/ps4gl_heap.c` contains dlmalloc 2.8.3 by Doug Lea, which its author released to the public domain, as explained at <http://creativecommons.org/licenses/publicdomain>.

The text of dlmalloc was taken from the copy in SDL2 (`src/stdlib/SDL_malloc.c`). The SDL file carries this notice:

```text
Simple DirectMedia Layer
Copyright (C) 1997-2021 Sam Lantinga <slouken@libsdl.org>

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgment in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
```

This is an altered version: `tools/gen_heap.py` removes the SDL wrappers, replaces the locks with a spin lock and adds a memory source for PS4.

## Build tools (not included in this repository)

- [OpenOrbis PS4 Toolchain](https://github.com/OpenOrbis/OpenOrbis-PS4-Toolchain) is licensed under GPLv3. Programs built with it link its libraries, and `tools/mkpkg.py` copies support files from your toolchain folder into the package. Check the toolchain's license before you distribute built packages.
- [LLVM](https://llvm.org/) is licensed under Apache 2.0 with LLVM exceptions.
