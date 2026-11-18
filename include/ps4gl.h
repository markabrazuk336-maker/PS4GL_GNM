// PS4GL_GNM — OpenGL ES-like API for PS4, implemented directly on GNM. See README.md.
//
// Usage:
//     ps4glInit( 0 );                       // opens the screen, 1920x1080
//     ... ordinary glXxx calls ...
//     ps4glSwapBuffers();                   // shows the frame
//
// Shaders are compiled ahead of time on the PC by tools/shadersbC.py and loaded with
// glShaderBinary( 1, &shader, 0, data, size ): the PS4 has no GLSL compiler at run time.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <GLES2/gl2.h>

#ifdef __cplusplus
extern "C" {
#endif

// Starts the driver: video output, video memory pool, default framebuffer with depth.
// videoPoolBytes — size of the GPU memory pool, 0 means 1 GiB. Returns 1 on success.
// Call it once, from the thread that will do the rendering.
int ps4glInit( size_t videoPoolBytes );

// Submits everything drawn since the previous call and queues the frame for display.
void ps4glSwapBuffers( void );

// 1 when the console is a PS4 Pro running in its faster Pro mode.
int ps4glIsProMode( void );

// Supersampling for the default framebuffer: scale 2 renders at 3840x2160 and averages the
// picture down to the 1920x1080 output, which removes jagged edges. Scale 1 turns it off.
// Your code keeps using 1920x1080 coordinates. The change applies from the next frame.
// It draws four times as many pixels and uses about 64 MiB more video memory.
void ps4glSetSupersampling( int scale );
int ps4glGetSupersampling( void );			// the scale used by the current frame

// Address of a GL function by name, or NULL (the same idea as eglGetProcAddress).
void *ps4glGetProcAddress( const char *name );

// Log. By default lines go to /data/ps4gl/log.txt; a callback replaces that.
typedef void ( *Ps4glLogFn )( const char *line );
void ps4glSetLogCallback( Ps4glLogFn fn );
void ps4glLog( const char *fmt, ... ) __attribute__(( format( printf, 1, 2 ) ));

// ---- OpenGL ES 3 entry points that GLES2/gl2.h does not declare ----
// Real implementations:
void glBlitFramebuffer( GLint srcX0, GLint srcY0, GLint srcX1, GLint srcY1, GLint dstX0, GLint dstY0, GLint dstX1, GLint dstY1, GLbitfield mask, GLenum filter );
void glCopyBufferSubData( GLenum readTarget, GLenum writeTarget, GLintptr readOffset, GLintptr writeOffset, GLsizeiptr size );
void glDrawRangeElements( GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices );
void glDrawRangeElementsBaseVertex( GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void *indices, GLint baseVertex );
void *glMapBufferRange( GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access );
GLboolean glUnmapBuffer( GLenum target );
void glFlushMappedBufferRange( GLenum target, GLintptr offset, GLsizeiptr length );
void glRenderbufferStorageMultisample( GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height );
void glBindSampler( GLuint unit, GLuint sampler );
void glGenSamplers( GLsizei count, GLuint *samplers );
void glDeleteSamplers( GLsizei count, const GLuint *samplers );
void glSamplerParameteri( GLuint sampler, GLenum pname, GLint param );
void glSamplerParameterf( GLuint sampler, GLenum pname, GLfloat param );
void glSamplerParameterfv( GLuint sampler, GLenum pname, const GLfloat *param );

// Accepted but do nothing (see "Limitations" in README.md):
void glBindVertexArray( GLuint array );
void glGenVertexArrays( GLsizei n, GLuint *arrays );
void glDeleteVertexArrays( GLsizei n, const GLuint *arrays );
void glBeginQuery( GLenum target, GLuint id );
void glEndQuery( GLenum target );
void glGenQueries( GLsizei n, GLuint *ids );
void glDeleteQueries( GLsizei n, const GLuint *ids );
void glGetQueryObjectuiv( GLuint id, GLenum pname, GLuint *params );
GLsync glFenceSync( GLenum condition, GLbitfield flags );
GLenum glClientWaitSync( GLsync sync, GLbitfield flags, GLuint64 timeout );
void glWaitSync( GLsync sync, GLbitfield flags, GLuint64 timeout );
void glDeleteSync( GLsync sync );
void glGetSynciv( GLsync sync, GLenum pname, GLsizei count, GLsizei *length, GLint *values );
void glDrawBuffers( GLsizei n, const GLenum *bufs );
void glReadBuffer( GLenum src );
void glTexImage3D( GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const void *pixels );
void glCompressedTexImage3D( GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLsizei imageSize, const void *data );

#ifdef __cplusplus
}
#endif
