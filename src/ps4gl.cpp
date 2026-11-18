// PS4GL_GNM: OpenGL ES-подобный драйвер для PS4 прямо на GNM, без SDK Sony.
//
// Функции glXxx складывают состояние, а при рисовании пишут команды видеокарте (PM4)
// в буфер кадра. Кадр целиком отправляется в ps4glSwapBuffers. Шейдеры заранее собраны
// на ПК (tools/shadersbC.py) и приходят через glShaderBinary.
//
// Лицензия: MIT (см. LICENSE).
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>
#include <orbis/libkernel.h>
#include <orbis/VideoOut.h>
#include <orbis/GnmDriver.h>
#include <orbis/SystemService.h>
#include <iterator>
#include <map>
#include <vector>
#include <string>
#include "ps4gl.h"
#include "gen/ps4gl_fetch.h"
#include "gen/ps4gl_builtin_shaders.h"

// ---- журнал ----
static Ps4glLogFn s_logFn;
void ps4glSetLogCallback( Ps4glLogFn fn ) { s_logFn = fn; }
void ps4glLog( const char *fmt, ... )
{
	char line[512];
	va_list ap;
	va_start( ap, fmt );
	vsnprintf( line, sizeof( line ), fmt, ap );
	va_end( ap );
	if ( s_logFn ) { s_logFn( line ); return; }
	static bool s_started;
	if ( !s_started ) { mkdir( "/data/ps4gl", 0777 ); unlink( "/data/ps4gl/log.txt" ); s_started = true; }
	FILE *f = fopen( "/data/ps4gl/log.txt", "a" );
	if ( f ) { fprintf( f, "%s\n", line ); fclose( f ); }
}

// ---- числа OpenGL ES 3, которых нет в gl2.h ----
#define GL_ALREADY_SIGNALED_ 0x911A
#define GL_SIGNALED_ 0x9119
#define GL_SYNC_STATUS_ 0x9114
#define GL_FRAMEBUFFER_BINDING_ 0x8CA6
#define GL_MAX_SAMPLES_ 0x8D57
#define GL_MAX_DRAW_BUFFERS_ 0x8824
#define GL_MAX_COLOR_ATTACHMENTS_ 0x8CDF
#define GL_NUM_EXTENSIONS_ 0x821D
#define GL_MAJOR_VERSION_ 0x821B
#define GL_MINOR_VERSION_ 0x821C
#define GL_MAX_3D_TEXTURE_SIZE_ 0x8073
#define GL_MAX_ARRAY_TEXTURE_LAYERS_ 0x88FF

static const int SCREEN_W = 1920, SCREEN_H = 1080;

// =====================================================================
// Устройство: VideoOut, прямая память, буфер команд, показ кадра
// =====================================================================
static int s_video = -1;
static OrbisKernelEqueue s_flipQueue;
static uint8_t *s_pool, *s_poolEnd;					// прямая память видеокарты (пул)
static uint32_t *s_fb[3];								// кадры на экран (BGRA, линейные): тройная буферизация
static int s_fbIdx;										// в какой из трёх рисуем этот кадр
static int s_flipsPending;								// отправлено на показ, но ещё не показано
static uint32_t s_ctxShadow[0x400];
static uint8_t s_ctxValid[0x400];
static uint32_t *s_dcb[2];								// буферы команд
static const size_t DCB_BYTES = 4u << 20;
static uint32_t *s_cmd;									// куда пишем команды текущего кадра
static int s_back;										// какой кадр сейчас рисуем
static int s_frame;
static bool s_clearedBackThisFrame;
static uint32_t s_lastClearBGRA = 0xFF000000u;

// ---- распределитель видеопамяти: свободные куски по адресу, первый подходящий ----
static std::map<uintptr_t, size_t> s_free;				// начало -> размер
static size_t s_gpuUsed, s_gpuPeak;
struct PendingFree { void *p; size_t size; int frame; };
static std::vector<PendingFree> s_pendingFree;			// освободить, когда видеокарта закончит кадр

// свободные куски: по адресу (для слияния) и по размеру (быстрый поиск)
static std::multimap<size_t, uintptr_t> s_freeBySize;
static void FreeInsert( uintptr_t p, size_t n ) { s_free[p] = n; s_freeBySize.insert( { n, p } ); }
static void FreeErase( std::map<uintptr_t, size_t>::iterator it )
{
	auto r = s_freeBySize.equal_range( it->second );
	for ( auto j = r.first; j != r.second; ++j ) if ( j->second == it->first ) { s_freeBySize.erase( j ); break; }
	s_free.erase( it );
}

static void *GpuAlloc( size_t bytes, size_t align )
{
	if ( align < 256 ) align = 256;
	bytes = ( bytes + 255 ) & ~(size_t)255;
	// самый маленький подходящий кусок (с запасом на выравнивание — проверяем несколько)
	for ( auto j = s_freeBySize.lower_bound( bytes ); j != s_freeBySize.end(); ++j )
	{
		uintptr_t start = j->second, end = start + j->first;
		uintptr_t p = ( start + align - 1 ) & ~(uintptr_t)( align - 1 );
		if ( p + bytes > end ) continue;
		FreeErase( s_free.find( start ) );
		if ( p > start ) FreeInsert( start, p - start );
		if ( p + bytes < end ) FreeInsert( p + bytes, end - ( p + bytes ) );
		s_gpuUsed += bytes;
		if ( s_gpuUsed > s_gpuPeak ) s_gpuPeak = s_gpuUsed;
		return (void *)p;
	}
	ps4glLog( "ps4gl: out of video memory (asked %llu bytes, used %llu MiB)", (unsigned long long)bytes,
	              (unsigned long long)( s_gpuUsed >> 20 ) );
	return NULL;
}

static void GpuFree( void *ptr, size_t bytes )
{
	if ( !ptr ) return;
	bytes = ( bytes + 255 ) & ~(size_t)255;
	uintptr_t p = (uintptr_t)ptr;
	s_gpuUsed -= bytes;
	auto next = s_free.lower_bound( p );
	if ( next != s_free.end() && next->first == p + bytes ) { bytes += next->second; auto n2 = std::next( next ); FreeErase( next ); next = n2; }
	if ( next != s_free.begin() )
	{
		auto prev = std::prev( next );
		if ( prev->first + prev->second == p ) { uintptr_t ps = prev->first; size_t pn = prev->second; FreeErase( prev ); FreeInsert( ps, pn + bytes ); return; }
	}
	FreeInsert( p, bytes );
}

static int s_frameNo;									// номер кадра (растёт на каждом показе)
// Видеокарта могла ещё не выполнить команды текущего кадра — освободим после показа.
static void GpuFreeLater( void *p, size_t bytes ) { if ( p ) s_pendingFree.push_back( { p, bytes, s_frameNo } ); }

static void *PoolAlloc( size_t bytes, size_t align ) { return GpuAlloc( bytes, align ); }

static inline uint32_t Pkt3( uint32_t op, uint32_t body ) { return 0xC0000000u | ( ( body - 1 ) << 16 ) | ( op << 8 ); }
static void DrawInit();
static void DrawFrameReset();
static void FlushCaches();
static int s_frameDrawsFb0;

// DMA_DATA: заполнить память числом (до 2 МБ за пакет)
static void DmaFill( void *dst, uint32_t value, size_t bytes )
{
	for ( size_t o = 0; o < bytes; o += 1024 * 1024 )
	{
		uint32_t n = (uint32_t)( bytes - o < 1024 * 1024 ? bytes - o : 1024 * 1024 );
		uint64_t a = (uint64_t)(uintptr_t)( (char *)dst + o );
		*s_cmd++ = 0xC0055000u;
		*s_cmd++ = ( 1u << 31 ) | ( 2u << 29 );
		*s_cmd++ = value;
		*s_cmd++ = 0;
		*s_cmd++ = (uint32_t)a;
		*s_cmd++ = (uint32_t)( a >> 32 );
		*s_cmd++ = n & 0x1FFFFF;
	}
}

static void BeginFrameCommands()
{
	// всё, что освободили до прошлого показа, видеокарта уже отработала
	size_t keep = 0;
	for ( size_t i = 0; i < s_pendingFree.size(); i++ )
		if ( s_pendingFree[i].frame < s_frameNo - 1 ) GpuFree( s_pendingFree[i].p, s_pendingFree[i].size );
		else s_pendingFree[keep++] = s_pendingFree[i];
	s_pendingFree.resize( keep );
	s_cmd = s_dcb[s_back];
	s_cmd += sceGnmDrawInitDefaultHardwareState350( s_cmd, 256 );
	DrawFrameReset();
	memset( s_ctxValid, 0, sizeof( s_ctxValid ) );
	s_clearedBackThisFrame = false;
}

static void *s_screenDepth;								// буфер глубины экрана (FBO 0), Z32F

// Суперсэмплинг: FBO 0 рисуется в буфер 3840x2160, а при показе сжимается до 1080p.
// Ровно вдвое по каждой стороне: линейный фильтр в центре пикселя экрана попадает точно
// между четырьмя пикселями буфера и даёт их среднее. Программа по-прежнему видит 1920x1080.
static int s_ssScale = 1, s_ssWanted = 1;				// сейчас и со следующего кадра
static void *s_ssColor, *s_ssDepth;
static void ResolveSupersampling();

void ps4glSetSupersampling( int scale ) { s_ssWanted = scale == 2 ? 2 : 1; }
int ps4glGetSupersampling( void ) { return s_ssScale; }
int ps4glIsProMode( void ) { return sceKernelIsNeoMode() ? 1 : 0; }

// меняем только между кадрами: внутри кадра все рисования идут в одну цель
static void ApplySupersampling()
{
	if ( s_ssWanted == s_ssScale )
		return;
	if ( s_ssWanted > 1 && !s_ssColor )
	{
		const size_t bytes = (size_t)SCREEN_W * s_ssWanted * SCREEN_H * s_ssWanted * 4;
		void *color = GpuAlloc( bytes, 0x10000 ), *depth = color ? GpuAlloc( bytes, 0x10000 ) : NULL;
		if ( !depth )
		{
			GpuFree( color, bytes );
			s_ssWanted = 1;
			ps4glLog( "ps4gl: not enough video memory for supersampling" );
			return;
		}
		s_ssColor = color;
		s_ssDepth = depth;
	}
	s_ssScale = s_ssWanted;
	ps4glLog( "ps4gl: rendering at %dx%d", SCREEN_W * s_ssScale, SCREEN_H * s_ssScale );
}

int ps4glInit( size_t videoPoolBytes )
{
	if ( s_video >= 0 )
		return 1;
	sceSystemServiceHideSplashScreen();
	s_video = sceVideoOutOpen( ORBIS_VIDEO_USER_MAIN, ORBIS_VIDEO_OUT_BUS_MAIN, 0, 0 );
	if ( s_video < 0 )
	{
		ps4glLog( "ps4gl: sceVideoOutOpen failed: %x", s_video );
		return 0;
	}
	sceKernelCreateEqueue( &s_flipQueue, "ps4gl flip" );
	sceVideoOutAddFlipEvent( s_flipQueue, s_video, 0 );

	// пул прямой памяти для видеокарты: кадры, буферы, шейдеры, текстуры, цели рендера
	const size_t align = 0x200000;
	size_t poolBytes = videoPoolBytes ? videoPoolBytes : (size_t)1 << 30;
	poolBytes = ( poolBytes + align - 1 ) & ~( align - 1 );
	off_t off = 0;
	int ra = sceKernelAllocateDirectMemory( 0, sceKernelGetDirectMemorySize(), poolBytes, align, 3, &off );
	void *mem = NULL;
	int rm = ra < 0 ? -1 : sceKernelMapDirectMemory( &mem, poolBytes, 0x33, 0, off, align );
	if ( !mem )
	{
		ps4glLog( "ps4gl: no video memory (%llu MiB): allocate %x, map %x", (unsigned long long)( poolBytes >> 20 ), ra, rm );
		return 0;
	}
	s_pool = (uint8_t *)mem;
	s_poolEnd = s_pool + poolBytes;
	FreeInsert( (uintptr_t)s_pool, poolBytes );

	const size_t fbBytes = (size_t)SCREEN_W * SCREEN_H * 4;
	for ( int i = 0; i < 3; i++ )
	{
		s_fb[i] = (uint32_t *)PoolAlloc( fbBytes, 0x10000 );
		memset( s_fb[i], 0, fbBytes );
	}
	s_screenDepth = PoolAlloc( fbBytes, 0x10000 );
	for ( int i = 0; i < 2; i++ ) s_dcb[i] = (uint32_t *)PoolAlloc( DCB_BYTES, 0x1000 );
	OrbisVideoOutBufferAttribute attr;
	sceVideoOutSetBufferAttribute( &attr, 0x80000000, 1, 0, SCREEN_W, SCREEN_H, SCREEN_W );
	int rr = sceVideoOutRegisterBuffers( s_video, 0, (void **)s_fb, 3, &attr );
	sceVideoOutSetFlipRate( s_video, 0 );
	ps4glLog( "ps4gl: started, video pool %llu MiB, display buffers registered: %d", (unsigned long long)( poolBytes >> 20 ), rr );
	ps4glLog( "ps4gl: PS4 Pro mode: %s", ps4glIsProMode() ? "on" : "off" );
	s_back = 0; s_fbIdx = 0; s_flipsPending = 0;
	DrawInit();
	ApplySupersampling();
	BeginFrameCommands();
	return 1;
}

void ps4glSwapBuffers( void )
{
	if ( s_video < 0 )
		return;
	FlushCaches();
	// в этот кадр ничего не рисовали: иначе на экран попал бы один из старых кадров
	if ( !s_clearedBackThisFrame && !s_frameDrawsFb0 )
		DmaFill( s_fb[s_fbIdx], s_lastClearBGRA, (size_t)SCREEN_W * SCREEN_H * 4 );
	else if ( s_ssScale > 1 )
		ResolveSupersampling();
	s_frameDrawsFb0 = 0;
	// заготовка для показа: последние 64 слова — NOP с меткой PrepareFlip,
	// без неё sceGnmSubmitAndFlipCommandBuffers возвращает ошибку
	s_cmd[0] = 0xC03E1000u;
	s_cmd[1] = 0x68750777u;
	for ( int k = 2; k < 64; k++ ) s_cmd[k] = 0;
	s_cmd += 64;
	__sync_synchronize();
	void *addrs[1] = { s_dcb[s_back] };
	uint32_t sizes[1] = { (uint32_t)( ( s_cmd - s_dcb[s_back] ) * 4 ) };
	int rs = sceGnmSubmitAndFlipCommandBuffers( 1, addrs, sizes, NULL, NULL, s_video, s_fbIdx, ORBIS_VIDEO_OUT_FLIP_VSYNC, s_frame );
	sceGnmSubmitDone();
	if ( rs != 0 )
	{
		static int s_said;
		if ( s_said++ < 10 ) ps4glLog( "ps4gl: frame %d: submit failed: %x", s_frame, rs );
	}
	// внахлёст: пока видеокарта рисует этот кадр, процессор считает следующий —
	// ждём показа только предыдущего (в очереди на показ не больше одного кадра)
	s_flipsPending++;
	while ( s_flipsPending > 1 )
	{
		OrbisKernelEvent ev;
		int n = 0;
		uint32_t timeout = 500000;
		int rw = sceKernelWaitEqueue( s_flipQueue, &ev, 1, &n, &timeout );
		if ( rw < 0 || n <= 0 ) { s_flipsPending = 0; break; }	// не дождались (зависание?) — не копим
		s_flipsPending--;
	}
	s_frame++;
	s_frameNo = s_frame;
	s_back ^= 1;
	s_fbIdx = ( s_fbIdx + 1 ) % 3;
	ApplySupersampling();
	BeginFrameCommands();
}

// =====================================================================
// Объекты GL
// =====================================================================
static GLuint s_nextName = 1;
static GLuint NewName() { return s_nextName++; }

// ---- буферы: данные — в видеопамяти. Кадр выполняется целиком при показе,
// поэтому если буфер уже нужен рисованию этого кадра, а программа его меняет —
// берём новую память (копию), старую освобождаем после показа.
struct Buffer { void *data; size_t size; int usedFrame; };
static Buffer s_buffers[65536];
static Buffer *FindBuffer( GLuint name ) { return name && name < 65536 ? &s_buffers[name] : NULL; }

static void BufferRealloc( Buffer *b, size_t size, bool keepData )
{
	void *nd = GpuAlloc( size ? size : 4, 256 );
	if ( keepData && b->data && nd ) memcpy( nd, b->data, size < b->size ? size : b->size );
	if ( b->data ) GpuFreeLater( b->data, b->size );
	b->data = nd;
	b->size = size;
	b->usedFrame = -1;
}

// перед записью на CPU: если буфер уже в рисовании этого кадра — работаем с копией
static void BufferBeforeWrite( Buffer *b )
{
	if ( b->data && b->usedFrame >= 0 && b->usedFrame >= s_frameNo - 1 )
		BufferRealloc( b, b->size, true );
}
static GLuint s_boundBuffer[8];
static int BufferSlot( GLenum target )
{
	switch ( target )
	{
	case GL_ARRAY_BUFFER: return 0;
	case GL_ELEMENT_ARRAY_BUFFER: return 1;
	case 0x88EB: return 2;		// PIXEL_PACK
	case 0x88EC: return 3;		// PIXEL_UNPACK
	case 0x8F36: return 4;		// COPY_READ
	case 0x8F37: return 5;		// COPY_WRITE
	case 0x8A11: return 6;		// UNIFORM
	default: return 7;
	}
}

static GLuint s_drawFbo, s_readFbo;
static bool s_fbChanged = true;	// цель сменилась: перед следующим рисованием сбросить кэши (то, что
								// нарисовали в прошлую цель, иначе читается как текстура наполовину старым —
								// рваные полосы bloom: уменьшенная картинка из кэша цвета не дошла до памяти)
static GLfloat s_clearColor[4];
static GLint s_viewport[4] = { 0, 0, SCREEN_W, SCREEN_H };
static GLboolean s_enabled[64];
static int CapSlot( GLenum cap ) { return (int)( cap & 63 ); }

// =====================================================================
// Шейдеры и программы: .sb из tools/shadersbC.py (разбор как в tools/sbformat.py)
// =====================================================================
struct UniformElem { std::string name; int off; int size; int count; };
struct SamplerDecl { std::string name; int index; };
struct Shader
{
	GLenum type = 0;
	bool valid = false;
	void *code = NULL;			// машинный код в видеопамяти (адрес кратен 256)
	size_t codeBytes = 0;
	uint32_t crc = 0;			// FNV-1a кода (tools: найти файл .sb по отпечатку)
	uint32_t hdr[64] = {};		// заголовок Shdr: [4] RSRC1, [5] RSRC2, VS [6] VS_OUT_CONFIG, PS [7] COL_FORMAT…
	int hdrDw = 0;
	std::vector<uint32_t> slots;	// {тип, номер в API, регистр (>=16 — расширенная таблица), доп.}
	int cbSize = 0;
	std::vector<UniformElem> elems;
	std::vector<SamplerDecl> samplers;
	std::vector<std::string> ins, outs;	// у VS outs — без gl_Position
};
static std::map<GLuint, Shader *> s_shaders;

struct ProgUniform { std::string name; int vsOff, psOff, size; int sampler; };	// sampler >= 0 — номер текстуры PS
struct Program
{
	GLuint vs = 0, ps = 0;
	bool linked = false;
	std::map<std::string, GLuint> attribLoc;	// glBindAttribLocation
	int vsAttribLoc[16];						// вход VS k -> номер массива вершин
	int nAttribs = 0;
	int psInputParam[32];						// вход PS k -> параметр VS
	int nPsInputs = 0;
	std::vector<ProgUniform> uniforms;
	std::vector<uint8_t> vsCB, psCB;			// «тень» буферов констант (копируется в кадр при рисовании)
	int samplerUnit[16];
};
static std::map<GLuint, Program *> s_programs;
static GLuint s_curProgram;

static inline uint32_t RdU32( const uint8_t *p ) { uint32_t v; memcpy( &v, p, 4 ); return v; }
static std::string RdStr( const uint8_t *p, const uint8_t *end )
{
	const uint8_t *e = p;
	while ( e < end && *e ) e++;
	return std::string( (const char *)p, e - p );
}

// Разобрать блок glShaderBinary (формат 0): u32 0xE891BC71, 100, 1, len(src), src, u32 len(prog), 0xBEEF0400, prog
static bool ParseSb( Shader *sh, const uint8_t *b, size_t n )
{
	size_t at = 0;
	while ( at + 4 <= n && RdU32( b + at ) != 0xE891BC71u ) at += 4;
	if ( at + 16 > n ) return false;
	b += at; n -= at;
	uint32_t srcLen = RdU32( b + 12 );
	if ( 24 + srcLen > n || RdU32( b + 20 + srcLen ) != 0xBEEF0400u ) return false;
	const uint8_t *prog = b + 24 + srcLen;
	size_t progLen = RdU32( b + 16 + srcLen );
	// в поле длины засчитан и 4-байтный тег 0xBEEF0400 — программа на 4 байта короче
	// (tools/sbformat.py режет срезом и этого не замечает); берём, сколько есть
	if ( 24 + srcLen + progLen > n ) progLen = n - ( 24 + srcLen );
	const uint8_t *progEnd = prog + progLen;
	uint32_t gsize = RdU32( prog + 12 );
	const uint8_t *g = prog + 32;
	if ( memcmp( g, "Shdr", 4 ) ) return false;
	int hdrDw = g[9];
	if ( hdrDw > 64 ) return false;
	sh->hdrDw = hdrDw;
	for ( int i = 0; i < hdrDw; i++ ) sh->hdr[i] = RdU32( g + 16 + i * 4 );
	size_t codeAt = 16 + hdrDw * 4;
	size_t shaderSize = sh->hdr[0] & 0x7FFFFF;
	size_t infoAt = codeAt + shaderSize - 28;
	if ( infoAt + 28 > gsize || memcmp( g + infoAt, "OrbShdr", 7 ) ) return false;
	size_t length = RdU32( g + infoAt + 8 ) >> 8;
	int chunkBase = g[infoAt + 12], nslots = g[infoAt + 13];
	size_t slotsAt = infoAt - ( chunkBase + nslots ) * 4;
	sh->slots.clear();
	for ( int i = 0; i < nslots; i++ ) sh->slots.push_back( RdU32( g + slotsAt + i * 4 ) );
	// код без первой инструкции (s_mov_b32 vcc_hi, K)
	sh->codeBytes = length - 8;
	sh->code = GpuAlloc( sh->codeBytes + 256, 256 );
	if ( !sh->code ) return false;
	memcpy( sh->code, g + codeAt + 8, sh->codeBytes );
	{
		uint32_t h = 2166136261u;
		const uint8_t *c = g + codeAt + 8;
		for ( size_t i = 0; i < sh->codeBytes; i++ ) { h ^= c[i]; h *= 16777619u; }
		sh->crc = h;
	}
	// описание: буферы, константы, элементы, входы, выходы
	const uint8_t *m = g + gsize;
	uint32_t nbuf = RdU32( m ), ncon = RdU32( m + 4 ), nel = RdU32( m + 8 );
	int nin = m[16], nout = m[17];
	size_t o = 28;
	for ( uint32_t i = 0; i < nbuf; i++, o += 24 )
	{
		std::string name = RdStr( m + o + 20 + RdU32( m + o + 20 ), progEnd );
		if ( name == "__GLOBAL_CB__" ) sh->cbSize = (int)RdU32( m + o + 4 );
		else sh->samplers.push_back( { name, (int)RdU32( m + o ) } );
	}
	o += ncon * 32;
	for ( uint32_t i = 0; i < nel; i++, o += 36 )
	{
		UniformElem e;
		e.name = RdStr( m + o + 28 + RdU32( m + o + 28 ), progEnd );
		e.off = (int)RdU32( m + o + 4 );
		e.size = (int)RdU32( m + o + 8 );
		e.count = (int)RdU32( m + o + 12 );
		sh->elems.push_back( e );
	}
	for ( int i = 0; i < nin; i++, o += 16 ) sh->ins.push_back( RdStr( m + o + 8 + RdU32( m + o + 8 ), progEnd ) );
	for ( int i = 0; i < nout; i++, o += 16 )
	{
		std::string name = RdStr( m + o + 8 + RdU32( m + o + 8 ), progEnd );
		if ( name != "gl_Position" && name != "gl_FragColor" ) sh->outs.push_back( name );
	}
	return true;
}

static void ShaderFree( Shader *sh )
{
	if ( sh && sh->code ) GpuFreeLater( sh->code, sh->codeBytes + 256 );
	delete sh;
}

static bool LinkProgram( GLuint id, Program *p )
{
	p->linked = false;
	p->uniforms.clear();
	Shader *vs = s_shaders.count( p->vs ) ? s_shaders[p->vs] : NULL;
	Shader *ps = s_shaders.count( p->ps ) ? s_shaders[p->ps] : NULL;
	if ( !vs || !ps || !vs->valid || !ps->valid ) return false;
	// атрибуты: вход VS k (по имени) -> номер массива вершин из glBindAttribLocation
	p->nAttribs = (int)vs->ins.size();
	if ( p->nAttribs > 16 ) return false;
	for ( int k = 0; k < p->nAttribs; k++ )
	{
		auto it = p->attribLoc.find( vs->ins[k] );
		p->vsAttribLoc[k] = it != p->attribLoc.end() ? (int)it->second : k;
	}
	// входы PS -> параметры VS по именам
	p->nPsInputs = (int)ps->ins.size();
	for ( int k = 0; k < p->nPsInputs && k < 32; k++ )
	{
		p->psInputParam[k] = -1;
		for ( size_t j = 0; j < vs->outs.size(); j++ )
			if ( vs->outs[j] == ps->ins[k] ) { p->psInputParam[k] = (int)j; break; }
		if ( p->psInputParam[k] < 0 )
		{
			static int s_said;
			if ( s_said++ < 10 ) ps4glLog( "ps4gl: program %u: pixel shader input \"%s\" is missing in the vertex shader", id, ps->ins[k].c_str() );
			p->psInputParam[k] = 0;
		}
	}
	// uniform: общий список по именам, у каждого — место в константах VS и/или PS
	auto add = [&]( const UniformElem &e, bool isVs ) {
		for ( auto &u : p->uniforms )
			if ( u.name == e.name ) { ( isVs ? u.vsOff : u.psOff ) = e.off; if ( e.size > u.size ) u.size = e.size; return; }
		ProgUniform u = { e.name, isVs ? e.off : -1, isVs ? -1 : e.off, e.size, -1 };
		p->uniforms.push_back( u );
	};
	for ( auto &e : vs->elems ) add( e, true );
	for ( auto &e : ps->elems ) add( e, false );
	for ( auto &sd : ps->samplers )
	{
		ProgUniform u = { sd.name, -1, -1, 0, sd.index };
		p->uniforms.push_back( u );
	}
	p->vsCB.assign( vs->cbSize > 0 ? vs->cbSize : 16, 0 );
	p->psCB.assign( ps->cbSize > 0 ? ps->cbSize : 16, 0 );
	for ( int i = 0; i < 16; i++ ) p->samplerUnit[i] = i;
	p->linked = true;
	return true;
}

static Program *CurProgram() { return s_programs.count( s_curProgram ) ? s_programs[s_curProgram] : NULL; }

// запись значения uniform: location = номер_в_списке * 1024 + индекс_в_массиве
static void UniformWrite( GLint loc, const void *data, size_t bytes, bool scalarInt, GLint ival )
{
	Program *p = CurProgram();
	if ( !p || loc < 0 ) return;
	size_t i = (size_t)loc >> 10, idx = (size_t)loc & 1023;
	if ( i >= p->uniforms.size() ) return;
	ProgUniform &u = p->uniforms[i];
	if ( u.sampler >= 0 )
	{
		if ( scalarInt && u.sampler < 16 ) p->samplerUnit[u.sampler] = ival;
		return;
	}
	auto put = [&]( std::vector<uint8_t> &cb, int off ) {
		if ( off < 0 ) return;
		size_t at = (size_t)off + idx * 16, lim = (size_t)off + (size_t)u.size;
		if ( at >= lim || at >= cb.size() ) return;
		size_t n = bytes;
		if ( at + n > lim ) n = lim - at;
		if ( at + n > cb.size() ) n = cb.size() - at;
		memcpy( &cb[at], data, n );
	};
	put( p->vsCB, u.vsOff );
	put( p->psCB, u.psOff );
}

// =====================================================================
// Текстуры, FBO, состояние, рисование, перенос кадра
// =====================================================================
static inline uint32_t F2U( float f ) { uint32_t u; memcpy( &u, &f, 4 ); return u; }
// «тень» регистров контекста (0xA000..0xA3FF): каждая смена контекстного регистра
// заставляет видеокарту заводить новый контекст (их 8) и ждать старые рисования —
// поэтому пишем только то, что изменилось. Сбрасывается в начале кадра
// (InitDefaultHardwareState всё переставляет).
static void SetRegs( uint32_t op, uint32_t base, uint32_t reg, const uint32_t *v, uint32_t n )
{
	if ( op == 0x69 && reg >= 0xA000 && reg + n <= 0xA400 )
	{
		uint32_t r0 = reg - 0xA000, i = 0;
		while ( i < n && s_ctxValid[r0 + i] && s_ctxShadow[r0 + i] == v[i] ) i++;
		if ( i == n ) return;										// всё уже так
		for ( uint32_t k = 0; k < n; k++ ) { s_ctxShadow[r0 + k] = v[k]; s_ctxValid[r0 + k] = 1; }
	}
	*s_cmd++ = Pkt3( op, n + 1 );
	*s_cmd++ = reg - base;
	for ( uint32_t i = 0; i < n; i++ ) *s_cmd++ = v[i];
}
static void Ctx( uint32_t reg, uint32_t v ) { SetRegs( 0x69, 0xA000, reg, &v, 1 ); }
static void Uc( uint32_t reg, uint32_t v ) { SetRegs( 0x79, 0xC000, reg, &v, 1 ); }

// сбросить кэши цвета/глубины в память и сбросить кэш текстур (перед очисткой DMA,
// перед чтением цели рендера как текстуры и перед показом)
static void FlushCaches()
{
	*s_cmd++ = Pkt3( 0x46, 1 ); *s_cmd++ = 0x16;			// EVENT_WRITE: CACHE_FLUSH_AND_INV_EVENT
	*s_cmd++ = Pkt3( 0x43, 4 );								// SURFACE_SYNC
	*s_cmd++ = ( 1u << 25 ) | ( 1u << 26 ) | ( 1u << 23 ) | ( 1u << 22 ) | ( 1u << 6 );	// CB, DB, TC, TCL1
	*s_cmd++ = 0xFFFFFFFFu; *s_cmd++ = 0; *s_cmd++ = 10;
}

static bool CmdRoom( size_t dwords )
{
	if ( (size_t)( s_cmd - s_dcb[s_back] ) + dwords + 1024 < DCB_BYTES / 4 ) return true;
	static int s_said;
	if ( s_said++ < 5 ) ps4glLog( "ps4gl: command buffer is full, draw skipped" );
	return false;
}

// ---- данные рисований на кадр (константы, таблицы V#, вершины переноса) ----
static uint8_t *s_upload[2];
static const size_t UPLOAD_BYTES = 48u << 20;
static size_t s_uploadOff;
static void *Upload( const void *src, size_t bytes, size_t align = 256 )
{
	size_t o = ( s_uploadOff + align - 1 ) & ~( align - 1 );
	if ( o + bytes > UPLOAD_BYTES )
	{
		static int s_said;
		if ( s_said++ < 5 ) ps4glLog( "ps4gl: per-frame upload memory is full" );
		return NULL;
	}
	s_uploadOff = o + bytes;
	uint8_t *p = s_upload[s_back] + o;
	if ( src ) memcpy( p, src, bytes );
	return p;
}

// ---- текстуры: раскладка «линейная» (индекс 8) ----
struct Texture
{
	GLenum target = 0;
	int w = 0, h = 0, pitch = 0, rows = 0;	// pitch и rows — в элементах (у BC — в блоках)
	int bpe = 4, block = 1;					// байт на элемент, размер блока (4 у BC)
	uint32_t dfmt = 10, nfmt = 0, sel = 0;	// формат T#, перестановка каналов
	bool bgra = false, depth = false;
	int cbFormat = 0xA, cbNumber = 0;		// формат как цели рендера (CB)
	void *mem = NULL;
	size_t bytes = 0;
	GLenum wrapS = 0x2901, wrapT = 0x2901, minF = 0x2601, magF = 0x2601;
	// мип-уровни (только RGBA8 со сторонами — степенями двойки): раскладка как у
	// «линейной выровненной» (shadPS4 ImageSizeLinearAligned): шаг уровня кратен 16
	// пикселям, уровни подряд; уровень 0 — как раньше (шаг кратен 64, у степеней двойки
	// >= 64 это одно и то же). Цепочка до 16x4 — у мелких уровней правила не проверены.
	int levels = 1;
	uint32_t filled = 1;					// какие уровни уже загружены
	size_t mipOff[16] = {};
	int mipPitch[16] = {}, mipW[16] = {}, mipH[16] = {};
};
static std::map<GLuint, Texture *> s_textures;
static GLuint s_activeUnit;
static GLuint s_bound2D[32];
static bool s_unitNot2D[32];				// в слот последней привязали куб/3D — их пока нет: даём чёрную
struct SamplerObj { GLenum wrapS = 0x2901, wrapT = 0x2901, minF = 0x2601, magF = 0x2601; };
static std::map<GLuint, SamplerObj> s_samplers;
static GLuint s_boundSampler[32];
static GLint s_unpackAlign = 4;

static const uint32_t SEL_RGBA = 4u | ( 5u << 3 ) | ( 6u << 6 ) | ( 7u << 9 );
static const uint32_t SEL_BGRA = 6u | ( 5u << 3 ) | ( 4u << 6 ) | ( 7u << 9 );

static Texture *TexOf( GLuint t ) { auto it = s_textures.find( t ); return it == s_textures.end() ? NULL : it->second; }

// выделить память под уровень 0 (под цель рендера — с запасом по 8 строк)
static bool TexAlloc( Texture *t, int w, int h )
{
	int ew = ( w + t->block - 1 ) / t->block, eh = ( h + t->block - 1 ) / t->block;
	int pitch = ( ew + 63 ) & ~63, rows = ( eh + 7 ) & ~7;
	size_t bytes = (size_t)pitch * rows * t->bpe;
	int levels = 1;
	size_t mipOff[16] = {};
	int mipPitch[16] = { pitch }, mipW[16] = { w }, mipH[16] = { h };
	bool pow2 = w >= 64 && h >= 4 && !( w & ( w - 1 ) ) && !( h & ( h - 1 ) );
	if ( pow2 && t->bpe == 4 && t->block == 1 && !t->depth )
	{
		size_t off = (size_t)pitch * h * t->bpe;
		for ( int l = 1; l < 16; l++ )
		{
			int wl = w >> l, hl = h >> l;
			if ( wl < 16 || hl < 4 ) break;
			off = ( off + 255 ) & ~(size_t)255;
			mipOff[l] = off; mipPitch[l] = ( wl + 15 ) & ~15; mipW[l] = wl; mipH[l] = hl;
			off += (size_t)mipPitch[l] * hl * t->bpe;
			levels = l + 1;
		}
		if ( off > bytes ) bytes = off;
	}
	bool same = t->mem && t->bytes == bytes && t->w == w && t->h == h && t->levels == levels;
	if ( !same )
	{
		if ( t->mem ) GpuFreeLater( t->mem, t->bytes );
		t->mem = GpuAlloc( bytes, 0x10000 );
		t->bytes = bytes;
		if ( t->mem ) memset( t->mem, 0, bytes );
		t->filled = 1;
	}
	t->w = w; t->h = h; t->pitch = pitch; t->rows = rows; t->levels = levels;
	memcpy( t->mipOff, mipOff, sizeof( mipOff ) ); memcpy( t->mipPitch, mipPitch, sizeof( mipPitch ) );
	memcpy( t->mipW, mipW, sizeof( mipW ) ); memcpy( t->mipH, mipH, sizeof( mipH ) );
	return t->mem != NULL;
}
// сколько уровней подряд от 0 загружено (минус один) — это LAST_LEVEL у T#
static int TexLastLevel( const Texture *t )
{
	int last = 0;
	while ( last + 1 < t->levels && ( t->filled >> ( last + 1 ) & 1 ) ) last++;
	return last;
}

static int FmtComps( GLenum fmt )
{
	switch ( fmt )
	{
	case GL_ALPHA: case GL_LUMINANCE: case 0x1903: case 0x1902: return 1;		// RED, DEPTH
	case GL_LUMINANCE_ALPHA: case 0x8227: case 0x84F9: return 2;				// RG, DEPTH_STENCIL
	case GL_RGB: case 0x80E0: return 3;
	}
	return 4;
}
static int TypeBytes( GLenum type )
{
	switch ( type )
	{
	case GL_UNSIGNED_BYTE: case GL_BYTE: return 1;
	case GL_UNSIGNED_SHORT: case GL_SHORT: case 0x140B: case 0x8D61: return 2;
	}
	return 4;
}

// формат хранения по параметрам glTexImage2D
static void TexChooseFormat( Texture *t, GLenum ifmt, GLenum fmt, GLenum type )
{
	t->block = 1; t->bpe = 4; t->dfmt = 10; t->nfmt = 0; t->sel = SEL_RGBA; t->bgra = false; t->depth = false;
	t->cbFormat = 0xA; t->cbNumber = 0;
	bool srgb = ifmt == 0x8C43 || ifmt == 0x8C41 || ifmt == 0x8C47 || ifmt == 0x8C45;	// SRGB8_ALPHA8, SRGB8, SLUMINANCE8…
	if ( fmt == 0x1902 || fmt == 0x84F9 )					// DEPTH_COMPONENT, DEPTH_STENCIL
	{ t->depth = true; t->dfmt = 4; t->nfmt = 7; t->sel = 4u | ( 4u << 3 ) | ( 4u << 6 ) | ( 1u << 9 ); return; }
	if ( type == GL_UNSIGNED_SHORT && !( fmt == 0x1902 || fmt == 0x84F9 ) )
	{
		// 16 бит на канал без знака — формат 16_16_16_16 UNORM;
		// 3 компоненты храним как 4 (недостающие — нулями/единицей в ConvertRow)
		t->bpe = 8; t->dfmt = 12; t->nfmt = 0; t->cbFormat = 0xC; t->cbNumber = 0;
		int comps = FmtComps( fmt );
		if ( comps == 1 ) t->sel = 4u | ( 4u << 3 ) | ( 4u << 6 ) | ( 1u << 9 );
		else if ( comps == 2 ) t->sel = 4u | ( 4u << 3 ) | ( 4u << 6 ) | ( 5u << 9 );
		else if ( comps == 3 ) t->sel = 4u | ( 5u << 3 ) | ( 6u << 6 ) | ( 1u << 9 );
		return;
	}
	bool half = type == 0x140B || type == 0x8D61;
	if ( half || type == GL_FLOAT )
	{
		// 1..4 компоненты; 3 храним как 4 (у цели рендера 3-компонентных float нет)
		int comps = FmtComps( fmt );
		static const uint32_t dHalf[5] = { 0, 2, 5, 12, 12 }, dFloat[5] = { 0, 4, 11, 14, 14 };
		static const uint32_t cHalf[5] = { 0, 2, 5, 0xC, 0xC }, cFloat[5] = { 0, 4, 0xB, 0xE, 0xE };
		int store = comps == 3 ? 4 : comps;
		t->bpe = store * ( half ? 2 : 4 );
		t->dfmt = half ? dHalf[comps] : dFloat[comps];
		t->cbFormat = half ? cHalf[comps] : cFloat[comps];
		t->nfmt = 7; t->cbNumber = 7;
		if ( fmt == GL_LUMINANCE ) t->sel = 4u | ( 4u << 3 ) | ( 4u << 6 ) | ( 1u << 9 );
		else if ( fmt == GL_ALPHA ) t->sel = 0u | ( 0u << 3 ) | ( 0u << 6 ) | ( 4u << 9 );
		else if ( fmt == GL_LUMINANCE_ALPHA ) t->sel = 4u | ( 4u << 3 ) | ( 4u << 6 ) | ( 5u << 9 );
		else if ( comps == 1 ) t->sel = 4u | ( 0u << 3 ) | ( 0u << 6 ) | ( 1u << 9 );
		else if ( comps == 2 ) t->sel = 4u | ( 5u << 3 ) | ( 0u << 6 ) | ( 1u << 9 );
		else if ( comps == 3 ) t->sel = 4u | ( 5u << 3 ) | ( 6u << 6 ) | ( 1u << 9 );
		return;
	}
	if ( fmt == 0x80E1 && ( type == GL_UNSIGNED_BYTE || type == 0x8367 ) )	// BGRA, 8_8_8_8_REV
		t->bgra = true, t->sel = SEL_BGRA;
	if ( srgb ) t->nfmt = 9, t->cbNumber = 6;
}

static void TexChooseCompressed( Texture *t, GLenum ifmt )
{
	t->block = 4; t->sel = SEL_RGBA; t->bgra = false; t->depth = false; t->nfmt = 0;
	switch ( ifmt )
	{
	case 0x83F0: case 0x83F1: case 0x8C4C: case 0x8C4D: t->dfmt = 35; t->bpe = 8; break;	// DXT1 -> BC1
	case 0x83F2: case 0x8C4E: t->dfmt = 36; t->bpe = 16; break;							// DXT3 -> BC2
	default: t->dfmt = 37; t->bpe = 16; break;												// DXT5 -> BC3
	}
	if ( ifmt >= 0x8C4C && ifmt <= 0x8C4F ) t->nfmt = 9;
}

// одна строка исходных пикселей -> строка хранения (RGBA8 или как есть)
static void ConvertRow( const Texture *t, uint8_t *dst, const uint8_t *src, int w, GLenum fmt, GLenum type )
{
	if ( t->bpe != 4 || t->depth || t->nfmt == 7 )
	{
		int se = FmtComps( fmt ) * TypeBytes( type );			// байт на пиксель в источнике
		if ( type == 0x84FA || type == 0x8367 ) se = 4;			// UNSIGNED_INT_24_8 и т. п. — одно слово
		if ( se == t->bpe ) { memcpy( dst, src, (size_t)w * se ); return; }
		int n = se < t->bpe ? se : t->bpe;
		for ( int x = 0; x < w; x++ ) { memcpy( dst + (size_t)x * t->bpe, src + (size_t)x * se, n ); if ( n < t->bpe ) memset( dst + (size_t)x * t->bpe + n, 0, t->bpe - n ); }
		return;
	}
	if ( type == GL_UNSIGNED_BYTE || type == 0x8367 )
	{
		switch ( fmt )
		{
		case GL_RGBA: case 0x80E1: memcpy( dst, src, (size_t)w * 4 ); return;
		case GL_RGB: for ( int x = 0; x < w; x++ ) { dst[4 * x] = src[3 * x]; dst[4 * x + 1] = src[3 * x + 1]; dst[4 * x + 2] = src[3 * x + 2]; dst[4 * x + 3] = 255; } return;
		case 0x80E0: for ( int x = 0; x < w; x++ ) { dst[4 * x] = src[3 * x + 2]; dst[4 * x + 1] = src[3 * x + 1]; dst[4 * x + 2] = src[3 * x]; dst[4 * x + 3] = 255; } return;	// BGR
		case GL_LUMINANCE: for ( int x = 0; x < w; x++ ) { dst[4 * x] = dst[4 * x + 1] = dst[4 * x + 2] = src[x]; dst[4 * x + 3] = 255; } return;
		case GL_ALPHA: for ( int x = 0; x < w; x++ ) { dst[4 * x] = dst[4 * x + 1] = dst[4 * x + 2] = 0; dst[4 * x + 3] = src[x]; } return;
		case GL_LUMINANCE_ALPHA: for ( int x = 0; x < w; x++ ) { dst[4 * x] = dst[4 * x + 1] = dst[4 * x + 2] = src[2 * x]; dst[4 * x + 3] = src[2 * x + 1]; } return;
		}
	}
	const uint16_t *s16 = (const uint16_t *)src;
	for ( int x = 0; x < w; x++ )
	{
		uint16_t v = s16[x];
		uint8_t r, g, b, a;
		if ( type == 0x8365 ) { a = ( v >> 12 ) * 17; r = ( ( v >> 8 ) & 15 ) * 17; g = ( ( v >> 4 ) & 15 ) * 17; b = ( v & 15 ) * 17; }		// 4_4_4_4_REV (A4R4G4B4)
		else if ( type == 0x8366 ) { a = ( v >> 15 ) ? 255 : 0; r = ( ( v >> 10 ) & 31 ) * 255 / 31; g = ( ( v >> 5 ) & 31 ) * 255 / 31; b = ( v & 31 ) * 255 / 31; }	// 1_5_5_5_REV
		else { r = ( v >> 11 ) * 255 / 31; g = ( ( v >> 5 ) & 63 ) * 255 / 63; b = ( v & 31 ) * 255 / 31; a = 255; }						// 5_6_5
		dst[4 * x] = r; dst[4 * x + 1] = g; dst[4 * x + 2] = b; dst[4 * x + 3] = a;
	}
}

static int SrcBytesPerPixel( GLenum fmt, GLenum type, const Texture *t )
{
	if ( t->bpe != 4 || t->depth || t->nfmt == 7 )
		return ( type == 0x84FA || type == 0x8367 ) ? 4 : FmtComps( fmt ) * TypeBytes( type );
	if ( type == 0x8365 || type == 0x8366 || type == 0x8363 ) return 2;
	switch ( fmt ) { case GL_RGB: case 0x80E0: return 3; case GL_LUMINANCE: case GL_ALPHA: return 1; case GL_LUMINANCE_ALPHA: return 2; }
	return 4;
}

static const uint8_t *UnpackSource( const void *pixels )
{
	if ( s_boundBuffer[3] )		// PIXEL_UNPACK_BUFFER: pixels — смещение в буфере
	{
		Buffer *b = FindBuffer( s_boundBuffer[3] );
		return b && b->data ? (const uint8_t *)b->data + (uintptr_t)pixels : NULL;
	}
	return (const uint8_t *)pixels;
}

static void TexUpload( Texture *t, int x0, int y0, int w, int h, GLenum fmt, GLenum type, const void *pixels, int level = 0 )
{
	const uint8_t *src = UnpackSource( pixels );
	if ( !src || !t->mem || level < 0 || level >= t->levels ) return;
	int sbpp = SrcBytesPerPixel( fmt, type, t );
	size_t srow = ( (size_t)w * sbpp + s_unpackAlign - 1 ) / s_unpackAlign * s_unpackAlign;
	uint8_t *base = (uint8_t *)t->mem + t->mipOff[level];
	int pitch = t->mipPitch[level], maxRows = level ? t->mipH[level] : t->rows, maxW = level ? t->mipW[level] : t->pitch;
	if ( x0 + w > maxW ) w = maxW - x0;
	if ( w <= 0 ) return;
	for ( int y = 0; y < h && y0 + y < maxRows; y++ )
		ConvertRow( t, base + ( (size_t)( y0 + y ) * pitch + x0 ) * t->bpe, src + y * srow, w, fmt, type );
}

static Texture *BoundTex( GLenum target )
{
	if ( target != GL_TEXTURE_2D ) return NULL;
	return TexOf( s_bound2D[s_activeUnit & 31] );
}

static void TSharp( uint32_t *d, const Texture *t )
{
	uint64_t a = (uint64_t)(uintptr_t)t->mem;
	d[0] = (uint32_t)( a >> 8 );
	d[1] = (uint32_t)( a >> 40 & 0xFF ) | ( t->dfmt << 20 ) | ( t->nfmt << 26 );
	d[2] = (uint32_t)( t->w - 1 ) | ( (uint32_t)( t->h - 1 ) << 14 );
	d[3] = t->sel | ( (uint32_t)TexLastLevel( t ) << 16 ) | ( 8u << 20 ) | ( 9u << 28 );	// LAST_LEVEL, раскладка 8, 2D
	d[4] = (uint32_t)( t->pitch * t->block - 1 ) << 13;			// шаг строки в пикселях
	d[5] = d[6] = d[7] = 0;
}

static uint32_t WrapMode( GLenum w ) { return w == 0x8370 ? 1 : w == 0x812F ? 2 : w == 0x812D ? 6 : 0; }
static uint32_t FilterMode( GLenum f ) { return ( f == GL_NEAREST || f == GL_NEAREST_MIPMAP_NEAREST || f == GL_NEAREST_MIPMAP_LINEAR ) ? 0 : 1; }
static void SSharp( uint32_t *d, GLenum ws, GLenum wt, GLenum minF, GLenum magF, int lastLevel = 0 )
{
	// мип-фильтр: *_MIPMAP_NEAREST — ближайший уровень, *_MIPMAP_LINEAR — смесь двух
	uint32_t mip = ( minF == 0x2700 || minF == 0x2701 ) ? 1 : ( minF == 0x2702 || minF == 0x2703 ) ? 2 : 0;
	if ( lastLevel <= 0 ) mip = 0;
	d[0] = WrapMode( ws ) | ( WrapMode( wt ) << 3 ) | ( 2u << 6 );
	d[1] = (uint32_t)( lastLevel > 0 ? lastLevel * 256 : 0 ) << 12;		// MAX_LOD (4.8)
	d[2] = ( FilterMode( magF ) << 20 ) | ( FilterMode( minF ) << 22 ) | ( mip << 26 );
	d[3] = 0;
}

// пустая текстура 1x1 (белая) — когда нужной нет
static Texture s_whiteTex, s_blackTex;

// ---- FBO и рендербуферы ----
struct Fbo { GLuint color = 0, depth = 0; bool colorRb = false, depthRb = false; };
static std::map<GLuint, Fbo> s_fbos;
static std::map<GLuint, Texture *> s_rbs;		// рендербуфер хранится как текстура
static GLuint s_boundRb;

// куда рисовать: цель цвета и глубины
struct Target
{
	void *color = NULL; int pitch = 0, w = 0, h = 0, rows = 0; uint32_t cbFormat = 0xA, cbNumber = 0, swap = 0;
	bool flipY = false;				// кадр на экран: строка 0 — сверху (у GL — снизу)
	void *depth = NULL; int dpitch = 0, drows = 0;
	int scale = 1;					// во сколько раз цель больше координат программы (суперсэмплинг FBO 0)
};

static bool GetTarget( GLuint fboId, Target &t )
{
	if ( fboId == 0 )
	{
		const bool ss = s_ssScale > 1;
		t.scale = s_ssScale;
		t.color = ss ? s_ssColor : s_fb[s_fbIdx];
		t.pitch = t.w = SCREEN_W * t.scale; t.h = t.rows = SCREEN_H * t.scale;
		t.cbFormat = 0xA; t.cbNumber = 0; t.swap = ss ? 0 : 1; t.flipY = true;	// большой буфер — RGBA, как текстура
		t.depth = ss ? s_ssDepth : s_screenDepth; t.dpitch = t.pitch; t.drows = t.rows;
		return true;
	}
	auto it = s_fbos.find( fboId );
	if ( it == s_fbos.end() ) return false;
	Texture *c = it->second.colorRb ? ( s_rbs.count( it->second.color ) ? s_rbs[it->second.color] : NULL ) : TexOf( it->second.color );
	if ( !c || !c->mem || c->depth ) return false;
	t.color = c->mem; t.pitch = c->pitch; t.w = c->w; t.h = c->h; t.rows = c->rows;
	t.cbFormat = c->cbFormat; t.cbNumber = c->cbNumber; t.swap = c->bgra ? 1 : 0; t.flipY = false;
	Texture *d = it->second.depthRb ? ( s_rbs.count( it->second.depth ) ? s_rbs[it->second.depth] : NULL ) : TexOf( it->second.depth );
	if ( d && d->mem && d->depth ) { t.depth = d->mem; t.dpitch = d->pitch; t.drows = d->rows; }
	return true;
}

// ---- состояние GL, нужное рисованию ----
static GLenum s_blendSrc = GL_ONE, s_blendDst = GL_ZERO, s_blendEq = 0x8006;
static GLenum s_blendSrcA = GL_ONE, s_blendDstA = GL_ZERO, s_blendEqA = 0x8006;
static GLenum s_cullFace = GL_BACK;
static GLfloat s_blendColor[4];
static GLenum s_depthFunc = GL_LESS;
static GLboolean s_depthMask = GL_TRUE;
static GLfloat s_polyFactor, s_polyUnits;		// glPolygonOffset: наклейки и оверлеи «над» поверхностью
static GLfloat s_depthNear = 0, s_depthFar = 1, s_clearDepth = 1;
static GLboolean s_colorMask[4] = { 1, 1, 1, 1 };
static GLint s_scissor[4] = { 0, 0, SCREEN_W, SCREEN_H };
static GLenum s_frontFace = GL_CCW;
struct Attrib { bool enabled = false; GLint size = 4; GLenum type = GL_FLOAT; GLboolean norm = 0; GLsizei stride = 0; GLuint buffer = 0; uintptr_t offset = 0; };
static Attrib s_attr[16];

static uint32_t BlendFactor( GLenum f )
{
	switch ( f )
	{
	case GL_ZERO: return 0; case GL_ONE: return 1; case GL_SRC_COLOR: return 2; case GL_ONE_MINUS_SRC_COLOR: return 3;
	case GL_SRC_ALPHA: return 4; case GL_ONE_MINUS_SRC_ALPHA: return 5; case GL_DST_ALPHA: return 6; case GL_ONE_MINUS_DST_ALPHA: return 7;
	case GL_DST_COLOR: return 8; case GL_ONE_MINUS_DST_COLOR: return 9; case GL_SRC_ALPHA_SATURATE: return 10;
	case GL_CONSTANT_COLOR: return 13; case GL_ONE_MINUS_CONSTANT_COLOR: return 14;
	case GL_CONSTANT_ALPHA: return 15; case GL_ONE_MINUS_CONSTANT_ALPHA: return 16;
	}
	return 1;
}
static uint32_t BlendFunc( GLenum e ) { return e == 0x800A ? 1 : e == 0x800B ? 4 : e == 0x8007 ? 2 : e == 0x8008 ? 3 : 0; }
static uint32_t CompareFunc( GLenum f ) { return f >= 0x0200 && f <= 0x0207 ? f - 0x0200 : 1; }

// цель + окно + ножницы + глубина + смешивание; blendOn/depthOn — с учётом GL
static void EmitTargetState( const Target &t, bool useGlState )
{
	uint64_t a = (uint64_t)(uintptr_t)t.color;
	Ctx( 0xA318, (uint32_t)( a >> 8 ) );						// CB_COLOR0_BASE
	Ctx( 0xA319, (uint32_t)( t.pitch / 8 - 1 ) );				// CB_COLOR0_PITCH
	Ctx( 0xA31A, (uint32_t)( (size_t)t.pitch * t.rows / 64 - 1 ) );	// CB_COLOR0_SLICE
	Ctx( 0xA31B, 0 );											// CB_COLOR0_VIEW
	Ctx( 0xA31C, ( t.cbFormat << 2 ) | ( t.cbNumber << 8 ) | ( t.swap << 11 ) );	// CB_COLOR0_INFO
	Ctx( 0xA31D, 8 );											// CB_COLOR0_ATTRIB: линейная раскладка
	uint32_t mask = useGlState ? ( ( s_colorMask[0] ? 1 : 0 ) | ( s_colorMask[1] ? 2 : 0 ) | ( s_colorMask[2] ? 4 : 0 ) | ( s_colorMask[3] ? 8 : 0 ) ) : 0xF;
	Ctx( 0xA08E, mask );										// CB_TARGET_MASK
	Ctx( 0xA08F, 0xF );											// CB_SHADER_MASK
	Ctx( 0xA202, ( 1u << 4 ) | ( 0xCCu << 16 ) );				// CB_COLOR_CONTROL
	bool blend = useGlState && s_enabled[CapSlot( GL_BLEND )];
	uint32_t bs = BlendFactor( s_blendSrc ), bd = BlendFactor( s_blendDst ), be = BlendFunc( s_blendEq );
	uint32_t bsA = BlendFactor( s_blendSrcA ), bdA = BlendFactor( s_blendDstA ), beA = BlendFunc( s_blendEqA );
	uint32_t sepA = ( bsA != bs || bdA != bd || beA != be ) ? ( 1u << 29 ) : 0;
	Ctx( 0xA1E0, blend ? ( bs | ( be << 5 ) | ( bd << 8 ) | ( bsA << 16 ) | ( beA << 21 ) | ( bdA << 24 ) | sepA | ( 1u << 30 ) ) : 0 );	// CB_BLEND0_CONTROL
	const uint32_t bc[4] = { F2U( s_blendColor[0] ), F2U( s_blendColor[1] ), F2U( s_blendColor[2] ), F2U( s_blendColor[3] ) };
	SetRegs( 0x69, 0xA000, 0xA105, bc, 4 );						// CB_BLEND_RED..ALPHA
	// глубина: Z_32_FLOAT, плитки Depth1DThin (индекс 5)
	bool depthOn = useGlState && t.depth && s_enabled[CapSlot( GL_DEPTH_TEST )];
	if ( t.depth )
	{
		uint64_t da = (uint64_t)(uintptr_t)t.depth;
		Ctx( 0xA00F, 2u << 4 );									// DB_DEPTH_INFO: 1D_TILED_THIN1
		Ctx( 0xA010, 3u | ( 5u << 20 ) );						// DB_Z_INFO
		Ctx( 0xA011, 0 );										// DB_STENCIL_INFO
		const uint32_t db[4] = { (uint32_t)( da >> 8 ), (uint32_t)( da >> 8 ), (uint32_t)( da >> 8 ), (uint32_t)( da >> 8 ) };
		SetRegs( 0x69, 0xA000, 0xA012, db, 4 );
		Ctx( 0xA016, (uint32_t)( t.dpitch / 8 - 1 ) | ( (uint32_t)( t.drows / 8 - 1 ) << 11 ) );
		Ctx( 0xA017, (uint32_t)( (size_t)t.dpitch * t.drows / 64 - 1 ) );
	}
	else
	{
		Ctx( 0xA010, 0 ); Ctx( 0xA011, 0 );
	}
	Ctx( 0xA002, 0 ); Ctx( 0xA005, 0 );
	Ctx( 0xA003, 2u | ( 2u << 2 ) | ( 2u << 4 ) );				// DB_RENDER_OVERRIDE: HiZ/HiS выкл
	Ctx( 0xA200, depthOn ? ( ( 1u << 1 ) | ( s_depthMask ? ( 1u << 2 ) : 0 ) | ( CompareFunc( s_depthFunc ) << 4 ) ) : 0 );
	Ctx( 0xA000, 0 );
	// окно: GL — снизу вверх. В целях рендера строка 0 = низ (как у GL текстур), на экране — верх
	int vx = useGlState ? s_viewport[0] * t.scale : 0, vy = useGlState ? s_viewport[1] * t.scale : 0;
	int vw = useGlState ? s_viewport[2] * t.scale : t.w, vh = useGlState ? s_viewport[3] * t.scale : t.h;
	float ys = vh / 2.0f, yo = vy + vh / 2.0f;
	if ( t.flipY ) { ys = -ys; yo = t.h - yo; }
	const uint32_t vp[6] = { F2U( vw / 2.0f ), F2U( vx + vw / 2.0f ), F2U( ys ), F2U( yo ),
	                         F2U( ( s_depthFar - s_depthNear ) / 2 ), F2U( ( s_depthFar + s_depthNear ) / 2 ) };
	SetRegs( 0x69, 0xA000, 0xA10F, vp, 6 );
	Ctx( 0xA206, 0x43F );										// PA_CL_VTE_CNTL
	Ctx( 0xA204, 0 );											// PA_CL_CLIP_CNTL
	{
		// защитная полоса
		const uint32_t gb[4] = { F2U( 1.0f ), F2U( 1.0f ), F2U( 1.0f ), F2U( 1.0f ) };
		SetRegs( 0x69, 0xA000, 0xA2FA, gb, 4 );					// PA_CL_GB_VERT_CLIP/DISC, HORZ_CLIP/DISC
	}
	uint32_t su = 0;
	if ( useGlState && s_enabled[CapSlot( GL_CULL_FACE )] )
	{
		// видеокарта считает обход в экранных координатах (Y вниз), GL — с Y вверх:
		// в целях без переворота (FBO) числа Y те же, значит обход виден зеркально
		bool cw = s_frontFace == GL_CW;
		if ( !t.flipY ) cw = !cw;
		su = ( s_cullFace != GL_BACK ? 1u : 0 ) | ( s_cullFace != GL_FRONT ? ( 1u << 1 ) : 0 ) | ( cw ? ( 1u << 2 ) : 0 );	// CULL_FRONT, CULL_BACK, FACE
	}
	// смещение глубины (glPolygonOffset). Глубина Z_32_FLOAT — как в Mesa radeonsi:
	// NEG_NUM_DB_BITS = -23, DB_IS_FLOAT_FMT, scale = factor * 16, offset = units
	if ( useGlState && t.depth && s_enabled[CapSlot( 0x8037 )] && ( s_polyFactor != 0.0f || s_polyUnits != 0.0f ) )
	{
		su |= ( 1u << 11 ) | ( 1u << 12 );						// POLY_OFFSET_FRONT/BACK_ENABLE
		Ctx( 0xA2DE, ( (uint32_t)-23 & 0xFF ) | ( 1u << 8 ) );	// PA_SU_POLY_OFFSET_DB_FMT_CNTL
		Ctx( 0xA2DF, 0 );										// PA_SU_POLY_OFFSET_CLAMP
		const uint32_t po[4] = { F2U( s_polyFactor * 16.0f ), F2U( s_polyUnits ), F2U( s_polyFactor * 16.0f ), F2U( s_polyUnits ) };
		SetRegs( 0x69, 0xA000, 0xA2E0, po, 4 );					// FRONT_SCALE, FRONT_OFFSET, BACK_SCALE, BACK_OFFSET
	}
	Ctx( 0xA205, su );											// PA_SU_SC_MODE_CNTL
	const uint32_t br = ( (uint32_t)t.h << 16 ) | (uint32_t)t.w;
	const uint32_t ssc[2] = { 0, br }, wsc[2] = { 0x80000000u, br };
	SetRegs( 0x69, 0xA000, 0xA00C, ssc, 2 );
	Ctx( 0xA080, 0 );
	SetRegs( 0x69, 0xA000, 0xA081, wsc, 2 );
	SetRegs( 0x69, 0xA000, 0xA094, wsc, 2 );
	uint32_t gsc[2] = { 0x80000000u, br };
	if ( useGlState && s_enabled[CapSlot( GL_SCISSOR_TEST )] )
	{
		int x0 = s_scissor[0] * t.scale, y0 = s_scissor[1] * t.scale;
		int x1 = ( s_scissor[0] + s_scissor[2] ) * t.scale, y1 = ( s_scissor[1] + s_scissor[3] ) * t.scale;
		if ( t.flipY ) { int a0 = t.h - y1, a1 = t.h - y0; y0 = a0; y1 = a1; }
		if ( x0 < 0 ) x0 = 0; if ( y0 < 0 ) y0 = 0; if ( x1 > t.w ) x1 = t.w; if ( y1 > t.h ) y1 = t.h;
		if ( x1 < x0 ) x1 = x0; if ( y1 < y0 ) y1 = y0;
		gsc[0] = 0x80000000u | (uint32_t)x0 | ( (uint32_t)y0 << 16 );
		gsc[1] = (uint32_t)x1 | ( (uint32_t)y1 << 16 );
	}
	SetRegs( 0x69, 0xA000, 0xA090, gsc, 2 );					// PA_SC_GENERIC_SCISSOR
	Ctx( 0xA2D5, 0 );											// VGT_SHADER_STAGES_EN
}

// V# для атрибута по glVertexAttribPointer
static void VSharpAttrib( uint32_t *d, const Attrib &a )
{
	static float *s_zero;
	Buffer *b = a.enabled ? FindBuffer( a.buffer ) : NULL;
	if ( !b || !b->data )
	{
		if ( !s_zero ) { s_zero = (float *)GpuAlloc( 256, 256 ); s_zero[0] = s_zero[1] = s_zero[2] = 0; s_zero[3] = 1; }
		uint64_t za = (uint64_t)(uintptr_t)s_zero;
		d[0] = (uint32_t)za; d[1] = (uint32_t)( za >> 32 & 0xFFFF ); d[2] = 16;
		d[3] = SEL_RGBA | ( 7u << 12 ) | ( 14u << 15 );
		return;
	}
	b->usedFrame = s_frameNo;
	int size = a.size;
	bool bgra = size == 0x80E1;
	if ( bgra ) size = 4;
	uint32_t dfmt = 14, nfmt = 7, comp = 4;
	switch ( a.type )
	{
	case GL_FLOAT: { static const uint32_t f[5] = { 0, 4, 11, 13, 14 }; dfmt = f[size < 5 ? size : 4]; nfmt = 7; comp = 4; break; }
	case 0x140B: case 0x8D61: { static const uint32_t f[5] = { 0, 2, 5, 12, 12 }; dfmt = f[size < 5 ? size : 4]; nfmt = 7; comp = 2; break; }
	case GL_UNSIGNED_BYTE: { static const uint32_t f[5] = { 0, 1, 3, 10, 10 }; dfmt = f[size < 5 ? size : 4]; nfmt = a.norm ? 0 : 2; comp = 1; break; }
	case GL_BYTE: { static const uint32_t f[5] = { 0, 1, 3, 10, 10 }; dfmt = f[size < 5 ? size : 4]; nfmt = a.norm ? 1 : 3; comp = 1; break; }
	case GL_UNSIGNED_SHORT: { static const uint32_t f[5] = { 0, 2, 5, 12, 12 }; dfmt = f[size < 5 ? size : 4]; nfmt = a.norm ? 0 : 2; comp = 2; break; }
	case GL_SHORT: { static const uint32_t f[5] = { 0, 2, 5, 12, 12 }; dfmt = f[size < 5 ? size : 4]; nfmt = a.norm ? 1 : 3; comp = 2; break; }
	default: break;
	}
	uint32_t elem = comp * ( size == 3 && comp < 4 ? 4 : size );
	uint32_t stride = a.stride ? (uint32_t)a.stride : elem;
	uint64_t va = (uint64_t)(uintptr_t)b->data + a.offset;
	// число целых вершин: последняя помещается, если после смещения хватает места на один элемент.
	uint32_t recs = 0;
	if ( a.offset + elem <= b->size )
		recs = stride ? (uint32_t)( ( b->size - a.offset - elem ) / stride + 1 ) : (uint32_t)( b->size - a.offset );
	uint32_t sel;
	if ( bgra ) sel = SEL_BGRA;
	else if ( size >= 4 ) sel = SEL_RGBA;
	else if ( size == 3 ) sel = 4u | ( 5u << 3 ) | ( 6u << 6 ) | ( 1u << 9 );
	else if ( size == 2 ) sel = 4u | ( 5u << 3 ) | ( 0u << 6 ) | ( 1u << 9 );
	else sel = 4u | ( 0u << 3 ) | ( 0u << 6 ) | ( 1u << 9 );
	d[0] = (uint32_t)va;
	d[1] = (uint32_t)( va >> 32 & 0xFFFF ) | ( stride << 16 );
	d[2] = recs;
	d[3] = sel | ( nfmt << 12 ) | ( dfmt << 15 );
}

static void *s_fetchCode[17], *s_fetchCodeI[2][17];
static void *FetchCodeIdx( int n, bool idx32 )
{
	if ( n < 1 || n > 16 ) return NULL;
	void *&c = s_fetchCodeI[idx32][n];
	if ( !c )
	{
		const unsigned char *code = idx32 ? g_fetchI32[n].code : g_fetchI16[n].code;
		unsigned size = idx32 ? g_fetchI32[n].size : g_fetchI16[n].size;
		c = GpuAlloc( size + 256, 256 );
		if ( c ) memcpy( c, code, size );
	}
	return c;
}
static void *FetchCode( int n )
{
	if ( n < 1 || n > 16 ) return NULL;
	if ( !s_fetchCode[n] )
	{
		s_fetchCode[n] = GpuAlloc( g_fetch[n].size + 256, 256 );
		if ( s_fetchCode[n] ) memcpy( s_fetchCode[n], g_fetch[n].code, g_fetch[n].size );
	}
	return s_fetchCode[n];
}

// пользовательские SGPR и расширенная таблица по слотам .sb
static void FillSlots( const Shader *sh, uint32_t *ud, uint8_t *ext, void *cbMem, int cbSize, void *fetch, void *vtable,
                       const Program *p )
{
	for ( uint32_t s : sh->slots )
	{
		uint32_t type = s & 0xFF, api = ( s >> 8 ) & 0xFF, reg = ( s >> 16 ) & 0xFF;
		uint32_t v[8] = {};
		int n = 0;
		if ( type == 0x12 && fetch ) { uint64_t a = (uint64_t)(uintptr_t)fetch; v[0] = (uint32_t)a; v[1] = (uint32_t)( a >> 32 ); n = 2; }
		else if ( type == 0x17 && vtable ) { uint64_t a = (uint64_t)(uintptr_t)vtable; v[0] = (uint32_t)a; v[1] = (uint32_t)( a >> 32 ); n = 2; }
		else if ( type == 0x02 && cbMem )
		{
			uint64_t a = (uint64_t)(uintptr_t)cbMem;
			v[0] = (uint32_t)a; v[1] = (uint32_t)( a >> 32 & 0xFFFF ); v[2] = (uint32_t)cbSize;
			v[3] = SEL_RGBA | ( 7u << 12 ) | ( 14u << 15 ); n = 4;
		}
		else if ( type == 0x00 || type == 0x01 )
		{
			int unit = p && api < 16 ? p->samplerUnit[api] : (int)api;
			Texture *t = TexOf( s_bound2D[unit & 31] );
			if ( !t || !t->mem ) t = &s_whiteTex;
			if ( s_unitNot2D[unit & 31] ) t = &s_blackTex;
			if ( type == 0x00 ) { TSharp( v, t ); n = 8; }
			else
			{
				GLuint so = s_boundSampler[unit & 31];
				if ( so && s_samplers.count( so ) )
				{
					const SamplerObj &sm = s_samplers[so];
					SSharp( v, sm.wrapS, sm.wrapT, sm.minF, sm.magF, TexLastLevel( t ) );
				}
				else SSharp( v, t->wrapS, t->wrapT, t->minF, t->magF, TexLastLevel( t ) );
				n = 4;
			}
		}
		else if ( type == 0x1B && ext ) { uint64_t a = (uint64_t)(uintptr_t)ext; v[0] = (uint32_t)a; v[1] = (uint32_t)( a >> 32 ); n = 2; }
		for ( int i = 0; i < n; i++ )
		{
			uint32_t r = reg + i;
			if ( r < 16 ) ud[r] = v[i];
			else if ( ext ) memcpy( ext + ( r - 16 ) * 4, &v[i], 4 );
		}
	}
}

static uint32_t PrimType( GLenum mode )
{
	switch ( mode ) { case GL_POINTS: return 1; case GL_LINES: return 2; case GL_LINE_STRIP: return 3; case GL_TRIANGLE_STRIP: return 6; case GL_TRIANGLE_FAN: return 5; }
	return 4;
}

static void DrawCommon( GLenum mode, GLsizei count, bool indexed, GLenum itype, const void *indices, GLint baseVertex, GLint first )
{
	if ( count <= 0 || s_video < 0 ) return;
	Program *p = CurProgram();
	Shader *vs = p && p->linked && s_shaders.count( p->vs ) ? s_shaders[p->vs] : NULL;
	Shader *ps = p && p->linked && s_shaders.count( p->ps ) ? s_shaders[p->ps] : NULL;
	Target t;
	if ( !vs || !ps || !GetTarget( s_drawFbo, t ) || !CmdRoom( 600 ) ) return;
	GLuint ibName = s_boundBuffer[1];
	if ( indexed && itype == GL_UNSIGNED_BYTE )
	{
		// байтовых индексов загрузчик не читает: расширяем до 16 бит в данных кадра
		const uint8_t *src = (const uint8_t *)indices;
		if ( ibName )
		{
			Buffer *ib = FindBuffer( ibName );
			uintptr_t off = (uintptr_t)indices;
			src = ib && ib->data && off + (size_t)count <= ib->size ? (const uint8_t *)ib->data + off : NULL;
		}
		uint16_t *wide = src ? (uint16_t *)Upload( NULL, (size_t)count * 2, 16 ) : NULL;
		if ( !wide ) return;
		for ( GLsizei k = 0; k < count; k++ ) wide[k] = src[k];
		indices = wide; ibName = 0; itype = GL_UNSIGNED_SHORT;
	}
	// то, что нарисовали в прошлую цель, могло остаться в кэше цвета: если читать её
	// как текстуру сразу, получаются рваные полосы
	if ( s_fbChanged ) { FlushCaches(); s_fbChanged = false; }
	EmitTargetState( t, true );
	// связь VS -> PS и форматы вывода — из заголовков .sb
	Ctx( 0xA1B1, vs->hdr[6] );									// SPI_VS_OUT_CONFIG
	Ctx( 0xA1C3, 4 );											// SPI_SHADER_POS_FORMAT
	Ctx( 0xA1C5, ps->hdr[7] );									// SPI_SHADER_COL_FORMAT
	Ctx( 0xA1B3, ps->hdr[8] ); Ctx( 0xA1B4, ps->hdr[9] );		// SPI_PS_INPUT_ENA / ADDR
	Ctx( 0xA1B6, (uint32_t)p->nPsInputs );						// SPI_PS_IN_CONTROL
	Ctx( 0xA203, ps->hdr[12] );									// DB_SHADER_CONTROL
	for ( int k = 0; k < p->nPsInputs && k < 32; k++ ) Ctx( 0xA191 + k, (uint32_t)p->psInputParam[k] );
	// VS: загрузчик, таблица V#, константы
	uint32_t vsud[16] = {};
	void *vtab = NULL;
	// DRAW_INDEX_2 на PS4 вешает видеокарту, поэтому индексы читает сам загрузчик вершин:
	// рисуем DRAW_INDEX_AUTO, а номер вершины v0 служит адресом в буфере индексов
	bool emu = indexed && p->nAttribs > 0;
	bool idx32 = itype == GL_UNSIGNED_INT;
	uint32_t isz = idx32 ? 4 : 2;
	uint64_t emuIa = 0;
	uint32_t emuBytes = 0;
	const uint8_t *indexSrc = (const uint8_t *)indices;			// индексы в памяти процессора
	if ( indexed && ibName )
	{
		Buffer *ib = FindBuffer( ibName );
		uintptr_t off = (uintptr_t)indices;
		indexSrc = ib && ib->data && off + (size_t)count * isz <= ib->size ? (const uint8_t *)ib->data + off : NULL;
		if ( indexSrc ) ib->usedFrame = s_frameNo;
	}
	if ( indexed && !indexSrc ) return;
	if ( emu )
	{
		emuBytes = (uint32_t)count * isz;
		emuIa = ibName ? (uint64_t)(uintptr_t)indexSrc : (uint64_t)(uintptr_t)Upload( indexSrc, emuBytes, 16 );
		if ( !emuIa ) return;
	}
	int seqFirst = -1;			// без атрибутов: индексы должны идти подряд, тогда рисуем без них
	if ( indexed && !emu )
	{
		uint32_t v0 = idx32 ? ( (const uint32_t *)indexSrc )[0] : ( (const uint16_t *)indexSrc )[0];
		bool seq = true;
		for ( int k = 1; k < count && seq; k++ )
			seq = ( idx32 ? ( (const uint32_t *)indexSrc )[k] : ( (const uint16_t *)indexSrc )[k] ) == v0 + (uint32_t)k;
		if ( !seq )
		{
			static int s_said;
			if ( s_said++ < 5 ) ps4glLog( "ps4gl: indexed draw without vertex attributes and with non-sequential indices is not supported" );
			return;
		}
		seqFirst = (int)v0 + baseVertex;
	}
	if ( p->nAttribs > 0 )
	{
		uint32_t tab[17 * 4];
		for ( int k = 0; k < p->nAttribs; k++ ) VSharpAttrib( tab + 4 * k, s_attr[p->vsAttribLoc[k] & 15] );
		int n = p->nAttribs;
		if ( emu )
		{
			// базовая вершина — сдвигом начала V# атрибутов
			for ( int k = 0; k < n && baseVertex > 0; k++ )
			{
				uint32_t *d = tab + 4 * k;
				uint32_t stride = d[1] >> 16;
				if ( !stride ) continue;
				uint64_t base = ( d[0] | (uint64_t)( d[1] & 0xFFFF ) << 32 ) + (uint64_t)baseVertex * stride;
				d[0] = (uint32_t)base; d[1] = (uint32_t)( base >> 32 & 0xFFFF ) | ( stride << 16 );
				d[2] = d[2] > (uint32_t)baseVertex ? d[2] - (uint32_t)baseVertex : 0;
			}
			uint32_t *d = tab + 4 * n;									// V# буфера индексов
			d[0] = (uint32_t)emuIa; d[1] = (uint32_t)( emuIa >> 32 & 0xFFFF );
			d[2] = emuBytes; d[3] = SEL_RGBA | ( 4u << 12 ) | ( 4u << 15 );	// 32 бита, UINT (для ushort/dword без формата)
			n++;
		}
		vtab = Upload( tab, 16 * n, 16 );
	}
	void *vcb = Upload( p->vsCB.data(), p->vsCB.size() );
	FillSlots( vs, vsud, NULL, vcb, (int)p->vsCB.size(), emu ? FetchCodeIdx( p->nAttribs, idx32 ) : FetchCode( p->nAttribs ), vtab, p );
	uint64_t a = (uint64_t)(uintptr_t)vs->code;
	const uint32_t vsr[4] = { (uint32_t)( a >> 8 ), (uint32_t)( a >> 40 ), vs->hdr[4], vs->hdr[5] };
	SetRegs( 0x76, 0x2C00, 0x2C48, vsr, 4 );
	SetRegs( 0x76, 0x2C00, 0x2C4C, vsud, 16 );
	// PS: текстуры, сэмплеры, константы (часть — в расширенной таблице)
	uint32_t psud[16] = {};
	uint8_t *ext = (uint8_t *)Upload( NULL, 1024 );				// до 16 текстур: константы с 720 байт
	if ( ext ) memset( ext, 0, 1024 );
	void *pcb = Upload( p->psCB.data(), p->psCB.size() );
	FillSlots( ps, psud, ext, pcb, (int)p->psCB.size(), NULL, NULL, p );
	a = (uint64_t)(uintptr_t)ps->code;
	const uint32_t psr[4] = { (uint32_t)( a >> 8 ), (uint32_t)( a >> 40 ), ps->hdr[4], ps->hdr[5] };
	SetRegs( 0x76, 0x2C00, 0x2C08, psr, 4 );
	SetRegs( 0x76, 0x2C00, 0x2C0C, psud, 16 );
	// рисуем
	Uc( 0xC242, PrimType( mode ) );								// VGT_PRIMITIVE_TYPE
	Ctx( 0xA100, 0xFFFFFFFFu ); Ctx( 0xA101, 0 );				// VGT_MAX/MIN_VTX_INDX
	Ctx( 0xA102, emu ? 0u : seqFirst >= 0 ? (uint32_t)seqFirst : (uint32_t)first );	// VGT_INDX_OFFSET
	*s_cmd++ = Pkt3( 0x2F, 1 ); *s_cmd++ = 1;					// NUM_INSTANCES
	*s_cmd++ = Pkt3( 0x2D, 2 ); *s_cmd++ = (uint32_t)count; *s_cmd++ = 2;	// DRAW_INDEX_AUTO
	if ( s_drawFbo == 0 ) s_frameDrawsFb0++;
}

// ---- перенос (glBlitFramebuffer): прямоугольник с текстурой ----
static void *s_blitVs, *s_blitPs, *s_clearPs;
static void BlitTexture( const Texture *src, const Target &t, GLint sx0, GLint sy0, GLint sx1, GLint sy1, GLint dx0, GLint dy0, GLint dx1, GLint dy1, GLenum filter )
{
	FlushCaches();						// источник только что был целью рендера
	EmitTargetState( t, false );
	// вершины: позиция (в координатах цели -> NDC) и u, v источника
	auto nx = [&]( float x ) { return x / t.w * 2.0f - 1.0f; };
	auto ny = [&]( float y ) { return y / t.h * 2.0f - 1.0f; };
	float u0 = (float)sx0 / src->w, u1 = (float)sx1 / src->w, v0 = (float)sy0 / src->h, v1 = (float)sy1 / src->h;
	const float q[6][8] = {
		{ nx( dx0 ), ny( dy0 ), 0, 1, u0, v0, 0, 0 }, { nx( dx1 ), ny( dy0 ), 0, 1, u1, v0, 0, 0 }, { nx( dx0 ), ny( dy1 ), 0, 1, u0, v1, 0, 0 },
		{ nx( dx1 ), ny( dy0 ), 0, 1, u1, v0, 0, 0 }, { nx( dx1 ), ny( dy1 ), 0, 1, u1, v1, 0, 0 }, { nx( dx0 ), ny( dy1 ), 0, 1, u0, v1, 0, 0 },
	};
	void *vb = Upload( q, sizeof( q ), 256 );
	if ( !vb ) return;
	Ctx( 0xA1B1, 0 ); Ctx( 0xA1C3, 4 ); Ctx( 0xA1C5, 9 );		// 1 параметр, позиция, цвет 32_ABGR
	Ctx( 0xA191, 0 ); Ctx( 0xA1B3, 2 ); Ctx( 0xA1B4, 2 ); Ctx( 0xA1B6, 1 ); Ctx( 0xA203, 0x10 );
	uint64_t a = (uint64_t)(uintptr_t)s_blitVs, va = (uint64_t)(uintptr_t)vb;
	const uint32_t vsr[4] = { (uint32_t)( a >> 8 ), (uint32_t)( a >> 40 ), ( 2u << 0 ) | ( 1u << 6 ), 4u << 1 };
	SetRegs( 0x76, 0x2C00, 0x2C48, vsr, 4 );
	const uint32_t vsh[4] = { (uint32_t)va, (uint32_t)( va >> 32 & 0xFFFF ) | ( 32u << 16 ), 6, SEL_RGBA | ( 7u << 12 ) | ( 14u << 15 ) };
	SetRegs( 0x76, 0x2C00, 0x2C4C, vsh, 4 );
	a = (uint64_t)(uintptr_t)s_blitPs;
	const uint32_t psr[4] = { (uint32_t)( a >> 8 ), (uint32_t)( a >> 40 ), ( 1u << 0 ) | ( 2u << 6 ), 12u << 1 };
	SetRegs( 0x76, 0x2C00, 0x2C08, psr, 4 );
	uint32_t ud[12];
	TSharp( ud, src );
	SSharp( ud + 8, 0x812F, 0x812F, filter, filter );
	SetRegs( 0x76, 0x2C00, 0x2C0C, ud, 12 );
	Uc( 0xC242, 4 );
	Ctx( 0xA100, 0xFFFFFFFFu ); Ctx( 0xA101, 0 ); Ctx( 0xA102, 0 );
	*s_cmd++ = Pkt3( 0x2F, 1 ); *s_cmd++ = 1;
	*s_cmd++ = Pkt3( 0x2D, 2 ); *s_cmd++ = 6; *s_cmd++ = 2;
	FlushCaches();
}

static void DoBlit( GLint sx0, GLint sy0, GLint sx1, GLint sy1, GLint dx0, GLint dy0, GLint dx1, GLint dy1, GLenum filter )
{
	auto it = s_fbos.find( s_readFbo );
	if ( it == s_fbos.end() ) return;
	Texture *src = it->second.colorRb ? ( s_rbs.count( it->second.color ) ? s_rbs[it->second.color] : NULL ) : TexOf( it->second.color );
	Target t;
	if ( !src || !src->mem || !GetTarget( s_drawFbo, t ) || !CmdRoom( 600 ) ) return;
	BlitTexture( src, t, sx0, sy0, sx1, sy1, dx0 * t.scale, dy0 * t.scale, dx1 * t.scale, dy1 * t.scale, filter );
	if ( s_drawFbo == 0 ) s_frameDrawsFb0++;
}

// Сжать большой буфер в настоящий кадр экрана. Строка 0 буфера — верх картинки.
// Место в буфере команд не проверяем: рисования всегда оставляют запас в 1024 слова.
static void ResolveSupersampling()
{
	Texture src;
	src.w = src.pitch = SCREEN_W * s_ssScale;
	src.h = src.rows = SCREEN_H * s_ssScale;
	src.sel = SEL_RGBA;
	src.mem = s_ssColor;
	Target t;
	t.color = s_fb[s_fbIdx];
	t.pitch = t.w = SCREEN_W; t.h = t.rows = SCREEN_H;
	t.swap = 1; t.flipY = true;
	memset( s_ctxValid, 0, sizeof( s_ctxValid ) );		// цель сменила размер (как при очистке)
	BlitTexture( &src, t, 0, 0, src.w, src.h, 0, SCREEN_H, SCREEN_W, 0, GL_LINEAR );
}

// Очистка цели. Быстрый путь — прямоугольник на всю цель самой видеокартой
// (заливка DMA в 1080p стоит порядка 20 мс на кадр). Цвет — шейдер g_clearPs из s0..s3,
// глубина — z в вершинах через обычное окно вывода.
static void DoClear( GLbitfield mask )
{
	Target t;
	if ( !GetTarget( s_drawFbo, t ) || !CmdRoom( 4096 ) ) return;
	bool doColor = ( mask & GL_COLOR_BUFFER_BIT ) != 0, doDepth = ( mask & GL_DEPTH_BUFFER_BIT ) && t.depth;
	// буфер глубины другого размера, чем цель: прямоугольник накроет его не целиком — заливаем DMA
	if ( doDepth && ( t.dpitch > t.pitch + 64 || t.drows > t.rows + 64 || t.dpitch < t.w || t.drows < t.h ) )
	{
		FlushCaches();
		DmaFill( t.depth, F2U( s_clearDepth ), (size_t)t.dpitch * t.drows * 4 );
		doDepth = false;
		if ( !doColor ) return;
	}
	if ( !doColor && !doDepth ) return;
	// Тень регистров здесь надо забыть: с ней очистка ложилась только в часть цели
	memset( s_ctxValid, 0, sizeof( s_ctxValid ) );
	EmitTargetState( t, false );
	uint32_t cm = doColor ? ( ( s_colorMask[0] ? 1 : 0 ) | ( s_colorMask[1] ? 2 : 0 ) | ( s_colorMask[2] ? 4 : 0 ) | ( s_colorMask[3] ? 8 : 0 ) ) : 0;
	Ctx( 0xA08E, cm );												// CB_TARGET_MASK
	Ctx( 0xA1E0, 0 );												// без смешивания
	Ctx( 0xA200, doDepth ? ( ( 1u << 1 ) | ( 1u << 2 ) | ( 7u << 4 ) ) : 0 );	// глубина: запись, ALWAYS
	// z_окна = z * (f-n)/2 + (f+n)/2 = clearDepth
	float zr = s_depthFar - s_depthNear;
	float zn = zr != 0.0f ? ( 2.0f * s_clearDepth - ( s_depthFar + s_depthNear ) ) / zr : 1.0f;
	if ( zn > 1.0f ) zn = 1.0f;
	if ( zn < -1.0f ) zn = -1.0f;
	const float q[6][8] = {
		{ -1, -1, zn, 1, 0, 0, 0, 0 }, { 1, -1, zn, 1, 0, 0, 0, 0 }, { -1, 1, zn, 1, 0, 0, 0, 0 },
		{ 1, -1, zn, 1, 0, 0, 0, 0 }, { 1, 1, zn, 1, 0, 0, 0, 0 }, { -1, 1, zn, 1, 0, 0, 0, 0 },
	};
	void *vb = Upload( q, sizeof( q ), 256 );
	if ( !vb ) return;
	Ctx( 0xA1B1, 0 ); Ctx( 0xA1C3, 4 ); Ctx( 0xA1C5, 4 );			// 1 параметр, позиция, цвет FP16
	Ctx( 0xA191, 0 ); Ctx( 0xA1B3, 2 ); Ctx( 0xA1B4, 2 ); Ctx( 0xA1B6, 1 ); Ctx( 0xA203, 0x10 );
	uint64_t a = (uint64_t)(uintptr_t)s_blitVs, va = (uint64_t)(uintptr_t)vb;
	const uint32_t vsr[4] = { (uint32_t)( a >> 8 ), (uint32_t)( a >> 40 ), ( 2u << 0 ) | ( 1u << 6 ), 4u << 1 };
	SetRegs( 0x76, 0x2C00, 0x2C48, vsr, 4 );
	const uint32_t vsh[4] = { (uint32_t)va, (uint32_t)( va >> 32 & 0xFFFF ) | ( 32u << 16 ), 6, SEL_RGBA | ( 7u << 12 ) | ( 14u << 15 ) };
	SetRegs( 0x76, 0x2C00, 0x2C4C, vsh, 4 );
	a = (uint64_t)(uintptr_t)s_clearPs;
	const uint32_t psr[4] = { (uint32_t)( a >> 8 ), (uint32_t)( a >> 40 ), 0u, 4u << 1 };	// 4 VGPR, 8 SGPR; USER_SGPR = 4
	SetRegs( 0x76, 0x2C00, 0x2C08, psr, 4 );
	const uint32_t col[4] = { F2U( s_clearColor[0] ), F2U( s_clearColor[1] ), F2U( s_clearColor[2] ), F2U( s_clearColor[3] ) };
	SetRegs( 0x76, 0x2C00, 0x2C0C, col, 4 );
	Uc( 0xC242, 4 );
	Ctx( 0xA100, 0xFFFFFFFFu ); Ctx( 0xA101, 0 ); Ctx( 0xA102, 0 );
	*s_cmd++ = Pkt3( 0x2F, 1 ); *s_cmd++ = 1;
	*s_cmd++ = Pkt3( 0x2D, 2 ); *s_cmd++ = 6; *s_cmd++ = 2;
	if ( doColor && s_drawFbo == 0 ) s_clearedBackThisFrame = true;
}

static void *GpuCopy( const void *code, size_t bytes )
{
	void *p = GpuAlloc( bytes + 256, 256 );
	if ( p ) memcpy( p, code, bytes );
	return p;
}

static void DrawInit()
{
	for ( int i = 0; i < 2; i++ ) s_upload[i] = (uint8_t *)GpuAlloc( UPLOAD_BYTES, 0x10000 );
	s_blitVs = GpuCopy( g_blitVs, sizeof( g_blitVs ) );
	s_blitPs = GpuCopy( g_blitPs, sizeof( g_blitPs ) );
	s_clearPs = GpuCopy( g_clearPs, sizeof( g_clearPs ) );
	s_whiteTex.w = s_whiteTex.h = 1; s_whiteTex.pitch = 64; s_whiteTex.rows = 8; s_whiteTex.bpe = 4;
	s_whiteTex.sel = SEL_RGBA;			// без этого все каналы = 0 (прозрачно-чёрная «белая» заглушка)
	s_blackTex = s_whiteTex;
	s_blackTex.mem = GpuAlloc( 64 * 8 * 4, 0x10000 );
	if ( s_blackTex.mem ) { uint32_t *b = (uint32_t *)s_blackTex.mem; for ( int i = 0; i < 64 * 8; i++ ) b[i] = 0xFF000000u; }
	s_whiteTex.mem = GpuAlloc( 64 * 8 * 4, 0x10000 );
	if ( s_whiteTex.mem ) memset( s_whiteTex.mem, 0xFF, 64 * 8 * 4 );
}
static void DrawFrameReset() { s_uploadOff = 0; }

// =====================================================================
// Функции GL
// =====================================================================
extern "C" {

void glActiveTexture( GLenum u ) { s_activeUnit = ( u - GL_TEXTURE0 ) & 31; }
void glAttachShader( GLuint prog, GLuint sh )
{
		if ( !s_programs.count( prog ) || !s_shaders.count( sh ) ) return;
	( s_shaders[sh]->type == GL_VERTEX_SHADER ? s_programs[prog]->vs : s_programs[prog]->ps ) = sh;
}
void glBeginQuery( GLenum, GLuint ) { }
void glBindAttribLocation( GLuint prog, GLuint index, const GLchar *name )
{
		if ( s_programs.count( prog ) && name ) s_programs[prog]->attribLoc[name] = index;
}
void glBindBuffer( GLenum target, GLuint b ) { s_boundBuffer[BufferSlot( target )] = b; }
void glBindFramebuffer( GLenum target, GLuint fb )
{
		if ( ( target == GL_FRAMEBUFFER || target == 0x8CA9 ) && s_drawFbo != fb ) s_fbChanged = true;
	if ( target == GL_FRAMEBUFFER || target == 0x8CA9 ) s_drawFbo = fb;		// DRAW
	if ( target == GL_FRAMEBUFFER || target == 0x8CA8 ) s_readFbo = fb;		// READ
}
void glBindRenderbuffer( GLenum, GLuint rb ) { s_boundRb = rb; }
void glBindSampler( GLuint unit, GLuint smp ) { s_boundSampler[unit & 31] = smp; }
void glBindTexture( GLenum target, GLuint tex )
{
		if ( target != GL_TEXTURE_2D ) { if ( tex ) s_unitNot2D[s_activeUnit & 31] = true; return; }	// кубы и 3D — позже
	s_bound2D[s_activeUnit & 31] = tex;
	if ( tex ) s_unitNot2D[s_activeUnit & 31] = false;
	if ( tex && !s_textures.count( tex ) ) s_textures[tex] = new Texture();
}
void glBindVertexArray( GLuint ) { }
void glBlendColor( GLfloat r, GLfloat g, GLfloat b, GLfloat a ) { s_blendColor[0] = r; s_blendColor[1] = g; s_blendColor[2] = b; s_blendColor[3] = a; }
void glBlendEquation( GLenum e ) { s_blendEq = s_blendEqA = e; }
void glBlendFunc( GLenum src, GLenum dst ) { s_blendSrc = s_blendSrcA = src; s_blendDst = s_blendDstA = dst; }
void glBlitFramebuffer( GLint sx0, GLint sy0, GLint sx1, GLint sy1, GLint dx0, GLint dy0, GLint dx1, GLint dy1, GLbitfield mask, GLenum filter )
{
		if ( !( mask & GL_COLOR_BUFFER_BIT ) ) return;
	DoBlit( sx0, sy0, sx1, sy1, dx0, dy0, dx1, dy1, filter );
	if ( s_drawFbo == 0 ) s_frameDrawsFb0++;
}
void glBufferData( GLenum target, GLsizeiptr size, const void *data, GLenum )
{
		Buffer *b = FindBuffer( s_boundBuffer[BufferSlot( target )] );
	if ( !b ) return;
	// всегда новая память: так делает и настоящий драйвер («осиротить» старую)
	BufferRealloc( b, (size_t)size, false );
	if ( data && b->data ) memcpy( b->data, data, size );
}
void glBufferSubData( GLenum target, GLintptr off, GLsizeiptr size, const void *data )
{
	Buffer *b = FindBuffer( s_boundBuffer[BufferSlot( target )] );
	if ( !b || !b->data || !data || (size_t)( off + size ) > b->size ) return;
	BufferBeforeWrite( b );
	if ( b->data ) memcpy( (char *)b->data + off, data, size );
}
GLenum glCheckFramebufferStatus( GLenum ) { return GL_FRAMEBUFFER_COMPLETE; }
void glClear( GLbitfield mask )
{
		DoClear( mask );
}
void glClearColor( GLfloat r, GLfloat g, GLfloat b, GLfloat a ) { s_clearColor[0] = r; s_clearColor[1] = g; s_clearColor[2] = b; s_clearColor[3] = a; }
void glClearDepthf( GLfloat d ) { s_clearDepth = d; }
void glClearStencil( GLint ) {}
GLenum glClientWaitSync( GLsync, GLbitfield, GLuint64 ) { return GL_ALREADY_SIGNALED_; }
void glColorMask( GLboolean r, GLboolean g, GLboolean b, GLboolean a ) { s_colorMask[0] = r; s_colorMask[1] = g; s_colorMask[2] = b; s_colorMask[3] = a; }
void glCompileShader( GLuint ) { }
// ---- сжатые текстуры (DXT1/3/5) распаковываем при загрузке в RGBA8:
// в линейной раскладке видеокарта PS4 на BC-форматах зависала (кадр 3, опыт notex)
static inline void Rgb565( uint16_t c, uint8_t *o )
{
	uint32_t r = ( c >> 11 ) & 31, g = ( c >> 5 ) & 63, b = c & 31;
	o[0] = (uint8_t)( ( r << 3 ) | ( r >> 2 ) ); o[1] = (uint8_t)( ( g << 2 ) | ( g >> 4 ) ); o[2] = (uint8_t)( ( b << 3 ) | ( b >> 2 ) );
}
// цвет блока BC1/BC2/BC3: 8 байт -> 16 пикселей RGBA (альфа ставится 255 или 0 у «дырки» BC1)
static void DecodeColorBlock( const uint8_t *b, uint8_t out[16][4], bool bc1 )
{
	uint16_t c0 = (uint16_t)( b[0] | ( b[1] << 8 ) ), c1 = (uint16_t)( b[2] | ( b[3] << 8 ) );
	uint8_t pal[4][4];
	Rgb565( c0, pal[0] ); Rgb565( c1, pal[1] );
	pal[0][3] = pal[1][3] = 255;
	if ( !bc1 || c0 > c1 )
	{
		for ( int k = 0; k < 3; k++ )
		{
			pal[2][k] = (uint8_t)( ( 2 * pal[0][k] + pal[1][k] ) / 3 );
			pal[3][k] = (uint8_t)( ( pal[0][k] + 2 * pal[1][k] ) / 3 );
		}
		pal[2][3] = pal[3][3] = 255;
	}
	else
	{
		for ( int k = 0; k < 3; k++ ) { pal[2][k] = (uint8_t)( ( pal[0][k] + pal[1][k] ) / 2 ); pal[3][k] = 0; }
		pal[2][3] = 255; pal[3][3] = 0;
	}
	uint32_t bits = (uint32_t)b[4] | ( (uint32_t)b[5] << 8 ) | ( (uint32_t)b[6] << 16 ) | ( (uint32_t)b[7] << 24 );
	for ( int i = 0; i < 16; i++ ) memcpy( out[i], pal[( bits >> ( 2 * i ) ) & 3], 4 );
}
static void DecodeBlock( int kind, const uint8_t *b, uint8_t out[16][4] )
{
	if ( kind == 1 ) { DecodeColorBlock( b, out, true ); return; }
	DecodeColorBlock( b + 8, out, false );
	if ( kind == 2 )
	{
		for ( int i = 0; i < 16; i++ ) { uint32_t a = ( b[i / 2] >> ( 4 * ( i & 1 ) ) ) & 15; out[i][3] = (uint8_t)( a * 17 ); }
		return;
	}
	uint8_t a[8];
	a[0] = b[0]; a[1] = b[1];
	if ( a[0] > a[1] ) for ( int k = 1; k < 7; k++ ) a[k + 1] = (uint8_t)( ( ( 7 - k ) * a[0] + k * a[1] ) / 7 );
	else { for ( int k = 1; k < 5; k++ ) a[k + 1] = (uint8_t)( ( ( 5 - k ) * a[0] + k * a[1] ) / 5 ); a[6] = 0; a[7] = 255; }
	uint64_t bits = 0;
	for ( int k = 0; k < 6; k++ ) bits |= (uint64_t)b[2 + k] << ( 8 * k );
	for ( int i = 0; i < 16; i++ ) out[i][3] = a[( bits >> ( 3 * i ) ) & 7];
}
static int CompressedKind( GLenum ifmt )
{
	switch ( ifmt )
	{
	case 0x83F0: case 0x83F1: case 0x8C4C: case 0x8C4D: return 1;		// DXT1
	case 0x83F2: case 0x8C4E: return 2;									// DXT3
	}
	return 3;															// DXT5
}
// распаковать прямоугольник блоков в текстуру RGBA8 (x0, y0 — в пикселях, кратны 4)
static void DecodeInto( Texture *t, int kind, int x0, int y0, int w, int h, const uint8_t *src, size_t size, int level = 0 )
{
	if ( level < 0 || level >= t->levels ) return;
	uint8_t *dstBase = (uint8_t *)t->mem + t->mipOff[level];
	int dPitch = t->mipPitch[level], dW = t->mipW[level], dH = t->mipH[level];
	int bw = ( w + 3 ) / 4, bh = ( h + 3 ) / 4, bb = kind == 1 ? 8 : 16;
	std::vector<uint8_t> rows( (size_t)bw * 4 * 4 * 4 );
	for ( int by = 0; by < bh; by++ )
	{
		if ( (size_t)( by + 1 ) * bw * bb > size ) break;
		for ( int bx = 0; bx < bw; bx++ )
		{
			uint8_t px[16][4];
			DecodeBlock( kind, src + ( (size_t)by * bw + bx ) * bb, px );
			for ( int yy = 0; yy < 4; yy++ ) memcpy( &rows[( (size_t)yy * bw * 4 + bx * 4 ) * 4], px[yy * 4], 16 );
		}
		for ( int yy = 0; yy < 4; yy++ )
		{
			int y = y0 + by * 4 + yy;
			if ( y >= dH ) break;
			int cw = w;
			if ( x0 + cw > dW ) cw = dW - x0;
			if ( cw <= 0 ) break;
			memcpy( dstBase + ( (size_t)y * dPitch + x0 ) * 4, &rows[(size_t)yy * bw * 4 * 4], (size_t)cw * 4 );
		}
	}
}
void glCompressedTexImage2D( GLenum target, GLint level, GLenum ifmt, GLsizei w, GLsizei h, GLint, GLsizei size, const void *data )
{
		Texture *t = BoundTex( target );
	if ( t && level > 0 )
	{
		if ( t->mem && level < t->levels && w == t->mipW[level] && h == t->mipH[level] && t->bpe == 4 )
		{
			const uint8_t *src = UnpackSource( data );
			if ( src ) { DecodeInto( t, CompressedKind( ifmt ), 0, 0, w, h, src, (size_t)size, level ); t->filled |= 1u << level; }
		}
		return;
	}
	if ( !t || level != 0 ) return;
	TexChooseFormat( t, ( ifmt >= 0x8C4C && ifmt <= 0x8C4F ) ? 0x8C43 : GL_RGBA, GL_RGBA, GL_UNSIGNED_BYTE );
	if ( !TexAlloc( t, w, h ) ) return;
	const uint8_t *src = UnpackSource( data );
	if ( !src ) return;
	DecodeInto( t, CompressedKind( ifmt ), 0, 0, w, h, src, (size_t)size );
}
void glCompressedTexImage3D( GLenum, GLint, GLenum, GLsizei, GLsizei, GLsizei, GLint, GLsizei, const void * ) { }
void glCopyBufferSubData( GLenum rt, GLenum wt, GLintptr ro, GLintptr wo, GLsizeiptr size )
{
	Buffer *r = FindBuffer( s_boundBuffer[BufferSlot( rt )] ), *w = FindBuffer( s_boundBuffer[BufferSlot( wt )] );
	if ( r && w && r->data && w->data && (size_t)( ro + size ) <= r->size && (size_t)( wo + size ) <= w->size )
	{
		BufferBeforeWrite( w );
		if ( w->data ) memmove( (char *)w->data + wo, (char *)r->data + ro, size );
	}
}
GLuint glCreateProgram( void )
{
		GLuint id = NewName();
	s_programs[id] = new Program();
	return id;
}
GLuint glCreateShader( GLenum type )
{
		GLuint id = NewName();
	Shader *sh = new Shader();
	sh->type = type;
	s_shaders[id] = sh;
	return id;
}
void glDeleteBuffers( GLsizei n, const GLuint *ids )
{
	for ( GLsizei i = 0; i < n; i++ )
	{
		Buffer *b = FindBuffer( ids[i] );
		if ( b && b->data ) { GpuFreeLater( b->data, b->size ); b->data = NULL; b->size = 0; }
	}
}
void glDeleteFramebuffers( GLsizei n, const GLuint *ids ) { for ( GLsizei i = 0; i < n; i++ ) s_fbos.erase( ids[i] ); }
void glDeleteProgram( GLuint id )
{
	auto it = s_programs.find( id );
	if ( it == s_programs.end() ) return;
	delete it->second;
	s_programs.erase( it );
	if ( s_curProgram == id ) s_curProgram = 0;
}
void glDeleteQueries( GLsizei, const GLuint * ) {}
void glDeleteRenderbuffers( GLsizei n, const GLuint *ids )
{
	for ( GLsizei i = 0; i < n; i++ )
	{
		auto it = s_rbs.find( ids[i] );
		if ( it == s_rbs.end() ) continue;
		if ( it->second->mem ) GpuFreeLater( it->second->mem, it->second->bytes );
		delete it->second;
		s_rbs.erase( it );
	}
}
void glDeleteSamplers( GLsizei, const GLuint * ) {}
void glDeleteShader( GLuint id )
{
	// не удаляем, пока на шейдер ссылается хоть одна программа
	for ( auto &pp : s_programs ) if ( pp.second->vs == id || pp.second->ps == id ) return;
	auto it = s_shaders.find( id );
	if ( it == s_shaders.end() ) return;
	ShaderFree( it->second );
	s_shaders.erase( it );
}
void glDeleteSync( GLsync ) {}
void glDeleteTextures( GLsizei n, const GLuint *ids )
{
	for ( GLsizei i = 0; i < n; i++ )
	{
		auto it = s_textures.find( ids[i] );
		if ( it == s_textures.end() ) continue;
		if ( it->second->mem ) GpuFreeLater( it->second->mem, it->second->bytes );
		delete it->second;
		s_textures.erase( it );
		for ( int u = 0; u < 32; u++ ) if ( s_bound2D[u] == ids[i] ) s_bound2D[u] = 0;
	}
}
void glDeleteVertexArrays( GLsizei, const GLuint * ) {}
void glDepthFunc( GLenum f ) { s_depthFunc = f; }
void glDepthMask( GLboolean m ) { s_depthMask = m; }
void glDepthRangef( GLfloat n, GLfloat f ) { s_depthNear = n; s_depthFar = f; }
void glDetachShader( GLuint, GLuint ) {}		// связь нужна программе и после glLinkProgram
void glDisable( GLenum cap ) { s_enabled[CapSlot( cap )] = GL_FALSE; }
void glDisableVertexAttribArray( GLuint i ) { if ( i < 16 ) s_attr[i].enabled = false; }
void glDrawArrays( GLenum mode, GLint first, GLsizei count )
{
		DrawCommon( mode, count, false, 0, NULL, 0, first );
	if ( s_drawFbo == 0 ) s_frameDrawsFb0++;
}
void glDrawBuffers( GLsizei, const GLenum * ) { }
void glDrawRangeElements( GLenum mode, GLuint, GLuint, GLsizei count, GLenum type, const void *idx )
{
		DrawCommon( mode, count, true, type, idx, 0, 0 );
	if ( s_drawFbo == 0 ) s_frameDrawsFb0++;
}
void glEnable( GLenum cap ) { s_enabled[CapSlot( cap )] = GL_TRUE; }
void glEnableVertexAttribArray( GLuint i ) { if ( i < 16 ) s_attr[i].enabled = true; }
void glEndQuery( GLenum ) {}
GLsync glFenceSync( GLenum, GLbitfield ) { return (GLsync)(uintptr_t)1; }
void glFinish( void ) {}
void glFlush( void ) {}
void glFlushMappedBufferRange( GLenum, GLintptr, GLsizeiptr ) {}
void glFramebufferRenderbuffer( GLenum target, GLenum att, GLenum, GLuint rb )
{
		GLuint fb = target == 0x8CA8 ? s_readFbo : s_drawFbo;
	if ( !fb ) return;
	Fbo &f = s_fbos[fb];
	if ( att == GL_COLOR_ATTACHMENT0 ) { f.color = rb; f.colorRb = true; }
	else if ( att == GL_DEPTH_ATTACHMENT || att == 0x821A ) { f.depth = rb; f.depthRb = true; }	// DEPTH_STENCIL
}
void glFramebufferTexture2D( GLenum target, GLenum att, GLenum, GLuint tex, GLint )
{
		GLuint fb = target == 0x8CA8 ? s_readFbo : s_drawFbo;
	if ( !fb ) return;
	Fbo &f = s_fbos[fb];
	if ( att == GL_COLOR_ATTACHMENT0 ) { f.color = tex; f.colorRb = false; }
	else if ( att == GL_DEPTH_ATTACHMENT || att == 0x821A ) { f.depth = tex; f.depthRb = false; }
}
void glFrontFace( GLenum m ) { s_frontFace = m; }
static void GenNames( GLsizei n, GLuint *ids ) { for ( GLsizei i = 0; i < n; i++ ) ids[i] = NewName(); }
static GLuint s_nextBuffer = 1;	// свой счётчик: номер буфера = место в s_buffers
void glGenBuffers( GLsizei n, GLuint *ids )
{
		for ( GLsizei i = 0; i < n; i++ )
	{
		if ( s_nextBuffer >= 65536 ) { ps4glLog( "ps4gl: more than 65535 buffers" ); ids[i] = 0; continue; }
		ids[i] = s_nextBuffer++;
	}
}
void glGenFramebuffers( GLsizei n, GLuint *ids ) { GenNames( n, ids ); }
void glGenQueries( GLsizei n, GLuint *ids ) { GenNames( n, ids ); }
void glGenRenderbuffers( GLsizei n, GLuint *ids ) { GenNames( n, ids ); }
void glGenSamplers( GLsizei n, GLuint *ids ) { GenNames( n, ids ); }
void glGenTextures( GLsizei n, GLuint *ids ) { GenNames( n, ids ); }
void glGenVertexArrays( GLsizei n, GLuint *ids ) { GenNames( n, ids ); }
void glGetIntegerv( GLenum pname, GLint *v );
void glGetBooleanv( GLenum pname, GLboolean *v ) { GLint i = 0; glGetIntegerv( pname, &i ); if ( v ) *v = i ? GL_TRUE : GL_FALSE; }
GLenum glGetError( void ) { return GL_NO_ERROR; }
void glGetFloatv( GLenum pname, GLfloat *v ) { GLint i = 0; glGetIntegerv( pname, &i ); if ( v ) *v = (GLfloat)i; }
void glGetFramebufferAttachmentParameteriv( GLenum, GLenum, GLenum, GLint *v ) { if ( v ) *v = 0; }
void glGetIntegerv( GLenum pname, GLint *v )
{
	if ( !v ) return;
	switch ( pname )
	{
	case GL_MAX_TEXTURE_SIZE: case GL_MAX_RENDERBUFFER_SIZE: case GL_MAX_CUBE_MAP_TEXTURE_SIZE: *v = 8192; return;
	case GL_MAX_3D_TEXTURE_SIZE_: *v = 2048; return;
	case GL_MAX_ARRAY_TEXTURE_LAYERS_: *v = 256; return;
	case GL_MAX_VERTEX_ATTRIBS: *v = 16; return;
	case GL_MAX_TEXTURE_IMAGE_UNITS: case GL_MAX_VERTEX_TEXTURE_IMAGE_UNITS: *v = 16; return;
	case GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS: *v = 32; return;
	case GL_MAX_VERTEX_UNIFORM_VECTORS: *v = 256; return;
	case GL_MAX_FRAGMENT_UNIFORM_VECTORS: *v = 224; return;
	case GL_MAX_VARYING_VECTORS: *v = 16; return;
	case GL_MAX_SAMPLES_: *v = 1; return;
	case GL_MAX_DRAW_BUFFERS_: case GL_MAX_COLOR_ATTACHMENTS_: *v = 4; return;
	case GL_MAJOR_VERSION_: *v = 3; return;
	case GL_MINOR_VERSION_: *v = 0; return;
	case GL_FRAMEBUFFER_BINDING_: *v = (GLint)s_drawFbo; return;
	case GL_VIEWPORT: memcpy( v, s_viewport, sizeof( s_viewport ) ); return;
	case GL_NUM_EXTENSIONS_: *v = 0; return;
	default: *v = 0; return;
	}
}
void glGetProgramInfoLog( GLuint, GLsizei, GLsizei *len, GLchar *log ) { if ( len ) *len = 0; if ( log ) log[0] = 0; }
void glGetProgramiv( GLuint id, GLenum pname, GLint *v )
{
	if ( !v ) return;
	*v = 0;
	if ( pname == GL_LINK_STATUS || pname == GL_VALIDATE_STATUS )
		*v = s_programs.count( id ) && s_programs[id]->linked ? 1 : 0;
}
void glGetQueryObjectuiv( GLuint, GLenum, GLuint *v ) { if ( v ) *v = 1; }
void glGetShaderInfoLog( GLuint, GLsizei, GLsizei *len, GLchar *log ) { if ( len ) *len = 0; if ( log ) log[0] = 0; }
void glGetShaderiv( GLuint id, GLenum pname, GLint *v )
{
	if ( !v ) return;
	*v = 0;
	if ( pname == GL_COMPILE_STATUS ) *v = s_shaders.count( id ) && s_shaders[id]->valid ? 1 : 0;
	else if ( pname == GL_SHADER_TYPE ) *v = s_shaders.count( id ) ? (GLint)s_shaders[id]->type : 0;
}
const GLubyte *glGetString( GLenum name )
{
	switch ( name )
	{
	case GL_VENDOR: return (const GLubyte *)"PS4GL_GNM";
	case GL_RENDERER: return (const GLubyte *)"PS4GL_GNM";
	case GL_VERSION: return (const GLubyte *)"OpenGL ES 3.0 PS4GL_GNM";
	case GL_SHADING_LANGUAGE_VERSION: return (const GLubyte *)"OpenGL ES GLSL ES 3.00";
	case GL_EXTENSIONS:
		return (const GLubyte *)"GL_OES_texture_npot GL_EXT_texture_filter_anisotropic GL_EXT_texture_rg GL_OES_rgb8_rgba8 "
		       "GL_OES_depth_texture GL_OES_depth32 GL_OES_fbo_render_mipmap GL_EXT_texture_format_BGRA8888 "
		       "GL_EXT_read_format_bgra GL_OES_texture_float GL_OES_texture_half_float GL_EXT_color_buffer_half_float "
		       "GL_OES_texture_float_linear GL_OES_texture_half_float_linear GL_OES_vertex_half_float GL_OES_element_index_uint "
		       "GL_EXT_texture_compression_dxt1 GL_EXT_texture_compression_dxt3 GL_EXT_texture_compression_dxt5 "
		       "GL_EXT_texture_compression_s3tc GL_EXT_texture_storage GL_OES_standard_derivatives GL_EXT_shader_texture_lod "
		       "GL_OES_texture_stencil8 GL_EXT_color_buffer_float GL_EXT_draw_instanced GL_EXT_instanced_arrays ";
	default: return (const GLubyte *)"";
	}
}
void glGetSynciv( GLsync, GLenum pname, GLsizei, GLsizei *len, GLint *v ) { if ( len ) *len = 1; if ( v ) *v = pname == GL_SYNC_STATUS_ ? GL_SIGNALED_ : 0; }
GLint glGetUniformLocation( GLuint prog, const GLchar *name )
{
	if ( !name || !s_programs.count( prog ) ) return -1;
	Program *p = s_programs[prog];
	// «имя[индекс]» -> имя и индекс; location = номер * 1024 + индекс
	std::string base = name;
	int idx = 0;
	size_t br = base.find( '[' );
	if ( br != std::string::npos ) { idx = atoi( base.c_str() + br + 1 ); base.resize( br ); }
	for ( size_t i = 0; i < p->uniforms.size(); i++ )
		if ( p->uniforms[i].name == base ) return (GLint)( i * 1024 + ( idx & 1023 ) );
	return -1;
}
GLboolean glIsEnabled( GLenum cap ) { return s_enabled[CapSlot( cap )]; }
GLboolean glIsTexture( GLuint t ) { return t ? GL_TRUE : GL_FALSE; }
void glLinkProgram( GLuint id )
{
		if ( !s_programs.count( id ) ) return;
	Program *p = s_programs[id];
	if ( !LinkProgram( id, p ) )
	{
		static int s_said;
		if ( s_said++ < 10 ) ps4glLog( "ps4gl: program %u failed to link (VS %u, PS %u)", id, p->vs, p->ps );
	}
}
void *glMapBufferRange( GLenum target, GLintptr off, GLsizeiptr len, GLbitfield access )
{
		Buffer *b = FindBuffer( s_boundBuffer[BufferSlot( target )] );
	if ( b && b->data )
	{
		if ( access & 0x0008 )						// MAP_INVALIDATE_BUFFER_BIT: старое не нужно
			BufferRealloc( b, b->size, false );
		else if ( !( access & 0x0020 ) )			// без MAP_UNSYNCHRONIZED_BIT — бережём рисования кадра
			BufferBeforeWrite( b );
	}
	if ( !b || !b->data || (size_t)( off + len ) > b->size )
	{
		static int s_bad;
		if ( s_bad++ < 5 ) ps4glLog( "ps4gl: glMapBufferRange out of range (%u, %lld+%lld)", s_boundBuffer[BufferSlot( target )], (long long)off, (long long)len );
		return NULL;
	}
	return (char *)b->data + off;
}
void glPixelStorei( GLenum p, GLint v ) { if ( p == GL_UNPACK_ALIGNMENT && v > 0 ) s_unpackAlign = v; }
void glPolygonOffset( GLfloat factor, GLfloat units ) { s_polyFactor = factor; s_polyUnits = units; }
void glReadBuffer( GLenum ) {}
void glReadPixels( GLint, GLint, GLsizei w, GLsizei h, GLenum, GLenum, void *px ) { if ( px && w > 0 && h > 0 ) memset( px, 0, (size_t)w * h * 4 ); }
void glRenderbufferStorageMultisample( GLenum, GLsizei, GLenum ifmt, GLsizei w, GLsizei h )
{
		if ( !s_boundRb ) return;
	Texture *&t = s_rbs[s_boundRb];
	if ( !t ) t = new Texture();
	bool depth = ifmt == 0x88F0 || ifmt == 0x81A5 || ifmt == 0x81A6 || ifmt == 0x81A7 || ifmt == 0x8CAC || ifmt == 0x8CAD;
	TexChooseFormat( t, ifmt, depth ? 0x1902 : GL_RGBA, GL_UNSIGNED_BYTE );
	TexAlloc( t, w, h );
}
void glSamplerParameterf( GLuint, GLenum, GLfloat ) {}		// LOD/анизотропия — позже
void glSamplerParameterfv( GLuint, GLenum, const GLfloat * ) {}
void glSamplerParameteri( GLuint smp, GLenum p, GLint v )
{
	SamplerObj &o = s_samplers[smp];
	if ( p == GL_TEXTURE_WRAP_S ) o.wrapS = v; else if ( p == GL_TEXTURE_WRAP_T ) o.wrapT = v;
	else if ( p == GL_TEXTURE_MIN_FILTER ) o.minF = v; else if ( p == GL_TEXTURE_MAG_FILTER ) o.magF = v;
}
void glScissor( GLint x, GLint y, GLsizei w, GLsizei h ) { s_scissor[0] = x; s_scissor[1] = y; s_scissor[2] = w; s_scissor[3] = h; }
void glShaderBinary( GLsizei n, const GLuint *ids, GLenum, const void *bin, GLsizei len )
{
		for ( GLsizei i = 0; i < n; i++ )
	{
		if ( !s_shaders.count( ids[i] ) ) continue;
		Shader *sh = s_shaders[ids[i]];
		sh->valid = ParseSb( sh, (const uint8_t *)bin, (size_t)len );
		if ( !sh->valid )
		{
			static int s_said;
			if ( s_said++ < 10 ) ps4glLog( "ps4gl: shader %u: bad binary (%d bytes)", ids[i], (int)len );
		}
	}
}
void glShaderSource( GLuint, GLsizei, const GLchar *const *, const GLint * ) { }
void glStencilFunc( GLenum, GLint, GLuint ) {}
void glStencilFuncSeparate( GLenum, GLenum, GLint, GLuint ) {}
void glStencilMask( GLuint ) {}
void glStencilOp( GLenum, GLenum, GLenum ) {}
void glStencilOpSeparate( GLenum, GLenum, GLenum, GLenum ) {}
void glTexImage2D( GLenum target, GLint level, GLint ifmt, GLsizei w, GLsizei h, GLint, GLenum fmt, GLenum type, const void *px )
{
		Texture *t = BoundTex( target );
	if ( t && level > 0 )
	{
		if ( t->mem && level < t->levels && w == t->mipW[level] && h == t->mipH[level] && ( px || s_boundBuffer[3] ) )
		{
			TexUpload( t, 0, 0, w, h, fmt, type, px, level );
			t->filled |= 1u << level;
		}
		return;
	}
	if ( !t || level != 0 ) return;	// кубы и 3D — позже
	TexChooseFormat( t, (GLenum)ifmt, fmt, type );
	if ( !TexAlloc( t, w, h ) ) return;
	if ( px || s_boundBuffer[3] ) TexUpload( t, 0, 0, w, h, fmt, type, px );
}
void glTexImage3D( GLenum, GLint, GLint, GLsizei, GLsizei, GLsizei, GLint, GLenum, GLenum, const void * ) { }
void glTexParameterfv( GLenum, GLenum, const GLfloat * ) {}
void glTexParameteri( GLenum target, GLenum p, GLint v )
{
	Texture *t = BoundTex( target );
	if ( !t ) return;
	if ( p == GL_TEXTURE_WRAP_S ) t->wrapS = v; else if ( p == GL_TEXTURE_WRAP_T ) t->wrapT = v;
	else if ( p == GL_TEXTURE_MIN_FILTER ) t->minF = v; else if ( p == GL_TEXTURE_MAG_FILTER ) t->magF = v;
}
void glTexSubImage2D( GLenum target, GLint level, GLint x, GLint y, GLsizei w, GLsizei h, GLenum fmt, GLenum type, const void *px )
{
		Texture *t = BoundTex( target );
	if ( !t || !t->mem || t->block != 1 || level < 0 || level >= t->levels ) return;
	TexUpload( t, x, y, w, h, fmt, type, px, level );
	if ( level > 0 && x == 0 && y == 0 && w == t->mipW[level] && h == t->mipH[level] ) { t->filled |= 1u << level; }
}
void glUniform1f( GLint loc, GLfloat v ) { UniformWrite( loc, &v, 4, false, 0 ); }
void glUniform1i( GLint loc, GLint v ) { UniformWrite( loc, &v, 4, true, v ); }
void glUniform4fv( GLint loc, GLsizei count, const GLfloat *v ) { if ( v && count > 0 ) UniformWrite( loc, v, (size_t)count * 16, false, 0 ); }
GLboolean glUnmapBuffer( GLenum ) { return GL_TRUE; }
void glUseProgram( GLuint id ) { s_curProgram = id; }
void glVertexAttribPointer( GLuint i, GLint size, GLenum type, GLboolean norm, GLsizei stride, const void *ptr )
{
		if ( i >= 16 ) return;
	Attrib &a = s_attr[i];
	a.size = size; a.type = type; a.norm = norm; a.stride = stride;
	a.buffer = s_boundBuffer[0]; a.offset = (uintptr_t)ptr;
}
void glViewport( GLint x, GLint y, GLsizei w, GLsizei h ) { s_viewport[0] = x; s_viewport[1] = y; s_viewport[2] = w; s_viewport[3] = h; }
void glWaitSync( GLsync, GLbitfield, GLuint64 ) {}
void glDrawRangeElementsBaseVertex( GLenum mode, GLuint, GLuint, GLsizei count, GLenum type, const void *idx, GLint base )
{
	DrawCommon( mode, count, true, type, idx, base, 0 );
}

// ---- остальное из OpenGL ES 2 ----
void glDrawElements( GLenum mode, GLsizei count, GLenum type, const void *idx ) { DrawCommon( mode, count, true, type, idx, 0, 0 ); }
GLint glGetAttribLocation( GLuint prog, const GLchar *name )
{
	if ( !name || !s_programs.count( prog ) ) return -1;
	Program *p = s_programs[prog];
	Shader *vs = p->linked && s_shaders.count( p->vs ) ? s_shaders[p->vs] : NULL;
	if ( !vs ) return -1;
	for ( int k = 0; k < p->nAttribs; k++ )
		if ( vs->ins[k] == name ) return p->vsAttribLoc[k];
	return -1;
}
void glUniform2f( GLint loc, GLfloat x, GLfloat y ) { const GLfloat v[2] = { x, y }; UniformWrite( loc, v, 8, false, 0 ); }
void glUniform3f( GLint loc, GLfloat x, GLfloat y, GLfloat z ) { const GLfloat v[3] = { x, y, z }; UniformWrite( loc, v, 12, false, 0 ); }
void glUniform4f( GLint loc, GLfloat x, GLfloat y, GLfloat z, GLfloat w ) { const GLfloat v[4] = { x, y, z, w }; UniformWrite( loc, v, 16, false, 0 ); }
void glUniform2i( GLint loc, GLint x, GLint y ) { const GLint v[2] = { x, y }; UniformWrite( loc, v, 8, false, 0 ); }
void glUniform3i( GLint loc, GLint x, GLint y, GLint z ) { const GLint v[3] = { x, y, z }; UniformWrite( loc, v, 12, false, 0 ); }
void glUniform4i( GLint loc, GLint x, GLint y, GLint z, GLint w ) { const GLint v[4] = { x, y, z, w }; UniformWrite( loc, v, 16, false, 0 ); }
// элементы массива лежат в буфере констант с шагом 16 байт
static void UniformArray( GLint loc, GLsizei count, const void *v, int comps )
{
	if ( !v || loc < 0 ) return;
	for ( GLsizei i = 0; i < count; i++ )
		UniformWrite( loc + i, (const uint8_t *)v + (size_t)i * comps * 4, (size_t)comps * 4, false, 0 );
}
void glUniform1fv( GLint loc, GLsizei count, const GLfloat *v ) { UniformArray( loc, count, v, 1 ); }
void glUniform2fv( GLint loc, GLsizei count, const GLfloat *v ) { UniformArray( loc, count, v, 2 ); }
void glUniform3fv( GLint loc, GLsizei count, const GLfloat *v ) { UniformArray( loc, count, v, 3 ); }
void glUniform1iv( GLint loc, GLsizei count, const GLint *v )
{
	if ( !v || loc < 0 ) return;
	for ( GLsizei i = 0; i < count; i++ ) UniformWrite( loc + i, &v[i], 4, true, v[i] );
}
void glUniform2iv( GLint loc, GLsizei count, const GLint *v ) { UniformArray( loc, count, v, 2 ); }
void glUniform3iv( GLint loc, GLsizei count, const GLint *v ) { UniformArray( loc, count, v, 3 ); }
void glUniform4iv( GLint loc, GLsizei count, const GLint *v ) { UniformArray( loc, count, v, 4 ); }
// матрицы лежат столбцами по 16 байт (и mat2, и mat3)
void glUniformMatrix2fv( GLint loc, GLsizei count, GLboolean, const GLfloat *v )
{
	if ( !v || loc < 0 ) return;
	for ( GLsizei i = 0; i < count; i++ )
		for ( int c = 0; c < 2; c++ ) UniformWrite( loc + i * 2 + c, v + i * 4 + c * 2, 8, false, 0 );
}
void glUniformMatrix3fv( GLint loc, GLsizei count, GLboolean, const GLfloat *v )
{
	if ( !v || loc < 0 ) return;
	for ( GLsizei i = 0; i < count; i++ )
		for ( int c = 0; c < 3; c++ ) UniformWrite( loc + i * 3 + c, v + i * 9 + c * 3, 12, false, 0 );
}
void glUniformMatrix4fv( GLint loc, GLsizei count, GLboolean, const GLfloat *v )
{
	if ( v && count > 0 ) UniformWrite( loc, v, (size_t)count * 64, false, 0 );
}
void glTexParameterf( GLenum target, GLenum p, GLfloat v ) { glTexParameteri( target, p, (GLint)v ); }
void glTexParameteriv( GLenum target, GLenum p, const GLint *v ) { if ( v ) glTexParameteri( target, p, v[0] ); }
void glCullFace( GLenum mode ) { s_cullFace = mode; }
void glBlendFuncSeparate( GLenum srcRgb, GLenum dstRgb, GLenum srcA, GLenum dstA )
{
	s_blendSrc = srcRgb; s_blendDst = dstRgb; s_blendSrcA = srcA; s_blendDstA = dstA;
}
void glBlendEquationSeparate( GLenum rgb, GLenum alpha ) { s_blendEq = rgb; s_blendEqA = alpha; }
void glRenderbufferStorage( GLenum target, GLenum ifmt, GLsizei w, GLsizei h ) { glRenderbufferStorageMultisample( target, 0, ifmt, w, h ); }
// мип-уровни усреднением 2x2 (только RGBA8 со сторонами — степенями двойки)
void glGenerateMipmap( GLenum target )
{
	Texture *t = BoundTex( target );
	if ( !t || !t->mem || t->bpe != 4 || t->block != 1 ) return;
	for ( int l = 1; l < t->levels; l++ )
	{
		const uint8_t *src = (const uint8_t *)t->mem + t->mipOff[l - 1];
		uint8_t *dst = (uint8_t *)t->mem + t->mipOff[l];
		const size_t sp = (size_t)t->mipPitch[l - 1] * 4;
		for ( int y = 0; y < t->mipH[l]; y++ )
		{
			uint8_t *d = dst + (size_t)y * t->mipPitch[l] * 4;
			const uint8_t *s0 = src + (size_t)y * 2 * sp, *s1 = s0 + sp;
			for ( int x = 0; x < t->mipW[l]; x++, d += 4, s0 += 8, s1 += 8 )
				for ( int c = 0; c < 4; c++ ) d[c] = (uint8_t)( ( s0[c] + s0[c + 4] + s1[c] + s1[c + 4] + 2 ) >> 2 );
		}
		t->filled |= 1u << l;
	}
}
void glLineWidth( GLfloat ) {}
void glHint( GLenum, GLenum ) {}
void glSampleCoverage( GLfloat, GLboolean ) {}
void glReleaseShaderCompiler( void ) {}
void glValidateProgram( GLuint ) {}
void glStencilMaskSeparate( GLenum, GLuint ) {}

}	// extern "C"

// Таблица имя -> функция для ps4glGetProcAddress
struct GlEntry { const char *name; void *fn; };
static const GlEntry s_glTable[] =
{
#define PS4GL_FN( name ) { #name, (void *)&name },
#include "gen/ps4gl_functions.inc"
#undef PS4GL_FN
	{ NULL, NULL }
};

void *ps4glGetProcAddress( const char *name )
{
	for ( const GlEntry *e = s_glTable; e->name; e++ )
		if ( !strcmp( e->name, name ) )
			return e->fn;
	return NULL;
}
