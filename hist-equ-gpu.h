#ifndef HISTOGRAM_COMMON_H
#define HISTOGRAM_COMMON_H

#define HISTOGRAM_GPU_BIN_COUNT 256 //define GUP bin counter
typedef unsigned int uint;
typedef unsigned char uchar;

#define WARP_SIZE 32  //define warp size
#define SHARED_MEMORY_BANKS 16 //maxmum value unclear
#define WARP_COUNT 6 //define warp count: 32x32=1024 threads in total

//Threadblock size
#define HISTOGRAM_GPU_THREADBLOCK_SIZE (WARP_COUNT * WARP_SIZE)

//Shared memory per threadblock
#define HISTOGRAM_GPU_THREADBLOCK_MEMORY (WARP_COUNT * HISTOGRAM_GPU_BIN_COUNT)

#define UMUL(a, b) ( (a) * (b) )
#define UMAD(a, b, c) ( UMUL((a), (b)) + (c) )

// GPU histogram
extern "C" void initHistorgramGpu(void);
extern "C" void closeHistorgramGpu(void);

extern "C" void historgramGpu(
    uint *d_Histogram,
    void *d_Data,
    uint byteCount
);

extern "C" void histogramEqu(
    uchar *d_ImgIn,
    uint *d_lut,
	uint numBlock,
	uint dataCount
);

extern "C" void rgb2hslGpu(
	uchar * img_inr, 
	uchar * img_ing, 
	uchar * img_inb, 
	float * img_outh, 
	float * img_outs, 
	uchar * img_outl,
	uint numBlock, 
	uint dataCount
);

extern "C" void hsl2rgbGpu(
	float * img_inh, 
	float * img_ins, 
	uchar * img_inl,
	uchar * img_outr, 
	uchar * img_outg, 
	uchar * img_outb, 
	uint numBlock, 
	uint dataCount
);

extern "C" void rgb2yuvGpu(
	uchar * img_inr, 
	uchar * img_ing, 
	uchar * img_inb, 
	uchar * img_outy, 
	uchar * img_outu, 
	uchar * img_outv,
	uint numBlock, 
	uint dataCount
);

extern "C" void yuv2rgbGpu(
	uchar * img_iny, 
	uchar * img_inu, 
	uchar * img_inv,
	uchar * img_outr, 
	uchar * img_outg, 
	uchar * img_outb, 
	uint numBlock, 
	uint dataCount
);

#endif
