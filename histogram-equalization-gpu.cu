#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hist-equ-gpu.h"
#include <time.h>


inline __device__ float Hue_2_RGB_GPU(float v1, float v2, float vH){
    if ( vH < 0 ) vH += 1;
    if ( vH > 1 ) vH -= 1;
    if ( ( 6 * vH ) < 1 ) return ( v1 + ( v2 - v1 ) * 6 * vH );
    if ( ( 2 * vH ) < 1 ) return ( v2 );
    if ( ( 3 * vH ) < 2 ) return ( v1 + ( v2 - v1 ) * ( ( 2.0f/3.0f ) - vH ) * 6 );
    return ( v1 );
}


inline __device__ unsigned char clip_rgb(int x)
{
    if(x > 255)
        return 255;
    if(x < 0)
        return 0;

    return (unsigned char)x;
}

inline __device__ void addByte(uint *s_WarpHist, uint data){
    atomicAdd(s_WarpHist + data, 1);
}

inline __device__ void addWord(uint *s_WarpHist, uint data){
    addByte(s_WarpHist, (data >>  0) & 0xFFU);
    addByte(s_WarpHist, (data >>  8) & 0xFFU);
    addByte(s_WarpHist, (data >> 16) & 0xFFU);
    addByte(s_WarpHist, (data >> 24) & 0xFFU);
}

__global__ void historgramGpuKernel(uint *d_PartialHistograms, uint *d_Data, uint dataCount)
{
 
	//set share_memory
    __shared__ uint s_Hist[HISTOGRAM_GPU_THREADBLOCK_MEMORY]; //every warp holds 256 bins in share memory
	uint *s_WarpHist= s_Hist + (threadIdx.x / WARP_SIZE) * HISTOGRAM_GPU_BIN_COUNT;

    //initialized share memory
	#pragma unroll
    for (uint i = 0; i < (HISTOGRAM_GPU_THREADBLOCK_MEMORY / HISTOGRAM_GPU_THREADBLOCK_SIZE); i++){
        s_Hist[threadIdx.x + i * HISTOGRAM_GPU_THREADBLOCK_SIZE] = 0;
    }

    __syncthreads();
    for (uint pos = UMAD(blockIdx.x, blockDim.x, threadIdx.x); pos < dataCount; pos += UMUL(blockDim.x, gridDim.x)){
        uint data = d_Data[pos];
        addWord(s_WarpHist, data);
    }

    //Merge per-warp histograms into per-block and write to global memory
    __syncthreads();
    for (uint bin = threadIdx.x; bin < HISTOGRAM_GPU_BIN_COUNT; bin += HISTOGRAM_GPU_THREADBLOCK_SIZE){
        uint sum = 0;

        for (uint i = 0; i < WARP_COUNT; i++){
            sum += s_Hist[bin + i * HISTOGRAM_GPU_BIN_COUNT];
        }

        d_PartialHistograms[blockIdx.x * HISTOGRAM_GPU_BIN_COUNT + bin] = sum;
    }
}

#define MERGE_THREADBLOCK_SIZE 256

__global__ void mergehistorgramGpuKernel(uint *d_Histogram, uint *d_PartialHistograms, uint histogramCount)
{
    uint sum = 0;

    for (uint i = threadIdx.x; i < histogramCount; i += MERGE_THREADBLOCK_SIZE){
        sum += d_PartialHistograms[blockIdx.x + i * HISTOGRAM_GPU_BIN_COUNT];
    }

    __shared__ uint data[MERGE_THREADBLOCK_SIZE];
    data[threadIdx.x] = sum;

    for (uint stride = MERGE_THREADBLOCK_SIZE / 2; stride > 0; stride >>= 1){
        __syncthreads();

        if (threadIdx.x < stride){
            data[threadIdx.x] += data[threadIdx.x + stride];
        }
    }

    if (threadIdx.x == 0)
    {
        d_Histogram[blockIdx.x] = data[0];
    }
}

__global__ void histogram_equalization_gpu(unsigned char * img_in, uint * lut, uint numBlock, uint dataCount)
{    
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	for (int index=tid + (bid*HISTOGRAM_GPU_THREADBLOCK_SIZE); index < (dataCount); index+=HISTOGRAM_GPU_THREADBLOCK_SIZE*(numBlock))
	{
		img_in[index] = (unsigned char)lut[img_in[index]];
	}
}

__global__ void rgb_to_hsl_gpu(
	unsigned char * img_inr, 
	unsigned char * img_ing, 
	unsigned char * img_inb, 
	float * img_outh, 
	float * img_outs, 
	unsigned char * img_outl,
	uint numBlock, 
	uint dataCount)
{
	int i;
    float H, S, L;
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	for (i = tid + (bid*HISTOGRAM_GPU_THREADBLOCK_SIZE); i < (dataCount); i += HISTOGRAM_GPU_THREADBLOCK_SIZE*(numBlock))
	{
		float var_r, var_g, var_b;
		var_r = ( (float)img_inr[i]/255 );//Convert RGB to [0,1]
        var_g = ( (float)img_ing[i]/255 );
        var_b = ( (float)img_inb[i]/255 );
        float var_min = (var_r < var_g) ? var_r : var_g;
        var_min = (var_min < var_b) ? var_min : var_b;   //min. value of RGB
        float var_max = (var_r > var_g) ? var_r : var_g;
        var_max = (var_max > var_b) ? var_max : var_b;   //max. value of RGB
        float del_max = var_max - var_min;               //Delta RGB value
        
        L = ( var_max + var_min ) / 2;

        if( del_max == 0 ){
            H = 0;         
            S = 0;    
        }
        else{
            if ( L < 0.5 )
                S = del_max/(var_max+var_min);
            else
                S = del_max/(2-var_max-var_min );

            float del_r = (((var_max-var_r)/6)+(del_max/2))/del_max;
            float del_g = (((var_max-var_g)/6)+(del_max/2))/del_max;
            float del_b = (((var_max-var_b)/6)+(del_max/2))/del_max;
            if( var_r == var_max ){
                H = del_b - del_g;
            }
            else{       
                if( var_g == var_max ){
                    H = (1.0/3.0) + del_r - del_b;
                }
                else{
                    H = (2.0/3.0) + del_g - del_r;
                }   
            }           
        }
        
        if ( H < 0 )
            H += 1;
        if ( H > 1 )
            H -= 1;

        img_outh[i] = H;
        img_outs[i] = S;
        img_outl[i] = (unsigned char)(L*255);
	}
}

__global__ void hsl_to_rgb_gpu(
	float * img_inh, 
	float * img_ins, 
	unsigned char * img_inl,
	unsigned char * img_outr, 
	unsigned char * img_outg, 
	unsigned char * img_outb, 
	uint numBlock, 
	uint dataCount)
{
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	for (int i = tid + (bid*HISTOGRAM_GPU_THREADBLOCK_SIZE); i < (dataCount); i += HISTOGRAM_GPU_THREADBLOCK_SIZE*(numBlock))
	{
		float H = img_inh[i];
        float S = img_ins[i];
        float L = img_inl[i]/255.0f;
        float var_1, var_2;
        
        unsigned char r,g,b;
        
        if ( S == 0 )
        {
            r = L * 255;
            g = L * 255;
            b = L * 255;
        }
        else
        {            
            if ( L < 0.5 )
                var_2 = L * ( 1 + S );
            else
                var_2 = ( L + S ) - ( S * L );

            var_1 = 2 * L - var_2;
            r = 255 * Hue_2_RGB_GPU( var_1, var_2, H + (1.0f/3.0f) );
            g = 255 * Hue_2_RGB_GPU( var_1, var_2, H );
            b = 255 * Hue_2_RGB_GPU( var_1, var_2, H - (1.0f/3.0f) );
        }
        img_outr[i] = r;
        img_outg[i] = g;
        img_outb[i] = b;
	}
}

__global__ void rgb_to_yuv_gpu(
	unsigned char * img_inr, 
	unsigned char * img_ing, 
	unsigned char * img_inb, 
	unsigned char * img_outy, 
	unsigned char * img_outu, 
	unsigned char * img_outv,
	uint numBlock, 
	uint dataCount)
{
	unsigned char r, g, b;
    unsigned char y, cb, cr;
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	for (int i = tid + (bid*HISTOGRAM_GPU_THREADBLOCK_SIZE); i < (dataCount); i += HISTOGRAM_GPU_THREADBLOCK_SIZE*(numBlock))
	{
		r = img_inr[i];
        g = img_ing[i];
        b = img_inb[i];
        
        y  = (unsigned char)( 0.299*r + 0.587*g +  0.114*b);
        cb = (unsigned char)(-0.169*r - 0.331*g +  0.499*b + 128);
        cr = (unsigned char)( 0.499*r - 0.418*g - 0.0813*b + 128);
        
        img_outy[i] = y;
        img_outu[i] = cb;
        img_outv[i] = cr;
	}
}

__global__ void yuv_to_rgb_gpu(
	unsigned char * img_iny, 
	unsigned char * img_inu, 
	unsigned char * img_inv,
	unsigned char * img_outr, 
	unsigned char * img_outg, 
	unsigned char * img_outb, 
	uint numBlock, 
	uint dataCount)
{
	int  rt,gt,bt;
    int y, cb, cr;
	const int tid = threadIdx.x;
	const int bid = blockIdx.x;
	for (int i = tid + (bid*HISTOGRAM_GPU_THREADBLOCK_SIZE); i < (dataCount); i += HISTOGRAM_GPU_THREADBLOCK_SIZE*(numBlock))
	{
		y  = (int)img_iny[i];
        cb = (int)img_inu[i] - 128;
        cr = (int)img_inv[i] - 128;
        
        rt  = (int)( y + 1.402*cr);
        gt  = (int)( y - 0.344*cb - 0.714*cr);
        bt  = (int)( y + 1.772*cb);

        img_outr[i] = clip_rgb(rt);
        img_outg[i] = clip_rgb(gt);
        img_outb[i] = clip_rgb(bt);
	}
}

static const uint PARTIAL_HISTOGRAM_GPU_COUNT = 256;
static uint *d_PartialHistograms;

extern "C" void initHistorgramGpu(void)
{
    cudaMalloc((void **)&d_PartialHistograms, PARTIAL_HISTOGRAM_GPU_COUNT * HISTOGRAM_GPU_BIN_COUNT * sizeof(uint));
}

extern "C" void closeHistorgramGpu(void)
{
	cudaFree(d_PartialHistograms);
}

extern "C" void historgramGpu(
    uint *d_Histogram,
    void *d_Data,
    uint byteCount
)
{
    assert(byteCount % sizeof(uint) == 0);
    historgramGpuKernel<<<PARTIAL_HISTOGRAM_GPU_COUNT, HISTOGRAM_GPU_THREADBLOCK_SIZE>>>(
        d_PartialHistograms,
        (uint *)d_Data,
        byteCount / sizeof(uint)
    );
//    getLastCudaError("historgramGpuKernel() execution failed\n");

    mergehistorgramGpuKernel<<<HISTOGRAM_GPU_BIN_COUNT, MERGE_THREADBLOCK_SIZE>>>(
        d_Histogram,
        d_PartialHistograms,
        PARTIAL_HISTOGRAM_GPU_COUNT
    );
//    getLastCudaError("mergehistorgramGpuKernel() execution failed\n");
}

extern "C" void histogramEqu(
    uchar *d_ImgIn,
    uint *d_lut,
	uint numBlock,
	uint byteCount
)
{
    histogram_equalization_gpu<<<numBlock, HISTOGRAM_GPU_THREADBLOCK_SIZE>>>(
        d_ImgIn,
        d_lut,
		numBlock,
		byteCount

    );
}

extern "C" void rgb2hslGpu(
	uchar * img_inr, 
	uchar * img_ing, 
	uchar * img_inb, 
	float * img_outh, 
	float * img_outs, 
	uchar * img_outl,
	uint numBlock, 
	uint dataCount
)
{
	rgb_to_hsl_gpu<<<numBlock, HISTOGRAM_GPU_THREADBLOCK_SIZE>>>(
    	img_inr, 
		img_ing, 
		img_inb, 
		img_outh, 
		img_outs, 
		img_outl,
		numBlock, 
		dataCount
    );
}

extern "C" void hsl2rgbGpu(
	float * img_inh, 
	float * img_ins, 
	uchar * img_inl,
	uchar * img_outr, 
	uchar * img_outg, 
	uchar * img_outb, 
	uint numBlock, 
	uint dataCount
)
{
	hsl_to_rgb_gpu<<<numBlock, HISTOGRAM_GPU_THREADBLOCK_SIZE>>>(
    	img_inh, 
		img_ins, 
		img_inl,
		img_outr, 
		img_outg, 
		img_outb, 
		numBlock, 
		dataCount
    );
}

extern "C" void rgb2yuvGpu(
	uchar * img_inr, 
	uchar * img_ing, 
	uchar * img_inb, 
	uchar * img_outy, 
	uchar * img_outu, 
	uchar * img_outv,
	uint numBlock, 
	uint dataCount
)
{
	rgb_to_yuv_gpu<<<numBlock, HISTOGRAM_GPU_THREADBLOCK_SIZE>>>(
    	img_inr, 
		img_ing, 
		img_inb, 
		img_outy, 
		img_outu, 
		img_outv,
		numBlock, 
		dataCount
    );
}

extern "C" void yuv2rgbGpu(
	uchar * img_iny, 
	uchar * img_inu, 
	uchar * img_inv,
	uchar * img_outr, 
	uchar * img_outg, 
	uchar * img_outb, 
	uint numBlock, 
	uint dataCount
)
{
	yuv_to_rgb_gpu<<<numBlock, HISTOGRAM_GPU_THREADBLOCK_SIZE>>>(
    	img_iny, 
		img_inu, 
		img_inv,
		img_outr, 
		img_outg, 
		img_outb, 
		numBlock, 
		dataCount
    );
}
