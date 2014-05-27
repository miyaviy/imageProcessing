#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// CUDA Runtime
#include <cuda_runtime.h>

// project include
#include "hist-equ-gpu.h"
#include "hist-equ.h"

#define numBlock 256

void getLUT(uint  *h_HistogramGPU, uint * lut, int lutSize, int img_size)
{
	uint cdf=0;
	int min=0;
	int i=0;
    while(min == 0){
        min = h_HistogramGPU[i++];
    }
    int d = img_size - min;
    for(i = 0; i < lutSize; i ++){
        cdf += h_HistogramGPU[i];
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }       
    }
}

uchar *getImgOut(uchar *init, uint byteCount)
{
	uchar *result;
	result = (unsigned char *)malloc(byteCount * sizeof(unsigned char));

    uint  *h_HistogramGPU;
    uchar *d_ImgIn;
    uint  *d_Lut;
    uint  *d_Histogram;

    h_HistogramGPU = (uint *)malloc(HISTOGRAM_GPU_BIN_COUNT * sizeof(uint));
	
	clock_t timer1 = clock();

    cudaMalloc((void **)&d_ImgIn, byteCount * sizeof(uchar));
    cudaMalloc((void **)&d_Histogram, HISTOGRAM_GPU_BIN_COUNT * sizeof(uint));
    cudaMemcpy(d_ImgIn, init, byteCount * sizeof(uchar), cudaMemcpyHostToDevice);

	initHistorgramGpu();
	historgramGpu(d_Histogram, d_ImgIn, byteCount);
	cudaDeviceSynchronize();

	cudaMemcpy(h_HistogramGPU, d_Histogram, HISTOGRAM_GPU_BIN_COUNT * sizeof(uint), cudaMemcpyDeviceToHost);

	printf("GPU His time: %f (ms)\n", (float)(clock()-timer1)/CLOCKS_PER_SEC);

	closeHistorgramGpu();
	cudaFree(d_Histogram);
    cudaFree(d_ImgIn);
    cudaDeviceReset();

	clock_t timer2 = clock();

	uint lut[256];
	getLUT(h_HistogramGPU, lut, 256, byteCount ); 

    cudaMalloc((void **)&d_ImgIn, byteCount * sizeof(uchar));
    cudaMalloc((void **)&d_Lut, HISTOGRAM_GPU_BIN_COUNT * sizeof(uint));
 
	cudaMemcpy(d_ImgIn, init, byteCount * sizeof(uchar), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Lut, lut, HISTOGRAM_GPU_BIN_COUNT * sizeof(uint), cudaMemcpyHostToDevice); 

	histogramEqu(d_ImgIn,d_Lut, numBlock,byteCount);

	cudaMemcpy(result, d_ImgIn, byteCount * sizeof(uchar), cudaMemcpyDeviceToHost);

	printf("GPU Equ time: %f (ms)\n", (float)(clock()-timer2)/CLOCKS_PER_SEC);

	cudaFree(d_ImgIn);
	cudaFree(d_Lut);

    return result;
}

PGM_IMG contrast_enhancement_gpu_g(PGM_IMG img_in)
{
	PGM_IMG result;
	result.w = img_in.w;
    result.h = img_in.h;
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
	result.img = getImgOut(img_in.img, img_in.w * img_in.h);

    return result;
}

PPM_IMG contrast_enhancement_g_rgb(PPM_IMG img_in)
{
    PPM_IMG result;
	result.w = img_in.w;
    result.h = img_in.h;
    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
	result.img_r = getImgOut(img_in.img_r, img_in.w * img_in.h);
	result.img_g = getImgOut(img_in.img_g, img_in.w * img_in.h);
	result.img_b = getImgOut(img_in.img_b, img_in.w * img_in.h);
 
    return result;
}


PPM_IMG contrast_enhancement_g_yuv(PPM_IMG img_in)
{
    YUV_IMG yuv_med;
    PPM_IMG result;
    
    unsigned char * y_equ;
    int hist[256];
    
    yuv_med = rgb2yuv_gpu(img_in);
    y_equ = (unsigned char *)malloc(yuv_med.h*yuv_med.w*sizeof(unsigned char));
	y_equ = getImgOut(yuv_med.img_y, yuv_med.h * yuv_med.w);
    
    free(yuv_med.img_y);
    yuv_med.img_y = y_equ;
    
    result = yuv2rgb_gpu(yuv_med);
    free(yuv_med.img_y);
    free(yuv_med.img_u);
    free(yuv_med.img_v);
    
    return result;
}

PPM_IMG contrast_enhancement_g_hsl(PPM_IMG img_in)
{
    HSL_IMG hsl_med;
    PPM_IMG result;
    
    unsigned char * l_equ;
    int hist[256];

    hsl_med = rgb2hsl_gpu(img_in);
    l_equ = (unsigned char *)malloc(hsl_med.height*hsl_med.width*sizeof(unsigned char));
	l_equ = getImgOut(hsl_med.l, hsl_med.width*hsl_med.height);

    free(hsl_med.l);
    hsl_med.l = l_equ;

    result = hsl2rgb_gpu(hsl_med);
    free(hsl_med.h);
    free(hsl_med.s);
    free(hsl_med.l);
    return result;
}


//Convert RGB to HSL, assume R,G,B in [0, 255]
//Output H, S in [0.0, 1.0] and L in [0, 255]
HSL_IMG rgb2hsl_gpu(PPM_IMG img_in)
{
    HSL_IMG img_out;
    img_out.width  = img_in.w;
    img_out.height = img_in.h;
    img_out.h = (float *)malloc(img_in.w * img_in.h * sizeof(float));
    img_out.s = (float *)malloc(img_in.w * img_in.h * sizeof(float));
    img_out.l = (unsigned char *)malloc(img_in.w * img_in.h * sizeof(unsigned char));

    uchar *d_ImgInr;
	uchar *d_ImgIng;
	uchar *d_ImgInb;
	float *d_ImgOuth;
	float *d_ImgOuts;
	uchar *d_ImgOutl;
	uint byteCount = img_in.w * img_in.h;

	clock_t timer1 = clock();

    cudaMalloc((void **)&d_ImgInr, byteCount * sizeof(uchar));
	cudaMalloc((void **)&d_ImgIng, byteCount * sizeof(uchar));
	cudaMalloc((void **)&d_ImgInb, byteCount * sizeof(uchar));
	cudaMalloc((void **)&d_ImgOuth, byteCount * sizeof(float));
	cudaMalloc((void **)&d_ImgOuts, byteCount * sizeof(float));
	cudaMalloc((void **)&d_ImgOutl, byteCount * sizeof(uchar));
 
	cudaMemcpy(d_ImgInr, img_in.img_r, byteCount * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ImgIng, img_in.img_g, byteCount * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ImgInb, img_in.img_b, byteCount * sizeof(uchar), cudaMemcpyHostToDevice);

	rgb2hslGpu(d_ImgInr, d_ImgIng, d_ImgInb, d_ImgOuth, d_ImgOuts, d_ImgOutl, numBlock, byteCount);

	cudaMemcpy(img_out.h, d_ImgOuth, byteCount * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(img_out.s, d_ImgOuts, byteCount * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(img_out.l, d_ImgOutl, byteCount * sizeof(uchar), cudaMemcpyDeviceToHost);

	printf("GPU RGB-HSL trans time: %f (ms)\n", (float)(clock()-timer1)/CLOCKS_PER_SEC);

	cudaFree(d_ImgInr);
	cudaFree(d_ImgIng);
	cudaFree(d_ImgInb);
	cudaFree(d_ImgOuth);
	cudaFree(d_ImgOuts);
	cudaFree(d_ImgOutl);

    return img_out;
}

PPM_IMG hsl2rgb_gpu(HSL_IMG img_in)
{
    PPM_IMG result;
    
    result.w = img_in.width;
    result.h = img_in.height;
    result.img_r = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_g = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    result.img_b = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    
	float *d_ImgInh;
	float *d_ImgIns;
	uchar *d_ImgInl;
	uchar *d_ImgOutr;
	uchar *d_ImgOutg;
	uchar *d_ImgOutb;
	uint byteCount = img_in.height * img_in.width;

	clock_t timer1 = clock();

    cudaMalloc((void **)&d_ImgInh, byteCount * sizeof(float));
	cudaMalloc((void **)&d_ImgIns, byteCount * sizeof(float));
	cudaMalloc((void **)&d_ImgInl, byteCount * sizeof(uchar));
	cudaMalloc((void **)&d_ImgOutr, byteCount * sizeof(uchar));
	cudaMalloc((void **)&d_ImgOutg, byteCount * sizeof(uchar));
	cudaMalloc((void **)&d_ImgOutb, byteCount * sizeof(uchar));
 
	cudaMemcpy(d_ImgInh, img_in.h, byteCount * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ImgIns, img_in.s, byteCount * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ImgInl, img_in.l, byteCount * sizeof(uchar), cudaMemcpyHostToDevice);

	hsl2rgbGpu(d_ImgInh, d_ImgIns, d_ImgInl, d_ImgOutr, d_ImgOutg, d_ImgOutb, numBlock, byteCount);

	cudaMemcpy(result.img_r, d_ImgOutr, byteCount * sizeof(uchar), cudaMemcpyDeviceToHost);
	cudaMemcpy(result.img_g, d_ImgOutg, byteCount * sizeof(uchar), cudaMemcpyDeviceToHost);
	cudaMemcpy(result.img_b, d_ImgOutb, byteCount * sizeof(uchar), cudaMemcpyDeviceToHost);

	printf("GPU HSL-RGB trans time: %f (ms)\n", (float)(clock()-timer1)/CLOCKS_PER_SEC);

	cudaFree(d_ImgInh);
	cudaFree(d_ImgIns);
	cudaFree(d_ImgInl);
	cudaFree(d_ImgOutr);
	cudaFree(d_ImgOutg);
	cudaFree(d_ImgOutb);

    return result;
}

YUV_IMG rgb2yuv_gpu(PPM_IMG img_in)
{
    YUV_IMG img_out;   
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_y = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_u = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_v = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);

    uchar *d_ImgInr;
	uchar *d_ImgIng;
	uchar *d_ImgInb;
	uchar *d_ImgOuty;
	uchar *d_ImgOutu;
	uchar *d_ImgOutv;
	uint byteCount = img_in.w * img_in.h;

	clock_t timer1 = clock();

    cudaMalloc((void **)&d_ImgInr, byteCount * sizeof(uchar));
	cudaMalloc((void **)&d_ImgIng, byteCount * sizeof(uchar));
	cudaMalloc((void **)&d_ImgInb, byteCount * sizeof(uchar));
	cudaMalloc((void **)&d_ImgOuty, byteCount * sizeof(uchar));
	cudaMalloc((void **)&d_ImgOutu, byteCount * sizeof(uchar));
	cudaMalloc((void **)&d_ImgOutv, byteCount * sizeof(uchar));
 
	cudaMemcpy(d_ImgInr, img_in.img_r, byteCount * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ImgIng, img_in.img_g, byteCount * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ImgInb, img_in.img_b, byteCount * sizeof(uchar), cudaMemcpyHostToDevice);

	rgb2yuvGpu(d_ImgInr, d_ImgIng, d_ImgInb, d_ImgOuty, d_ImgOutu, d_ImgOutv, numBlock, byteCount);

	cudaMemcpy(img_out.img_y, d_ImgOuty, byteCount * sizeof(uchar), cudaMemcpyDeviceToHost);
	cudaMemcpy(img_out.img_u, d_ImgOutu, byteCount * sizeof(uchar), cudaMemcpyDeviceToHost);
	cudaMemcpy(img_out.img_v, d_ImgOutv, byteCount * sizeof(uchar), cudaMemcpyDeviceToHost);

	printf("GPU RGB-YUV trans time: %f (ms)\n", (float)(clock()-timer1)/CLOCKS_PER_SEC);

	cudaFree(d_ImgInr);
	cudaFree(d_ImgIng);
	cudaFree(d_ImgInb);
	cudaFree(d_ImgOuty);
	cudaFree(d_ImgOutu);
	cudaFree(d_ImgOutv);
   
    return img_out;
}

PPM_IMG yuv2rgb_gpu(YUV_IMG img_in)
{
    PPM_IMG img_out;       
    img_out.w = img_in.w;
    img_out.h = img_in.h;
    img_out.img_r = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_g = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);
    img_out.img_b = (unsigned char *)malloc(sizeof(unsigned char)*img_out.w*img_out.h);

	uchar *d_ImgIny;
	uchar *d_ImgInu;
	uchar *d_ImgInv;
	uchar *d_ImgOutr;
	uchar *d_ImgOutg;
	uchar *d_ImgOutb;
	uint byteCount = img_in.w * img_in.h;

	clock_t timer1 = clock();

    cudaMalloc((void **)&d_ImgIny, byteCount * sizeof(uchar));
	cudaMalloc((void **)&d_ImgInu, byteCount * sizeof(uchar));
	cudaMalloc((void **)&d_ImgInv, byteCount * sizeof(uchar));
	cudaMalloc((void **)&d_ImgOutr, byteCount * sizeof(uchar));
	cudaMalloc((void **)&d_ImgOutg, byteCount * sizeof(uchar));
	cudaMalloc((void **)&d_ImgOutb, byteCount * sizeof(uchar));
 
	cudaMemcpy(d_ImgIny, img_in.img_y, byteCount * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ImgInu, img_in.img_u, byteCount * sizeof(uchar), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ImgInv, img_in.img_v, byteCount * sizeof(uchar), cudaMemcpyHostToDevice);

	yuv2rgbGpu(d_ImgIny, d_ImgInu, d_ImgInv, d_ImgOutr, d_ImgOutg, d_ImgOutb, numBlock, byteCount);

	cudaMemcpy(img_out.img_r, d_ImgOutr, byteCount * sizeof(uchar), cudaMemcpyDeviceToHost);
	cudaMemcpy(img_out.img_g, d_ImgOutg, byteCount * sizeof(uchar), cudaMemcpyDeviceToHost);
	cudaMemcpy(img_out.img_b, d_ImgOutb, byteCount * sizeof(uchar), cudaMemcpyDeviceToHost);

	printf("GPU YUV-RGB trans time: %f (ms)\n", (float)(clock()-timer1)/CLOCKS_PER_SEC);

	cudaFree(d_ImgIny);
	cudaFree(d_ImgInu);
	cudaFree(d_ImgInv);
	cudaFree(d_ImgOutr);
	cudaFree(d_ImgOutg);
	cudaFree(d_ImgOutb);
     
    return img_out;
}
