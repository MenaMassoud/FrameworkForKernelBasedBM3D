/*
 * Copyright (c) 2011, Marc Lebrun <marc.lebrun@cmla.ens-cachan.fr>
 * All rights reserved.
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either
 * version 3 of the License, or (at your option) any later
 * version. You should have received a copy of this license along
 * this program. If not, see <http://www.gnu.org/licenses/>.
 */


/**
 * @file bm3d.cpp
 * @brief BM3D denoising functions
 *
 * @author Marc Lebrun <marc.lebrun@cmla.ens-cachan.fr>
 **/


#include <iostream>
#include <algorithm>
#include <math.h>

#include <stdio.h>
#include <stdlib.h>
#include <bits/stdc++.h> 
#include <numeric>  
#include <unordered_map>
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_fit.h>
#include <vector>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_multilarge_nlinear.h>

#include "bm3d.h"
#include "utilities.h"
#include "lib_transforms.h"

#define PI 3.14159265
#define SQRT2     1.414213562373095
#define SQRT2_INV 0.7071067811865475
#define YUV       0
#define YCBCR     1
#define OPP       2
#define RGB       3
#define DCT       4
#define BIOR      5
#define HADAMARD  6

#ifdef _OPENMP
    #include <omp.h>
#endif

using namespace std;

bool ComparaisonFirst(pair<float,unsigned> pair1, pair<float,unsigned> pair2)
{
	return pair1.first < pair2.first;
}

/** ----------------- **/
/** - Main function - **/
/** ----------------- **/
/**
 * @brief run BM3D process. Depending on if OpenMP is used or not,
 *        and on the number of available threads, it divides the noisy
 *        image in sub_images, to process them in parallel.
 *
 * @param sigma: value of assumed noise of the noisy image;
 * @param img_noisy: noisy image;
 * @param img_basic: will be the basic estimation after the 1st step
 * @param img_denoised: will be the denoised final image;
 * @param width, height, chnls: size of the image;
 * @param useSD_h (resp. useSD_w): if true, use weight based
 *        on the standard variation of the 3D group for the
 *        first (resp. second) step, otherwise use the number
 *        of non-zero coefficients after Hard Thresholding
 *        (resp. the norm of Wiener coefficients);
 * @param tau_2D_hard (resp. tau_2D_wien): 2D transform to apply
 *        on every 3D group for the first (resp. second) part.
 *        Allowed values are DCT and BIOR;
 * @param color_space: Transformation from RGB to YUV. Allowed
 *        values are RGB (do nothing), YUV, YCBCR and OPP.
 *
 * @return EXIT_FAILURE if color_space has not expected
 *         type, otherwise return EXIT_SUCCESS.
 **/
int run_bm3d(
    const float sigma
    ,   vector<float> &img
    ,   vector<float> &img_noisy
    ,   vector<float> &img_basic
    ,   vector<float> &img_denoised
    ,   const unsigned width
    ,   const unsigned height
    ,   const unsigned chnls
    ,   const bool useSD_h
    ,   const bool useSD_w
    ,   const unsigned tau_2D_hard
    ,   const unsigned tau_2D_wien
    ,   const unsigned color_space
    ,  int image_no
    ){
    //! Parameters
    const unsigned nHard = 16; //! Half size of the search window
    const unsigned nWien = 16; //! Half size of the search window
    const unsigned kHard = (tau_2D_hard == BIOR || sigma < 40.f ? 8 : 12); //! Must be a power of 2 if tau_2D_hard == BIOR
    const unsigned kWien = (tau_2D_wien == BIOR || sigma < 40.f ? 8 : 12); //! Must be a power of 2 if tau_2D_wien == BIOR
    const unsigned NHard = 16; //! Must be a power of 2
    const unsigned NWien = 32; //! Must be a power of 2
    const unsigned pHard = 3;
    const unsigned pWien = 3;
    vector<float>img_basic_s,img_basic_c ,img_basic_h ,img_basic_v ,img_basic_D135 ,img_basic_D45;
    vector<float>img_denoised_s, img_denoised_c,img_denoised_hor ,img_denoised_ver ,img_denoised_D135 ,img_denoised_D45;
    //! Check memory allocation
    if (img_basic.size() != img_noisy.size())
        img_basic.resize(img_noisy.size());
    if(img_basic_s.size() != img_noisy.size())
        img_basic_s.resize(img_noisy.size());
    if(img_basic_c.size() != img_noisy.size())
        img_basic_c.resize(img_noisy.size());
    if(img_basic_h.size() != img_noisy.size())
        img_basic_h.resize(img_noisy.size());
    if(img_basic_v.size() != img_noisy.size())
        img_basic_v.resize(img_noisy.size());
    if(img_basic_D135.size() != img_noisy.size())
        img_basic_D135.resize(img_noisy.size());
    if(img_basic_D45.size() != img_noisy.size())
        img_basic_D45.resize(img_noisy.size());

    if (img_denoised.size() != img_noisy.size())
        img_denoised.resize(img_noisy.size());
    if (img_denoised_hor.size() != img_noisy.size())
        img_denoised_hor.resize(img_noisy.size());
    if (img_denoised_ver.size() != img_noisy.size())
        img_denoised_ver.resize(img_noisy.size());
    if (img_denoised_D135.size() != img_noisy.size())
        img_denoised_D135.resize(img_noisy.size());
    if (img_denoised_D45.size() != img_noisy.size())
        img_denoised_D45.resize(img_noisy.size());
    if (img_denoised_s.size() != img_noisy.size())
        img_denoised_s.resize(img_noisy.size());
    if (img_denoised_c.size() != img_noisy.size())
        img_denoised_c.resize(img_noisy.size());

    //! Transformation to YUV color space
    if (color_space_transform(img_noisy, color_space, width, height, chnls, true)
        != EXIT_SUCCESS) return EXIT_FAILURE;

    //! Check if OpenMP is used or if number of cores of the computer is > 1
        unsigned nb_threads = 1;

#ifdef _OPENMP
    cout << "Open MP used" << endl;
    nb_threads = omp_get_num_procs();

    //! In case where the number of processors isn't a power of 2
    if (!power_of_2(nb_threads))
        nb_threads = closest_power_of_2(nb_threads);
#endif

    cout << endl << "Number of threads which will be used: " << nb_threads;
#ifdef _OPENMP
    cout << " (real available cores: " << omp_get_num_procs() << ")" << endl;
#endif

    //! Allocate plan for FFTW library
    fftwf_plan plan_2d_for_1[nb_threads];
    fftwf_plan plan_2d_for_2[nb_threads];
    fftwf_plan plan_2d_inv[nb_threads];

    //! In the simple case
    if (nb_threads == 1)
    {
        //! Add boundaries and symetrize them
        const unsigned h_b = height + 2 * nHard;
        const unsigned w_b = width  + 2 * nHard;
        vector<float> img_sym, img_sym_noisy, img_sym_basic, img_sym_denoised;
        vector<float> img_sym_basic_s,img_sym_basic_c,img_sym_basic_h,img_sym_basic_v,img_sym_basic_D135,img_sym_basic_D45;
        symetrize(img_noisy, img_sym_noisy, width, height, chnls, nHard);

        //! Allocating Plan for FFTW process
        if (tau_2D_hard == DCT)
        {
            const unsigned nb_cols = ind_size(w_b - kHard + 1, nHard, pHard);
            allocate_plan_2d(&plan_2d_for_1[0], kHard, FFTW_REDFT10,
                w_b * (2 * nHard + 1) * chnls);
            allocate_plan_2d(&plan_2d_for_2[0], kHard, FFTW_REDFT10,
                w_b * pHard * chnls);
            allocate_plan_2d(&plan_2d_inv  [0], kHard, FFTW_REDFT01,
                NHard * nb_cols * chnls);
        }

        //! Denoising, 1st Step
        cout << "step 1...";
        bm3d_1st_step(sigma, img_sym_noisy, img_sym_basic,
            img_sym_basic_s,img_sym_basic_c,img_sym_basic_h,img_sym_basic_v,img_sym_basic_D135,img_sym_basic_D45,
            w_b, h_b, chnls, nHard,kHard, NHard, pHard, useSD_h, color_space, tau_2D_hard,
            &plan_2d_for_1[0], &plan_2d_for_2[0], &plan_2d_inv[0]);
        cout << "done." << endl;

        //! To avoid boundaries problem
for (unsigned c = 0; c < chnls; c++)
{
    const unsigned dc_b = c * w_b * h_b + nHard * w_b + nHard;
    unsigned dc = c * width * height;
    for (unsigned i = 0; i < height; i++)
        for (unsigned j = 0; j < width; j++, dc++){
            img_basic[dc] = img_sym_basic[dc_b + i * w_b + j];
            img_basic_s[dc] = img_sym_basic_s[dc_b + i * w_b + j];
            img_basic_c[dc] = img_sym_basic_c[dc_b + i * w_b + j];
            img_basic_h[dc] = img_sym_basic_h[dc_b + i * w_b + j];
            img_basic_v[dc] = img_sym_basic_v[dc_b + i * w_b + j];
            img_basic_D135[dc] = img_sym_basic_D135[dc_b + i * w_b + j];
            img_basic_D45[dc] = img_sym_basic_D45[dc_b + i * w_b + j];
        }
    }
    symetrize(img_basic, img_sym_basic, width, height, chnls, nHard);
    symetrize(img_basic_s, img_sym_basic_s, width, height, chnls, nHard);
    symetrize(img_basic_c, img_sym_basic_c, width, height, chnls, nHard);
    symetrize(img_basic_h, img_sym_basic_h, width, height, chnls, nHard);
    symetrize(img_basic_v, img_sym_basic_v, width, height, chnls, nHard);
    symetrize(img_basic_D135, img_sym_basic_D135, width, height, chnls, nHard);
    symetrize(img_basic_D45, img_sym_basic_D45, width, height, chnls, nHard);

//calculate psnr
    float psnr_s, rmse_s,psnr_c,rmse_c, psnr_h, rmse_h, psnr_basic, rmse_basic;
    float psnr_v, rmse_v, psnr_D135, rmse_D135, psnr_D45, rmse_D45;

if(compute_psnr(img, img_basic, &psnr_basic, &rmse_basic) != EXIT_SUCCESS)
   cout<<"Error calculating psnr";
if(compute_psnr(img, img_basic_s, &psnr_s, &rmse_s) != EXIT_SUCCESS)
    cout<<"Error calculating psnr";
if(compute_psnr(img, img_basic_c, &psnr_c, &rmse_c) != EXIT_SUCCESS)
    cout<<"Error calculating psnr";
if(compute_psnr(img, img_basic_h, &psnr_h, &rmse_h) != EXIT_SUCCESS)
   cout<<"Error calculating psnr";
if(compute_psnr(img, img_basic_v, &psnr_v, &rmse_v) != EXIT_SUCCESS)
   cout<<"Error calculating psnr";
if(compute_psnr(img, img_basic_D135, &psnr_D135, &rmse_D135) != EXIT_SUCCESS)
   cout<<"Error calculating psnr";
if(compute_psnr(img, img_basic_D45, &psnr_D45, &rmse_D45) != EXIT_SUCCESS)
   cout<<"Error calculating psnr";


cout << endl << "For stage 1 image :" << endl;
cout << "(basic image) :" << endl;
cout << "PSNR basic: " << psnr_basic << endl;
cout << "RMSE basic: " << rmse_basic << endl << endl;
cout << "(squ image) :" << endl;
cout << "PSNR squ: " << psnr_s << endl;
cout << "RMSE squ: " << rmse_s << endl << endl;
cout << "(circle image) :" << endl;
cout << "PSNR cir: " << psnr_c << endl;
cout << "RMSE cir: " << rmse_c << endl << endl;
cout << "(hor image) :" << endl;
cout << "PSNR hor: " << psnr_h << endl;
cout << "RMSE hor: " << rmse_h << endl << endl;
cout << "(ver image) :" << endl;
cout << "PSNR ver: " << psnr_v << endl;
cout << "RMSE ver: " << rmse_v << endl << endl;
cout << "(D135 image) :" << endl;
cout << "PSNR D135: " << psnr_D135 << endl;
cout << "RMSE D135: " << rmse_D135 << endl << endl;
cout << "(D45 image) :" << endl;
cout << "PSNR D45: " << psnr_D45 << endl;
cout << "RMSE D45: " << rmse_D45 << endl << endl;

char *ImbasicSqu = (char*)malloc(25);
sprintf(ImbasicSqu, "ImgSt1Squ%d.%s", (int)sigma,"png");
if (save_image(ImbasicSqu, img_basic_s, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
char *ImbasicCir = (char*)malloc(25);
sprintf(ImbasicCir, "ImgSt1Cir%d.%s", (int)sigma,"png");
if (save_image(ImbasicCir, img_basic_c, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
char *ImbasicHor = (char*)malloc(25);
sprintf(ImbasicHor, "ImgSt1Hor%d.%s", (int)sigma,"png");
if (save_image(ImbasicHor, img_basic_h, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
char *ImbasicVer = (char*)malloc(25);
sprintf(ImbasicVer, "ImgSt1Ver%d.%s", (int)sigma,"png");
if (save_image(ImbasicVer, img_basic_v, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
char *ImbasicD135 = (char*)malloc(25);
sprintf(ImbasicD135, "ImgSt1D135%d.%s", (int)sigma,"png");
if (save_image(ImbasicD135, img_basic_D135, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
char *ImbasicD45 = (char*)malloc(25);
sprintf(ImbasicD45, "ImgSt1D45%d.%s", (int)sigma,"png");
if (save_image(ImbasicD45, img_basic_D45, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}



// char *Imbasic = (char*)malloc(25);
// sprintf(Imbasic, "ImgBasic%d.%s", (int)sigma,"png");
// if (save_image(Imbasic, img_basic, width, height, chnls) != EXIT_SUCCESS){
//  cout<<"Error";
// }

std::vector<float> OptimalVal;
OptimalVal.resize(img_noisy.size());
std::vector<float> OptimalSqu;
OptimalSqu.resize(img_noisy.size());
std::vector<float> OptimalCir;
OptimalCir.resize(img_noisy.size());
std::vector<float> OptimalHor;
OptimalHor.resize(img_noisy.size());
std::vector<float> OptimalVer;
OptimalVer.resize(img_noisy.size());
std::vector<float> OptimalD135;
OptimalD135.resize(img_noisy.size());
std::vector<float> OptimalD45;
OptimalD45.resize(img_noisy.size());

int count_s = 0;
int count_c = 0;
int count_h = 0;
int count_v = 0;
int count_D135 = 0;
int count_D45 = 0;

//calculate the optimal solution
for(int m = 0; m < width;m++){
    for(int n =0; n<height; n++){
       float ress= abs(img[m+n*width] - img_basic_s[m+n*width]);
       float resc = abs(img[m+n*width] - img_basic_c[m+n*width]);
       float resh = abs(img[m+n*width] - img_basic_h[m+n*width]);
       float resv = abs(img[m+n*width] - img_basic_v[m+n*width]);
        float res135 = abs(img[m+n*width] - img_basic_D135[m+n*width]);
        float res45 = abs(img[m+n*width] - img_basic_D45[m+n*width]);

        std::vector<float> resp_val = {ress,resc,resh,resv,res135,res45};

        int miniInd = std::min_element(resp_val.begin(),resp_val.end()) - resp_val.begin();
        switch(miniInd){
            case 0:
            OptimalVal[m+n*width] = img_basic_s[m+n*width];
            OptimalSqu[m+n*width] = img_basic_s[m+n*width];
            count_s++;
            break;
            case 1:
            OptimalVal[m+n*width] = img_basic_c[m+n*width];
            OptimalCir[m+n*width] = img_basic_c[m+n*width];
            count_c++;
            break;
            case 2:
            OptimalVal[m+n*width] = img_basic_h[m+n*width];
            OptimalHor[m+n*width] = img_basic_h[m+n*width];
            count_h++;
            break;
            case 3:
            OptimalVal[m+n*width] = img_basic_v[m+n*width];
            OptimalVer[m+n*width] = img_basic_v[m+n*width];
            count_v++;
            break;
            case 4:
            OptimalVal[m+n*width] = img_basic_D135[m+n*width];
            OptimalD135[m+n*width] = img_basic_D135[m+n*width];
            count_D135++;
            break;
            default:
            OptimalVal[m+n*width] = img_basic_D45[m+n*width];
            OptimalD45[m+n*width] = img_basic_D45[m+n*width];
            count_D45++;

        }

    }
}


float psnrNSE, rmseNSE;
if(compute_psnr(img, img_noisy, &psnrNSE, &rmseNSE) != EXIT_SUCCESS)
  cout<<"Error";

float psnrOpt, rmseOpt;
if(compute_psnr(img, OptimalVal, &psnrOpt, &rmseOpt) != EXIT_SUCCESS)
  cout<<"Error";


char *ImNoisy = (char*)malloc(25);
sprintf(ImNoisy, "ImgNoisy%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImNoisy, img_noisy , width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImNoisy);
char *ImOptimal = (char*)malloc(25);
sprintf(ImOptimal, "ImprovementSt1%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImOptimal, OptimalVal, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImOptimal);
char *ImBasic = (char*)malloc(25);
sprintf(ImBasic, "ImOriginalSt1%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImBasic, img_basic, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImBasic);
vector<float>img_basic1;
img_basic1.resize(img_noisy.size());
vector<float>img_basic2;
img_basic2.resize(img_noisy.size());
vector<float>img_basic3;
img_basic3.resize(img_noisy.size());
vector<float>img_basic4;
img_basic4.resize(img_noisy.size());
vector<float>img_basic5;
img_basic5.resize(img_noisy.size());
vector<float>img_basic6;
img_basic6.resize(img_noisy.size());

int BlockSizeR = 3;
int BlockSizeC = 3;
int n = BlockSizeR*BlockSizeC;
//mse
vector<float>patch_mse;
patch_mse.resize(img_noisy.size());

for(int row= 0; row < width ;  row += BlockSizeR){
    for(int col = 0 ; col < height ; col += BlockSizeC){

unsigned  row1 = row;
unsigned row2 = row1 + BlockSizeR - 1;
row2 = min(width,row2);// to prevent going outside image borders

unsigned col1 = col;
unsigned col2 = col1 + BlockSizeC - 1;
col2 = min(height,col2);

float patch_vec_s[BlockSizeR*BlockSizeC] = {};
float patch_vec_c[BlockSizeR*BlockSizeC] = {};
float patch_vec_h[BlockSizeR*BlockSizeC] = {};
float patch_vec_v[BlockSizeR*BlockSizeC] = {};
float patch_vec_d135[BlockSizeR*BlockSizeC] = {};
float patch_vec_d45[BlockSizeR*BlockSizeC] = {};
int pixel_no = 0;
//verify that the block have the correct size
if((row2 - row1 + 1)== BlockSizeR && (col2 - col1 + 1) == BlockSizeC){
    for(int x = row1; x<=row2 ; x++){
        for(int y = col1; y<=col2; y++){
           patch_vec_s [pixel_no] =pow(img_basic_s[x+y*width]-img[x+y*width],2);
           patch_vec_c [pixel_no] =pow(img_basic_c[x+y*width]-img[x+y*width],2);
           patch_vec_h [pixel_no] = pow(img_basic_h[x+y*width]-img[x+y*width],2);
           patch_vec_v [pixel_no] = pow(img_basic_v[x+y*width]-img[x+y*width],2);
           patch_vec_d135 [pixel_no] = pow(img_basic_D135[x+y*width]-img[x+y*width],2);
           patch_vec_d45 [pixel_no] = pow(img_basic_D45[x+y*width]-img[x+y*width],2);
           pixel_no ++;

       }
   //  cout<<endl;
   }
  // cout<<endl<<endl;
    //calculate the mse of each patch and compare
   float sum_s = accumulate(patch_vec_s , patch_vec_s + (BlockSizeR*BlockSizeC), 0.0);
   float MSE_psnr_s = sqrtf(sum_s / (float)(BlockSizeR*BlockSizeC));
   float psnr_s = 20.0f * log10f(255.0/MSE_psnr_s);

    float sum_c = accumulate(patch_vec_c , patch_vec_c + (BlockSizeR*BlockSizeC), 0.0);
   float MSE_psnr_c = sqrtf(sum_c / (float)(BlockSizeR*BlockSizeC));
   float psnr_c = 20.0f * log10f(255.0/MSE_psnr_c);
//cout<<sum_s<<endl;
   float sum_h = accumulate(patch_vec_h , patch_vec_h + (BlockSizeR*BlockSizeC), 0.0);
   float MSE_psnr_h = sqrtf(sum_h / (float)(BlockSizeR*BlockSizeC));
   float psnr_h = 20.0f * log10f(255.0/MSE_psnr_h);

   float sum_v = accumulate(patch_vec_v , patch_vec_v + (BlockSizeR*BlockSizeC), 0.0);
   float MSE_psnr_v = sqrtf(sum_v / (float)(BlockSizeR*BlockSizeC));
   float psnr_v = 20.0f * log10f(255.0/MSE_psnr_v);

   float sum_d135 = accumulate(patch_vec_d135 , patch_vec_d135 + (BlockSizeR*BlockSizeC), 0.0);
   float MSE_psnr_d135 = sqrtf(sum_d135 / (float)(BlockSizeR*BlockSizeC));
   float psnr_d135 = 20.0f * log10f(255.0/MSE_psnr_d135);

   float sum_d45 = accumulate(patch_vec_d45 , patch_vec_d45 + (BlockSizeR*BlockSizeC), 0.0);
   float MSE_psnr_d45 = sqrtf(sum_d45 / (float)(BlockSizeR*BlockSizeC));
   float psnr_d45 = 20.0f * log10f(255.0/MSE_psnr_d45);

// cout<<psnr_s <<" "<<psnr_h<<" "<<psnr_v<<" "<<psnr_d135<<" "<<psnr_d45<<endl;

   std::vector<float> resp_val = {psnr_s,psnr_c,psnr_h, psnr_v, psnr_d45, psnr_d135};
   int maxElementIndex = std::max_element(resp_val.begin(),resp_val.end()) - resp_val.begin();
// cout<<maxElementIndex<<" ";
   switch(maxElementIndex){
    case 0:
    for(int x = row1; x<=row2 ; x++){
        for(int y = col1; y<=col2; y++){
            img_basic1[x+y*width] = img_basic_s[x+y*width];
          // cout<<angles[x+y*width]<<endl;
            img_basic_s[x+y*width] = img_basic_s[x+y*width];
        }
    }
   // patch_mse[row+col*width] = 180;
    break;
    case 1:
    for(int x = row1; x<=row2 ; x++){
        for(int y = col1; y<=col2; y++){
            img_basic2[x+y*width] = img_basic_c[x+y*width];
            img_basic_s[x+y*width] = img_basic_c[x+y*width];
        }
    }
    //patch_mse[row+col*width] = 0;

    break;
    case 2:
    for(int x = row1; x<=row2 ; x++){
        for(int y = col1; y<=col2; y++){
            img_basic3[x+y*width] = img_basic_h[x+y*width];
            img_basic_s[x+y*width]= img_basic_h[x+y*width];
        }
    }
    //patch_mse[row+col*width] = 90;

    break;
    case 3:
    for(int x = row1; x<=row2 ; x++){
        for(int y = col1; y<=col2; y++){
            img_basic4[x+y*width] = img_basic_v[x+y*width];
             img_basic_s[x+y*width] = img_basic_v[x+y*width];
        }
    }
    //patch_mse[row+col*width] = 45;
   
    break;
   case 4 :
    for(int x = row1; x<=row2 ; x++){
        for(int y = col1; y<=col2; y++){
            img_basic5[x+y*width] = img_basic_D45[x+y*width];
            img_basic_s[x+y*width] = img_basic_D45[x+y*width];
        }
    }
   // patch_mse[row+col*width] = 135;
     default:
    for(int x = row1; x<=row2 ; x++){
        for(int y = col1; y<=col2; y++){
            img_basic6[x+y*width] = img_basic_D135[x+y*width];
            img_basic_s[x+y*width] = img_basic_D135[x+y*width];
        }
    }
}
}
}
}

char *ImbasicSqu1 = (char*)malloc(25);
sprintf(ImbasicSqu1, "PatchSt1Squ%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImbasicSqu1, img_basic1, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImbasicSqu1);
char *ImbasicCir1 = (char*)malloc(25);
sprintf(ImbasicCir1, "PatchSt1Cir%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImbasicCir1, img_basic2, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImbasicCir1);
char *ImbasicHor1 = (char*)malloc(25);
sprintf(ImbasicHor1, "PatchSt1Hor%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImbasicHor1, img_basic3, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImbasicHor1);
char *ImbasicVer1 = (char*)malloc(25);
sprintf(ImbasicVer1, "PatchSt1Ver%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImbasicVer1, img_basic4, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImbasicVer1);
char *ImbasicD1351 = (char*)malloc(25);
sprintf(ImbasicD1351, "PatchSt1D135%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImbasicD1351, img_basic6, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImbasicD1351);
char *ImbasicD451 = (char*)malloc(25);
sprintf(ImbasicD451, "PatchSt1D45%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImbasicD451, img_basic5, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImbasicD45);

float psnrmse1, rmsemse1;
if(compute_psnr(img, img_basic_s, &psnrmse1, &rmsemse1) != EXIT_SUCCESS)
  cout<<"Error";

char *ImPatchesSt1 = (char*)malloc(25);
sprintf(ImPatchesSt1 , "ImPatchesSt1%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImPatchesSt1 , img_basic_s, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImPatchesSt1);

vector<float>final_img;
final_img.resize(img_noisy.size());

int filter_width = 3;
int filter_height = 3;
int offset = 3 / 2;
int count = 0;
//combine based on edges
vector<float> img_noisy2;
img_noisy2.resize(img_noisy.size());

//generate gaussian filter
// intialising standard deviation to 1.0 
double Gsigma = 1.0; 
double r, s = 2.0 * Gsigma * Gsigma; 
double GKernel[3][3];
// sum is for normalization 
double sum = 0.0; 

// generating 3x3 kernel 
for (int x = -1; x <= 1; x++) { 
    for (int y = -1; y <= 1; y++) { 
        r = sqrt(x * x + y * y); 
        GKernel[x + 1][y + 1] = (exp(-(r * r) / s)) / (M_PI * s); 
        sum += GKernel[x + 1][y + 1]; 
    } 
} 

// normalising the Kernel 
for (int i = 0; i < 3; ++i) //instead of 3 to 5
    for (int j = 0; j < 3; ++j) 
        GKernel[i][j] /= sum; 

//apply gaussian to remove noise
for (int x= 1 ; x < width  ; x++){
for (int y = 1 ; y < height ; y++){
    //generate the pixel neighbours
    //int loc = i + j*patch_width;
float img_total_noisy2 = 0.0;

for(int i = 0 ; i < filter_width ; i++){               
for (int j = 0 ; j < filter_height ; j++){
 int xloc = x + i - offset ;
 int yloc = y + j - offset ;
 int loc = xloc + (width*yloc);
// Calculate the convolution
// We sum all the neighboring pixels multiplied by the values in the convolution matrix.
img_total_noisy2 += img_noisy[loc] * GKernel[i][j];

}
}

int loc2 = x + y * width;
img_noisy2[loc2] = img_total_noisy2;  
//threshold Noisy image to conform 0-255 values
if(img_noisy2[loc2] < 0){
    img_noisy2[loc2] = 0;
} 
if(img_noisy2[loc2] > 255){
    img_noisy2[loc2] = 255;
}
}
} 
// if (save_image("gaussian.png", img_noisy2, width, height, chnls) != EXIT_SUCCESS){
// cout<<"Error";
// }

//Initialize edge response vectors
vector<float> img_hor_edge, img_ver_edge
, img_D135_edge, img_D45_edge, img_squ_edge;

//set the size of output edge 6 images
img_hor_edge.resize(img_noisy.size());
img_ver_edge.resize(img_noisy.size());
img_D135_edge.resize(img_noisy.size());
img_D45_edge.resize(img_noisy.size());
img_squ_edge.resize(img_noisy.size());


vector<float>angles;
angles.resize(img_noisy.size());

vector<float>angles2;
angles2.resize(img_noisy.size());

vector<float>gradient;
gradient.resize(img_noisy.size());


float BW_ver[3][3] = {{-1,0,1,},
                       {-2,0,2},
                       {-1,0,1}}; 

float BW_hor[3][3] = {{1,2,1},
                       {0,0,0},
                       {-1,-2,-1}};

float BW_D135[3][3] = {{0,1,2},
                       {-1,0,1},
                       {-2,-1,0}};

float BW_D45[3][3] = {{-2,-1,0},
                       {-1,0,1},
                       {0,1,2}};


for (int x = 1 ; x < width ; x++){
for (int y = 1 ; y < height ; y++){
//generate the pixel neighbours
//int loc = i + j*patch_width;
 float img_total_hor = 0.0;
 float img_total_ver = 0.0;
 float img_total_D135 = 0.0;
 float img_total_D45 = 0.0;
// float img_total_squ = 0.0;


 for(int i = 0 ; i < filter_width ; i++){               
    for (int j = 0 ; j < filter_height ; j++){
       int xloc = x + i - offset ;
       int yloc = y + j - offset ;
       int loc = xloc + width*yloc;
  // Calculate the convolution
 // We sum all the neighboring pixels multiplied by the values in the convolution matrix.
    img_total_hor += img_noisy2[loc] * BW_hor[i][j];
    img_total_ver += img_noisy2[loc] * BW_ver[i][j];
    img_total_D135 += img_noisy2[loc] * BW_D135[i][j];
    img_total_D45 += img_noisy2[loc] * BW_D45[i][j];


}
}

 int loc2 = x + y * width;
  
img_hor_edge[loc2] = img_total_hor/4.0;
img_ver_edge[loc2] = img_total_ver/4.0;
img_D135_edge[loc2] = img_total_D135/4.0;
img_D45_edge[loc2] = img_total_D45/4.0;

angles[loc2] = atan2(img_ver_edge[loc2],img_hor_edge[loc2]);
angles[loc2]= angles[loc2]*180/M_PI;

if(angles[loc2]<0){
angles[loc2] = 360 + angles[loc2];
}

angles2[loc2] = angles[loc2];

if ((angles[loc2] >= 0 ) && (angles[loc2] < 22.5) || (angles[loc2] >= 157.5) && (angles[loc2] < 202.5) || (angles[loc2] >= 337.5) && (angles[loc2] <= 360))
    angles[loc2] = 0;
else if ((angles[loc2] >= 22.5) && (angles[loc2] < 67.5) || (angles[loc2] >= 202.5) && (angles[loc2] < 247.5))
   angles[loc2] = 45;
else if ((angles[loc2] >= 67.5 && angles[loc2] < 112.5) || (angles[loc2] >= 247.5 && angles[loc2] < 292.5))
    angles[loc2] = 90;
else if ((angles[loc2] >= 112.5 && angles[loc2] < 157.5) || (angles[loc2] >= 292.5 && angles[loc2] < 337.5))
    angles[loc2] = 135;

//Calculate Gradient
//cout<<img_ver_edge[loc2]<<" "<<img_hor_edge[loc2]<<endl;
gradient[loc2] = sqrt(pow(img_ver_edge[loc2],2)+pow(img_hor_edge[loc2],2));

 }
}
std::vector<float> angles0;
angles0.resize(img_noisy.size());
std::vector<float> angles90;
angles90.resize(img_noisy.size());
std::vector<float> angles45;
angles45.resize(img_noisy.size());
std::vector<float> angles135;
angles135.resize(img_noisy.size());

for(int i = 0 ; i<width ; i++){
    for(int j = 0; j<height ; j++){
        if(angles[i+j*width] == 0)
            angles0[i+j*width] = img_basic_h[i+j*width];
        else if(angles[i+j*width] == 90)
            angles90[i+j*width] = img_basic_v[i+j*width];
        else if(angles[i+j*width] == 45)
            angles45[i+j*width] = img_basic_D45[i+j*width];
        else
            angles135[i+j*width] = img_basic_D135[i+j*width];
    }
}

// char *EdgeHor = (char*)malloc(25);
// sprintf(EdgeHor, "EdgeHor%d_%d.%s", (int)sigma,image_no,"png");
//    if (save_image(EdgeHor, angles0, width, height, chnls) != EXIT_SUCCESS){
//      cout<<"Error";
//  }
//  free(EdgeHor);
// char *Edgever = (char*)malloc(25);
// sprintf(Edgever, "EdgeVer%d_%d.%s", (int)sigma,image_no,"png");
//    if (save_image(Edgever, angles90, width, height, chnls) != EXIT_SUCCESS){
//      cout<<"Error";
//  }
//  free(Edgever);
// char *Edged45 = (char*)malloc(25);
// sprintf(Edged45, "EdgeD45%d_%d.%s", (int)sigma,image_no,"png");
//    if (save_image(Edged45, angles45, width, height, chnls) != EXIT_SUCCESS){
//      cout<<"Error";
//  }
//  free(Edged45);
// char *Edged135 = (char*)malloc(25);
// sprintf(Edged135, "EdgeD135%d_%d.%s", (int)sigma,image_no,"png");
//    if (save_image(Edged135, angles135, width, height, chnls) != EXIT_SUCCESS){
//      cout<<"Error";
//  }
// free(Edged135);

// char *EdgeHor = (char*)malloc(25);
// sprintf(EdgeHor, "EdgeHor%d.%s", (int)sigma,"png");
//    if (save_image(EdgeHor, img_hor_edge, width, height, chnls) != EXIT_SUCCESS){
//      cout<<"Error";
//  }
//  char *EdgeVer = (char*)malloc(25);
//  sprintf(EdgeVer, "EdgeVer%d.%s", (int)sigma,"png");
//    if (save_image(EdgeVer, img_ver_edge, width, height, chnls) != EXIT_SUCCESS){
//      cout<<"Error";
//  }
//  char *EdgeD135 = (char*)malloc(25);
//  sprintf(EdgeD135, "EdgeD135%d.%s", (int)sigma,"png");
//    if (save_image(EdgeD135, img_D135_edge, width, height, chnls) != EXIT_SUCCESS){
//      cout<<"Error";
//  }
//  char *EdgeD45 = (char*)malloc(25);
//  sprintf(EdgeD45, "EdgeD45%d.%s", (int)sigma,"png");
//    if (save_image(EdgeD45, img_D45_edge, width, height, chnls) != EXIT_SUCCESS){
//      cout<<"Error";
//  }
 char *AnglesImage = (char*)malloc(25);
 sprintf(AnglesImage, "ImgAngle%d_%d.%s", (int)sigma,image_no,"png");
   if (save_image(AnglesImage, angles, width, height, chnls) != EXIT_SUCCESS){
     cout<<"Error";
 }



char *ImOptimalSqu = (char*)malloc(25);
sprintf(ImOptimalSqu, "BestLocSquSt1%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImOptimalSqu, OptimalSqu, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImOptimalSqu);
char *ImOptimalCir = (char*)malloc(25);
sprintf(ImOptimalCir, "BestLocCirSt1%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImOptimalCir, OptimalCir, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImOptimalCir);
char *ImOptimalHor = (char*)malloc(25);
sprintf(ImOptimalHor, "BestLocHorSt1%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImOptimalHor, OptimalHor, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImOptimalHor);
char *ImOptimalVer = (char*)malloc(25);
sprintf(ImOptimalVer, "BestLocVerSt1%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImOptimalVer, OptimalVer, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImOptimalVer);
char *ImOptimalD135 = (char*)malloc(25);
sprintf(ImOptimalD135, "BestLocD135St1%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImOptimalD135, OptimalD135, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImOptimalD135);
char *ImOptimalD45 = (char*)malloc(25);
sprintf(ImOptimalD45, "BestLocD45St1%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImOptimalD45, OptimalD45, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImOptimalD45);

// cout<<"The optimal solution should be "<<endl;
// cout<<"number of square pixels = "<<count_s<<endl;
// cout<<"number of circle pixels = "<<count_c<<endl;
// cout<<"number of hor pixels = "<<count_h<<endl;
// cout<<"number of ver pixels = "<<count_v<<endl;
// cout<<"number of D135 pixels = "<<count_D135<<endl;
// cout<<"number of D45 pixels = "<<count_D45<<endl;


//! Allocating Plan for FFTW process
if (tau_2D_wien == DCT)
{
    const unsigned nb_cols = ind_size(w_b - kWien + 1, nWien, pWien);
    allocate_plan_2d(&plan_2d_for_1[0], kWien, FFTW_REDFT10,
        w_b * (2 * nWien + 1) * chnls);
    allocate_plan_2d(&plan_2d_for_2[0], kWien, FFTW_REDFT10,
        w_b * pWien * chnls);
    allocate_plan_2d(&plan_2d_inv  [0], kWien, FFTW_REDFT01,
        NWien * nb_cols * chnls);
}

//Original BM3d stage 2
//! Denoising, 2nd Step
cout << "step 2...";
bm3d_2nd_step(sigma, img_sym_noisy, img_sym_basic, img_sym_denoised,
    img_sym_basic_s,img_sym_basic_c,img_sym_basic_h,img_sym_basic_v,img_sym_basic_D135,img_sym_basic_D45,
    w_b, h_b, chnls, nWien, kWien, NWien, pWien, useSD_w, color_space,
    tau_2D_wien, &plan_2d_for_1[0], &plan_2d_for_2[0], &plan_2d_inv[0]);
cout << "done." << endl;

//! Obtention of img_denoised
for (unsigned c = 0; c < chnls; c++)
{
const unsigned dc_b = c * w_b * h_b + nWien * w_b + nWien;
unsigned dc = c * width * height;
for (unsigned i = 0; i < height; i++){
    for (unsigned j = 0; j < width; j++, dc++){
        img_denoised[dc] = img_sym_denoised[dc_b + i * w_b + j];
    }
    }
}
//store stage one original solution
char *ImDenoised = (char*)malloc(25);
sprintf(ImDenoised, "ImOriginalSt2%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImDenoised, img_denoised, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
//calculate psnr
  float psnr_stg2, rmse_stg2;
if(compute_psnr(img, img_denoised, &psnr_stg2, &rmse_stg2) != EXIT_SUCCESS)
cout<<"Error calculating psnr";

//store image
char *Imnse= (char*)malloc(20);
sprintf(Imnse, "ImOriginal%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(Imnse, img_denoised, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}

//use the optimal result as an input
cout<<"optimal solution stage 2 "<< endl;
for(int u = 0; u<width; u++){
    for(int v = 0; v<height; v++){
        img_basic[u+v*width] = OptimalVal[u+v*width];
    }
}
 symetrize(img_basic, img_sym_basic, width, height, chnls, nHard);
//! Denoising, 2nd Step
cout << "step 2...";
bm3d_2nd_step(sigma, img_sym_noisy, img_sym_basic, img_sym_denoised,
    img_sym_basic_s,img_sym_basic_c,img_sym_basic_h,img_sym_basic_v,img_sym_basic_D135,img_sym_basic_D45,
    w_b, h_b, chnls, nWien, kWien, NWien, pWien, useSD_w, color_space,
    tau_2D_wien, &plan_2d_for_1[0], &plan_2d_for_2[0], &plan_2d_inv[0]);
cout << "done." << endl;

//! Obtention of img_denoised
for (unsigned c = 0; c < chnls; c++)
{
    const unsigned dc_b = c * w_b * h_b + nWien * w_b + nWien;
    unsigned dc = c * width * height;
    for (unsigned i = 0; i < height; i++){
        for (unsigned j = 0; j < width; j++, dc++){
            img_denoised[dc] = img_sym_denoised[dc_b + i * w_b + j];
            img_denoised_s[dc] = img_sym_basic_s[dc_b + i * w_b + j];
            img_denoised_c[dc] = img_sym_basic_c[dc_b + i * w_b + j];
            img_denoised_hor[dc] = img_sym_basic_h[dc_b + i * w_b + j];
            img_denoised_ver[dc] = img_sym_basic_v[dc_b + i * w_b + j];
            img_denoised_D135[dc] = img_sym_basic_D135[dc_b + i * w_b + j];
            img_denoised_D45[dc] = img_sym_basic_D45[dc_b + i * w_b + j];
        }
        }
    }

//calculate psnr
    float psnr_s2, rmse_s2, psnr_c2, rmse_c2, psnr_h2, rmse_h2, psnr_denoised, rmse_denoised;
    float psnr_v2, rmse_v2, psnr2_D135, rmse2_D135, psnr2_D45, rmse2_D45;

if(compute_psnr(img, img_denoised, &psnr_denoised, &rmse_denoised) != EXIT_SUCCESS)
   cout<<"Error calculating psnr";
if(compute_psnr(img, img_denoised_s, &psnr_s2, &rmse_s2) != EXIT_SUCCESS)
    cout<<"Error calculating psnr";
if(compute_psnr(img, img_denoised_c, &psnr_c2, &rmse_c2) != EXIT_SUCCESS)
    cout<<"Error calculating psnr";
if(compute_psnr(img, img_denoised_hor, &psnr_h2, &rmse_h2) != EXIT_SUCCESS)
   cout<<"Error calculating psnr";
if(compute_psnr(img, img_denoised_ver, &psnr_v2, &rmse_v2) != EXIT_SUCCESS)
   cout<<"Error calculating psnr";
if(compute_psnr(img, img_denoised_D135, &psnr2_D135, &rmse2_D135) != EXIT_SUCCESS)
   cout<<"Error calculating psnr";
if(compute_psnr(img, img_denoised_D45, &psnr2_D45, &rmse2_D45) != EXIT_SUCCESS)
   cout<<"Error calculating psnr";


cout << endl << "For stage 2 image :" << endl;
cout << "(denoised image) :" << endl;
cout << "PSNR denoised: " << psnr_denoised << endl;
cout << "RMSE denoised: " << rmse_denoised << endl << endl;
cout << "(squ image) :" << endl;
cout << "PSNR squ: " << psnr_s2 << endl;
cout << "RMSE squ: " << rmse_s2 << endl << endl;
cout << "(cir image) :" << endl;
cout << "PSNR cir: " << psnr_c2 << endl;
cout << "RMSE cir: " << rmse_c2 << endl << endl;
cout << "(hor image) :" << endl;
cout << "PSNR hor: " << psnr_h2 << endl;
cout << "RMSE hor: " << rmse_h2 << endl << endl;
cout << "(ver image) :" << endl;
cout << "PSNR ver: " << psnr_v2 << endl;
cout << "RMSE ver: " << rmse_v2 << endl << endl;
cout << "(D135 image) :" << endl;
cout << "PSNR D135: " << psnr2_D135 << endl;
cout << "RMSE D135: " << rmse2_D135 << endl << endl;
cout << "(D45 image) :" << endl;
cout << "PSNR D45: " << psnr2_D45 << endl;
cout << "RMSE D45: " << rmse2_D45 << endl << endl;


char *ImDenoisedSqu2 = (char*)malloc(20);
sprintf(ImDenoisedSqu2, "ImgSt2Squ%d.%s", (int)sigma,"png");
if (save_image(ImDenoisedSqu2, img_denoised_s, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
char *ImDenoisedCir2 = (char*)malloc(20);
sprintf(ImDenoisedCir2, "ImgSt2Cir%d.%s", (int)sigma,"png");
if (save_image(ImDenoisedCir2, img_denoised_c, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
char *ImDenoisedHor2 = (char*)malloc(20);
sprintf(ImDenoisedHor2, "ImgSt2Hor%d.%s", (int)sigma,"png");
if (save_image(ImDenoisedHor2, img_denoised_hor, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
char *ImDenoisedVer2 = (char*)malloc(20);
sprintf(ImDenoisedVer2, "ImgSt2Ver%d.%s", (int)sigma,"png");
if (save_image(ImDenoisedVer2, img_denoised_ver, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
char *ImDenoisedD1352 = (char*)malloc(20);
sprintf(ImDenoisedD1352, "ImgSt2D135%d.%s", (int)sigma,"png");
if (save_image(ImDenoisedD1352, img_denoised_D135, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
char *ImDenoisedD452 = (char*)malloc(20);
sprintf(ImDenoisedD452, "ImgSt2D45%d.%s", (int)sigma,"png");
if (save_image(ImDenoisedD452, img_denoised_D45, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}




std::vector<float> OptimalVal2;
OptimalVal2.resize(img_noisy.size());
std::vector<float> OptimalSqu2;
OptimalSqu2.resize(img_noisy.size());
std::vector<float> OptimalCir2;
OptimalCir2.resize(img_noisy.size());
std::vector<float> OptimalHor2;
OptimalHor2.resize(img_noisy.size());
std::vector<float> OptimalVer2;
OptimalVer2.resize(img_noisy.size());
std::vector<float> OptimalD135_2;
OptimalD135_2.resize(img_noisy.size());
std::vector<float> OptimalD45_2;
OptimalD45_2.resize(img_noisy.size());

int count_s2 = 0;
int count_c2 = 0;
int count_h2 = 0;
int count_v2 = 0;
int count_D135_2 = 0;
int count_D45_2 = 0;

//calculate the optimal solution
for(int m = 0; m < width;m++){
    for(int n =0; n<height; n++){
       float ress= abs(img[m+n*width] - img_denoised_s[m+n*width]);
       float resc= abs(img[m+n*width] - img_denoised_c[m+n*width]);
       float resh = abs(img[m+n*width] - img_denoised_hor[m+n*width]);
       float resv = abs(img[m+n*width] - img_denoised_ver[m+n*width]);
        float res135 = abs(img[m+n*width] - img_denoised_D135[m+n*width]);
        float res45 = abs(img[m+n*width] - img_denoised_D45[m+n*width]);

        std::vector<float> resp_val = {ress,resc,resh,resv,res135,res45};

        int miniInd = std::min_element(resp_val.begin(),resp_val.end()) - resp_val.begin();
        switch(miniInd){
            case 0:
            OptimalVal2[m+n*width] = img_denoised_s[m+n*width];
            OptimalSqu2[m+n*width] = img_denoised_s[m+n*width];
            count_s2++;
            break;
            case 1:
            OptimalVal2[m+n*width] = img_denoised_c[m+n*width];
            OptimalCir2[m+n*width] = img_denoised_c[m+n*width];
            count_c2++;
            break;
            case 2:
            OptimalVal2[m+n*width] = img_denoised_hor[m+n*width];
            OptimalHor2[m+n*width] = img_denoised_hor[m+n*width];
            count_h++;
            break;
            case 3:
            OptimalVal2[m+n*width] = img_denoised_ver[m+n*width];
            OptimalVer2[m+n*width] = img_denoised_ver[m+n*width];
            count_v2++;
            break;
            case 4:
            OptimalVal2[m+n*width] = img_denoised_D135[m+n*width];
            OptimalD135_2[m+n*width] = img_denoised_D135[m+n*width];
            count_D135_2++;
            break;
            default:
            OptimalVal2[m+n*width] = img_denoised_D45[m+n*width];
            OptimalD45_2[m+n*width] = img_denoised_D45[m+n*width];
            count_D45_2++;

        }

    }
}

char *ImOptimalSqu2 = (char*)malloc(25);
sprintf(ImOptimalSqu2, "BestLocSquSt2%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImOptimalSqu2, OptimalSqu2, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImOptimalSqu2);

char *ImOptimalCir2 = (char*)malloc(25);
sprintf(ImOptimalCir2, "BestLocCirSt2%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImOptimalCir2, OptimalCir2, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImOptimalCir2);

char *ImOptimalHor2 = (char*)malloc(25);
sprintf(ImOptimalHor2, "BestLocHorSt2%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImOptimalHor2, OptimalHor2, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImOptimalHor2);
char *ImOptimalVer2 = (char*)malloc(25);
sprintf(ImOptimalVer2, "BestLocVerSt2%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImOptimalVer2, OptimalVer2, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImOptimalVer2);
char *ImOptimalD135_2 = (char*)malloc(25);
sprintf(ImOptimalD135_2, "BestLocD135St2%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImOptimalD135_2, OptimalD135_2, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImOptimalD135_2);
char *ImOptimalD45_2 = (char*)malloc(25);
sprintf(ImOptimalD45_2, "BestLocD45St2%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImOptimalD45_2, OptimalD45_2, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImOptimalD45_2);

//psnr
float psnrOpt2, rmseOpt2;
if(compute_psnr(img, OptimalVal2, &psnrOpt2, &rmseOpt2) != EXIT_SUCCESS)
  cout<<"Error";


//store images
char *ImOptimalSt2 = (char*)malloc(25);
sprintf(ImOptimalSt2 , "ImImprovementSt2%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImOptimalSt2 , OptimalVal2, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImOptimalSt2);
char path[20] = "Data.csv";
ofstream filee(path, ios::out | ios::app);
if(filee)
{
filee<<"Noisy_PSNR "<<"Original_stage1 "<<"Squ_PSNR "<<"Cir_PSNR "<<"Hor_PSNR "<<"Ver_PSNR "<<"D135_PSNR "<<"D45_PSNR "<<"Optimal_Stage1 ";
filee<<"Original_stage2 "<<"squ_PSNR "<<"Cir_PSNR "<<"Hor_PSNR "<<"Ver_PSNR "<<"D135_PSNR "<<"D45_PSNR "<<"Optimal_Stage2"<<endl;
filee<<psnrNSE<<" "<<psnr_basic<<" "<<psnr_s<<" "<<psnr_c<<" "<<psnr_h<<" "<<psnr_v<<" "<<psnr_D135<<" "<<psnr_D45<<" "<<psnrOpt<<" ";
filee<<psnr_stg2<<" "<<psnr_s2<<" "<<psnr_c2<<" "<<psnr_h2<<" "<<psnr_v2<<" "<<psnr2_D135<<" "<<psnr2_D45<<" "<<psnrOpt2<<endl;
}
filee.close();

// char path2[20] = "Count.csv";
// ofstream file2(path2, ios::out | ios::app);
// if(file2)
// {
// file2<<"CountSqu1 "<<"CountHor1 "<<"CountVer1 "<<"CountD135_1 "<<"CountD45_1 ";;
// file2<<"CountSqu2 "<<"CountHor2 "<<"CountVer2 "<<"CountD135_2 "<<"CountD45_2 "<<endl;
// file2<<count_s<<" "<<count_h<<" "<<count_v<<" "<<" "<<count_D135<<" "<<count_D45<<" ";
// file2<<count_s2<<" "<<count_h2<<" "<<count_v2<<" "<<" "<<count_D135_2<<" "<<count_D45_2<<endl;
// }
// file2.close();
cout<<"donexx"<<endl;

for(int u = 0; u<width; u++){
    for(int v = 0; v<height; v++){
        img_basic[u+v*width] = img_basic_s[u+v*width];
    }
}

 symetrize(img_basic, img_sym_basic, width, height, chnls, nHard);
//! Denoising, 2nd Step
cout << "step 2...";
bm3d_2nd_step(sigma, img_sym_noisy, img_sym_basic, img_sym_denoised,
    img_sym_basic_s,img_sym_basic_c,img_sym_basic_h,img_sym_basic_v,img_sym_basic_D135,img_sym_basic_D45,
    w_b, h_b, chnls, nWien, kWien, NWien, pWien, useSD_w, color_space,
    tau_2D_wien, &plan_2d_for_1[0], &plan_2d_for_2[0], &plan_2d_inv[0]);
cout << "done." << endl;

//! Obtention of img_denoised
for (unsigned c = 0; c < chnls; c++)
{
    const unsigned dc_b = c * w_b * h_b + nWien * w_b + nWien;
    unsigned dc = c * width * height;
    for (unsigned i = 0; i < height; i++){
        for (unsigned j = 0; j < width; j++, dc++){
            img_denoised[dc] = img_sym_denoised[dc_b + i * w_b + j];
            img_denoised_s[dc] = img_sym_basic_s[dc_b + i * w_b + j];
            img_denoised_c[dc] = img_sym_basic_c[dc_b + i * w_b + j];
            img_denoised_hor[dc] = img_sym_basic_h[dc_b + i * w_b + j];
            img_denoised_ver[dc] = img_sym_basic_v[dc_b + i * w_b + j];
            img_denoised_D135[dc] = img_sym_basic_D135[dc_b + i * w_b + j];
            img_denoised_D45[dc] = img_sym_basic_D45[dc_b + i * w_b + j];
        }
        }
    }

vector<float>img_denoised1;
img_denoised1.resize(img_noisy.size());
vector<float>img_denoised2;
img_denoised2.resize(img_noisy.size());
vector<float>img_denoised3;
img_denoised3.resize(img_noisy.size());
vector<float>img_denoised4;
img_denoised4.resize(img_noisy.size());
vector<float>img_denoised5;
img_denoised5.resize(img_noisy.size());
vector<float>img_denoised6;
img_denoised6.resize(img_noisy.size());

// int BlockSizeR = 3;
// int BlockSizeC = 3;
// int n = BlockSizeR*BlockSizeC;
//mse

for(int row= 0; row < width ;  row += BlockSizeR){
    for(int col = 0 ; col < height ; col += BlockSizeC){

unsigned  row1 = row;
unsigned row2 = row1 + BlockSizeR - 1;
row2 = min(width,row2);// to prevent going outside image borders

unsigned col1 = col;
unsigned col2 = col1 + BlockSizeC - 1;
col2 = min(height,col2);

float patch_vec_s[BlockSizeR*BlockSizeC] = {};
float patch_vec_c[BlockSizeR*BlockSizeC] = {};
float patch_vec_h[BlockSizeR*BlockSizeC] = {};
float patch_vec_v[BlockSizeR*BlockSizeC] = {};
float patch_vec_d135[BlockSizeR*BlockSizeC] = {};
float patch_vec_d45[BlockSizeR*BlockSizeC] = {};
int pixel_no = 0;
//verify that the block have the correct size
if((row2 - row1 + 1)== BlockSizeR && (col2 - col1 + 1) == BlockSizeC){
    for(int x = row1; x<=row2 ; x++){
        for(int y = col1; y<=col2; y++){
           patch_vec_s [pixel_no] =pow(img_denoised_s[x+y*width]-img[x+y*width],2);
           patch_vec_c [pixel_no] =pow(img_denoised_c[x+y*width]-img[x+y*width],2);
           patch_vec_h [pixel_no] = pow(img_denoised_hor[x+y*width]-img[x+y*width],2);
           patch_vec_v [pixel_no] = pow(img_denoised_ver[x+y*width]-img[x+y*width],2);
           patch_vec_d135 [pixel_no] = pow(img_denoised_D135[x+y*width]-img[x+y*width],2);
           patch_vec_d45 [pixel_no] = pow(img_denoised_D45[x+y*width]-img[x+y*width],2);
           pixel_no ++;

       }
   //  cout<<endl;
   }
  // cout<<endl<<endl;
    //calculate the mse of each patch and compare
   float sum_s = accumulate(patch_vec_s , patch_vec_s + (BlockSizeR*BlockSizeC), 0.0);
   float MSE_psnr_s = sqrtf(sum_s / (float)(BlockSizeR*BlockSizeC));
   float psnr_s = 20.0f * log10f(255.0/MSE_psnr_s);

    float sum_c = accumulate(patch_vec_c , patch_vec_c + (BlockSizeR*BlockSizeC), 0.0);
   float MSE_psnr_c = sqrtf(sum_c / (float)(BlockSizeR*BlockSizeC));
   float psnr_c = 20.0f * log10f(255.0/MSE_psnr_c);
//cout<<sum_s<<endl;
   float sum_h = accumulate(patch_vec_h , patch_vec_h + (BlockSizeR*BlockSizeC), 0.0);
   float MSE_psnr_h = sqrtf(sum_h / (float)(BlockSizeR*BlockSizeC));
   float psnr_h = 20.0f * log10f(255.0/MSE_psnr_h);

   float sum_v = accumulate(patch_vec_v , patch_vec_v + (BlockSizeR*BlockSizeC), 0.0);
   float MSE_psnr_v = sqrtf(sum_v / (float)(BlockSizeR*BlockSizeC));
   float psnr_v = 20.0f * log10f(255.0/MSE_psnr_v);

   float sum_d135 = accumulate(patch_vec_d135 , patch_vec_d135 + (BlockSizeR*BlockSizeC), 0.0);
   float MSE_psnr_d135 = sqrtf(sum_d135 / (float)(BlockSizeR*BlockSizeC));
   float psnr_d135 = 20.0f * log10f(255.0/MSE_psnr_d135);

   float sum_d45 = accumulate(patch_vec_d45 , patch_vec_d45 + (BlockSizeR*BlockSizeC), 0.0);
   float MSE_psnr_d45 = sqrtf(sum_d45 / (float)(BlockSizeR*BlockSizeC));
   float psnr_d45 = 20.0f * log10f(255.0/MSE_psnr_d45);

// cout<<psnr_s <<" "<<psnr_h<<" "<<psnr_v<<" "<<psnr_d135<<" "<<psnr_d45<<endl;

   std::vector<float> resp_val = {psnr_s,psnr_c,psnr_h, psnr_v, psnr_d45, psnr_d135};
   int maxElementIndex = std::max_element(resp_val.begin(),resp_val.end()) - resp_val.begin();
// cout<<maxElementIndex<<" ";
   switch(maxElementIndex){
    case 0:
    for(int x = row1; x<=row2 ; x++){
        for(int y = col1; y<=col2; y++){
            img_denoised1[x+y*width] = img_denoised_s[x+y*width];
          // cout<<angles[x+y*width]<<endl;
            img_denoised_s[x+y*width] = img_denoised_s[x+y*width];
        }
    }
   // patch_mse[row+col*width] = 180;
    break;
    case 1:
    for(int x = row1; x<=row2 ; x++){
        for(int y = col1; y<=col2; y++){
            img_denoised2[x+y*width] = img_denoised_c[x+y*width];
            img_denoised_s[x+y*width] = img_denoised_c[x+y*width];
        }
    }
    //patch_mse[row+col*width] = 0;

    break;
    case 2:
    for(int x = row1; x<=row2 ; x++){
        for(int y = col1; y<=col2; y++){
            img_denoised3[x+y*width] = img_denoised_hor[x+y*width];
            img_denoised_s[x+y*width]= img_denoised_hor[x+y*width];
        }
    }
    //patch_mse[row+col*width] = 90;

    break;
    case 3:
    for(int x = row1; x<=row2 ; x++){
        for(int y = col1; y<=col2; y++){
            img_denoised4[x+y*width] = img_denoised_ver[x+y*width];
             img_denoised_s[x+y*width] = img_denoised_ver[x+y*width];
        }
    }
    //patch_mse[row+col*width] = 45;
   
    break;
   case 4 :
    for(int x = row1; x<=row2 ; x++){
        for(int y = col1; y<=col2; y++){
            img_denoised5[x+y*width] = img_denoised_D45[x+y*width];
            img_denoised_s[x+y*width] = img_denoised_D45[x+y*width];
        }
    }
   // patch_mse[row+col*width] = 135;
     default:
    for(int x = row1; x<=row2 ; x++){
        for(int y = col1; y<=col2; y++){
            img_denoised6[x+y*width] = img_denoised_D135[x+y*width];
            img_denoised_s[x+y*width] = img_denoised_D135[x+y*width];
        }
    }
}
}
}
}
float psnrmse2, rmsemse2;
if(compute_psnr(img, img_denoised_s, &psnrmse2, &rmsemse2) != EXIT_SUCCESS)
  cout<<"Error";

char *ImPatchesSt2 = (char*)malloc(25);
sprintf(ImPatchesSt2 , "ImPatchesSt2%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImPatchesSt2 , img_denoised_s, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImPatchesSt2);
char pathx[20] = "MSE.csv";
ofstream filex(pathx, ios::out | ios::app);
if(filex)
{
filex<<"mse1 "<<"mse2 "<<endl;
filex<<psnrmse1<<" "<<psnrmse2<<endl;
}
filex.close();

cout<<"afetrfilreee" <<endl;

char *ImdenSqu = (char*)malloc(25);
sprintf(ImdenSqu, "PatchSt2Squ%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImdenSqu, img_denoised1, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImdenSqu);
char *ImdenCir = (char*)malloc(25);
sprintf(ImdenCir, "PatchSt2Cir%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImdenCir, img_denoised2, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImdenCir);
char *ImdenHor = (char*)malloc(25);
sprintf(ImdenHor, "PatchSt2Hor%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImdenHor, img_denoised3, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImdenHor);
char *ImdenVer = (char*)malloc(25);
sprintf(ImdenVer, "PatchSt2Ver%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImdenVer, img_denoised4, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImdenVer);
char *ImdenD135 = (char*)malloc(25);
sprintf(ImdenD135, "PatchSt2D135%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImdenD135, img_denoised6, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImdenD135);
char *ImdenD45 = (char*)malloc(25);
sprintf(ImdenD45, "PatchSt2D45%d_%d.%s", (int)sigma,image_no,"png");
if (save_image(ImdenD45, img_denoised5, width, height, chnls) != EXIT_SUCCESS){
 cout<<"Error";
}
free(ImdenD45);
}
    //! If more than 1 threads are used
else
{
        //! Cut the image in nb_threads parts
    vector<vector<float> > sub_noisy(nb_threads);
    vector<vector<float> > sub_basic(nb_threads);
    vector<vector<float> > sub_basic_s(nb_threads);
    vector<vector<float> > sub_basic_c(nb_threads);
    vector<vector<float> > sub_basic_h(nb_threads);
    vector<vector<float> > sub_basic_v(nb_threads);
    vector<vector<float> > sub_basic_D135(nb_threads);
    vector<vector<float> > sub_basic_D45(nb_threads);

    vector<vector<float> > sub_denoised(nb_threads);
    vector<vector<float> > sub_denoised_s(nb_threads);
     vector<vector<float> > sub_denoised_c(nb_threads);
    vector<vector<float> > sub_denoised_hor(nb_threads);
    vector<vector<float> > sub_denoised_ver(nb_threads);
    vector<vector<float> > sub_denoised_D135(nb_threads);
    vector<vector<float> > sub_denoised_D45(nb_threads);

    vector<vector<float> > sub_img(nb_threads);
    vector<unsigned> h_table(nb_threads);
    vector<unsigned> w_table(nb_threads);
    sub_divide(img_noisy, sub_noisy, w_table, h_table, width, height, chnls,
        2 * nWien, true);

        //! Allocating Plan for FFTW process
    if (tau_2D_hard == DCT)
        for (unsigned n = 0; n < nb_threads; n++)
        {
            const unsigned nb_cols = ind_size(w_table[n] - kHard + 1, nHard, pHard);
            allocate_plan_2d(&plan_2d_for_1[n], kHard, FFTW_REDFT10,
                w_table[n] * (2 * nHard + 1) * chnls);
            allocate_plan_2d(&plan_2d_for_2[n], kHard, FFTW_REDFT10,
                w_table[n] * pHard * chnls);
            allocate_plan_2d(&plan_2d_inv  [n], kHard, FFTW_REDFT01,
                NHard * nb_cols * chnls);
        }

//! denoising : 1st Step
cout << "step 1...";
#pragma omp parallel shared(sub_noisy, sub_basic, w_table, h_table, \
plan_2d_for_1, plan_2d_for_2, plan_2d_inv)
#pragma omp parallel shared(sub_noisy, sub_basic_s, w_table, h_table, \
plan_2d_for_1, plan_2d_for_2, plan_2d_inv)
#pragma omp parallel shared(sub_noisy, sub_basic_c, w_table, h_table, \
plan_2d_for_1, plan_2d_for_2, plan_2d_inv)
#pragma omp parallel shared(sub_noisy, sub_basic_h, w_table, h_table, \
plan_2d_for_1, plan_2d_for_2, plan_2d_inv)
#pragma omp parallel shared(sub_noisy, sub_basic_v, w_table, h_table, \
plan_2d_for_1, plan_2d_for_2, plan_2d_inv)
#pragma omp parallel shared(sub_noisy, sub_basic_D135, w_table, h_table, \
plan_2d_for_1, plan_2d_for_2, plan_2d_inv)
#pragma omp parallel shared(sub_noisy, sub_basic_D45, w_table, h_table, \
plan_2d_for_1, plan_2d_for_2, plan_2d_inv)
{
    #pragma omp for schedule(dynamic) nowait
    for (unsigned n = 0; n < nb_threads; n++)
    {
        bm3d_1st_step(sigma, sub_noisy[n], sub_basic[n],
            sub_basic_s[n], sub_basic_c[n], sub_basic_h[n],sub_basic_v[n],sub_basic_D135[n],
            sub_basic_D45[n],w_table[n],
            h_table[n], chnls, nHard, kHard, NHard, pHard, useSD_h,
            color_space, tau_2D_hard, &plan_2d_for_1[n],
            &plan_2d_for_2[n], &plan_2d_inv[n]);
    }
}
cout << "done." << endl;

sub_divide(img_basic, sub_basic, w_table, h_table,
    width, height, chnls, 2 * nHard, false);

sub_divide(img_basic, sub_basic, w_table, h_table, width, height, chnls,
    2 * nHard, true);

//! Allocating Plan for FFTW process
if (tau_2D_wien == DCT)
    for (unsigned n = 0; n < nb_threads; n++)
    {
        const unsigned nb_cols = ind_size(w_table[n] - kWien + 1, nWien, pWien);
        allocate_plan_2d(&plan_2d_for_1[n], kWien, FFTW_REDFT10,
            w_table[n] * (2 * nWien + 1) * chnls);
        allocate_plan_2d(&plan_2d_for_2[n], kWien, FFTW_REDFT10,
            w_table[n] * pWien * chnls);
        allocate_plan_2d(&plan_2d_inv  [n], kWien, FFTW_REDFT01,
            NWien * nb_cols * chnls);
    }

//! Denoising: 2nd Step
    cout << "step 2...";
#pragma omp parallel shared(sub_noisy, sub_basic, sub_denoised,  w_table, \
    h_table, plan_2d_for_1, plan_2d_for_2,  \
    plan_2d_inv)
#pragma omp parallel shared(sub_noisy, sub_denoised_s, w_table, h_table, \
plan_2d_for_1, plan_2d_for_2, plan_2d_inv)
#pragma omp parallel shared(sub_noisy, sub_denoised_c, w_table, h_table, \
plan_2d_for_1, plan_2d_for_2, plan_2d_inv)
#pragma omp parallel shared(sub_noisy, sub_denoised_hor, w_table, h_table, \
plan_2d_for_1, plan_2d_for_2, plan_2d_inv)
#pragma omp parallel shared(sub_noisy, sub_denoised_ver, w_table, h_table, \
plan_2d_for_1, plan_2d_for_2, plan_2d_inv)
#pragma omp parallel shared(sub_noisy, sub_denoised_D135, w_table, h_table, \
plan_2d_for_1, plan_2d_for_2, plan_2d_inv)
#pragma omp parallel shared(sub_noisy, sub_denoised_D45, w_table, h_table, \
plan_2d_for_1, plan_2d_for_2, plan_2d_inv)

{
    #pragma omp for schedule(dynamic) nowait
    for (unsigned n = 0; n < nb_threads; n++)
    {
        bm3d_2nd_step(sigma, sub_noisy[n], sub_basic[n], sub_denoised[n],sub_denoised_s[n],sub_denoised_c[n],
          sub_denoised_hor[n],sub_denoised_ver[n],sub_denoised_D135[n],sub_denoised_D45[n],
          w_table[n], h_table[n], chnls, nWien, kWien, NWien, pWien,
          useSD_w, color_space, tau_2D_wien, &plan_2d_for_1[n],
          &plan_2d_for_2[n], &plan_2d_inv[n]);
    }
}
cout << "done." << endl;

//! Reconstruction of the image
sub_divide(img_denoised, sub_denoised, w_table, h_table,
    width, height, chnls, 2 * nWien, false);
}

//! Inverse color space transform to RGB
if (color_space_transform(img_denoised, color_space, width, height, chnls, false)
!= EXIT_SUCCESS) return EXIT_FAILURE;
if (color_space_transform(img_noisy, color_space, width, height, chnls, false)
!= EXIT_SUCCESS) return EXIT_FAILURE;
if (color_space_transform(img_basic, color_space, width, height, chnls, false)
!= EXIT_SUCCESS) return EXIT_FAILURE;

//! Free Memory
if (tau_2D_hard == DCT || tau_2D_wien == DCT)
    for (unsigned n = 0; n < nb_threads; n++)
    {
        fftwf_destroy_plan(plan_2d_for_1[n]);
        fftwf_destroy_plan(plan_2d_for_2[n]);
        fftwf_destroy_plan(plan_2d_inv[n]);
    }
    fftwf_cleanup();

    return EXIT_SUCCESS;
}

/**
 * @brief Run the basic process of BM3D (1st step). The result
 *        is contained in img_basic. The image has boundary, which
 *        are here only for block-matching and doesn't need to be
 *        denoised.
 *
 * @param sigma: value of assumed noise of the image to denoise;
 * @param img_noisy: noisy image;
 * @param img_basic: will contain the denoised image after the 1st step;
 * @param width, height, chnls : size of img_noisy;
 * @param nHard: size of the boundary around img_noisy;
 * @param useSD: if true, use weight based on the standard variation
 *        of the 3D group for the first step, otherwise use the number
 *        of non-zero coefficients after Hard-thresholding;
 * @param tau_2D: DCT or BIOR;
 * @param plan_2d_for_1, plan_2d_for_2, plan_2d_inv : for convenience. Used
 *        by fftw.
 *
 * @return none.
**/
void bm3d_1st_step(
const float sigma
,   vector<float> const& img_noisy
,   vector<float> &img_basic
,   vector<float> &img_basic_s
,   vector<float> &img_basic_c
,   vector<float> &img_basic_h
,   vector<float> &img_basic_v
,   vector<float> &img_basic_D135
,   vector<float> &img_basic_D45
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned nHard
,   const unsigned kHard
,   const unsigned NHard
,   const unsigned pHard
,   const bool     useSD
,   const unsigned color_space
,   const unsigned tau_2D
,   fftwf_plan *  plan_2d_for_1
,   fftwf_plan *  plan_2d_for_2
,   fftwf_plan *  plan_2d_inv
){

//! Estimatation of sigma on each channel
vector<float> sigma_table(chnls);
if (estimate_sigma(sigma, sigma_table, chnls, color_space) != EXIT_SUCCESS)
return;

    //! Parameters initialization
    const float    lambdaHard3D = 2.7f;            //! Threshold for Hard Thresholding
    const float    tauMatch = (chnls == 1 ? 3.f : 1.f) * (sigma_table[0] < 35.0f ? 2500 : 5000); //! threshold used to determinate similarity between patches

    //! Initialization for convenience
    vector<unsigned> row_ind;
    ind_initialize(row_ind, height - kHard + 1, nHard, pHard);
    vector<unsigned> column_ind;
    ind_initialize(column_ind, width - kHard + 1, nHard, pHard);
    const unsigned kHard_2 = kHard * kHard;
    vector<float> group_3D_table(chnls * kHard_2 * NHard * column_ind.size());
    vector<float> wx_r_table;
    wx_r_table.reserve(chnls * column_ind.size());
    vector<float> hadamard_tmp(NHard);

 //   ! Check allocation memory
    if (img_basic.size() != img_noisy.size())
        img_basic.resize(img_noisy.size());
    if(img_basic_s.size() != img_noisy.size())
        img_basic_s.resize(img_noisy.size());
    if(img_basic_c.size() != img_noisy.size())
        img_basic_c.resize(img_noisy.size());
    if(img_basic_h.size() != img_noisy.size())
        img_basic_h.resize(img_noisy.size());
    if(img_basic_v.size() != img_noisy.size())
        img_basic_v.resize(img_noisy.size());
    if(img_basic_D135.size() != img_noisy.size())
        img_basic_D135.resize(img_noisy.size());
    if(img_basic_D45.size() != img_noisy.size())
        img_basic_D45.resize(img_noisy.size());


    //! Preprocessing (KaiserWindow, Threshold, DCT normalization, ...)
    for (int Kernel_type = 0; Kernel_type <= 6 ; Kernel_type++){
        vector<float> kaiser_window(kHard_2);
        vector<float> coef_norm(kHard_2);
        vector<float> coef_norm_inv(kHard_2);
        preProcess(kaiser_window, coef_norm, coef_norm_inv, kHard,Kernel_type);

    //! Preprocessing of Bior table
        vector<float> lpd, hpd, lpr, hpr;
        bior15_coef(lpd, hpd, lpr, hpr);

    //! For aggregation part
        vector<float> denominator(width * height * chnls, 0.0f);
        vector<float> numerator  (width * height * chnls, 0.0f);

    //! Precompute Bloc-Matching
        vector<vector<unsigned> > patch_table;
        precompute_BM(patch_table, img_noisy, width, height, kHard, NHard, nHard, pHard, tauMatch);//add kernel to multiply by patches her 

    //! table_2D[p * N + q + (i * width + j) * kHard_2 + c * (2 * nHard + 1) * width * kHard_2]
        vector<float> table_2D((2 * nHard + 1) * width * chnls * kHard_2, 0.0f);

    //! Loop on i_r
        for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
        {
            const unsigned i_r = row_ind[ind_i];

        //! Update of table_2D
            if (tau_2D == DCT)
                dct_2d_process(table_2D, img_noisy, plan_2d_for_1, plan_2d_for_2, nHard,
                 width, height, chnls, kHard, i_r, pHard, coef_norm,
                 row_ind[0], row_ind.back());
            else if (tau_2D == BIOR)
                bior_2d_process(table_2D, img_noisy, nHard, width, height, chnls,
                    kHard, i_r, pHard, row_ind[0], row_ind.back(), lpd, hpd);

            wx_r_table.clear();
            group_3D_table.clear();

        //! Loop on j_r
            for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
            {
            //! Initialization
                const unsigned j_r = column_ind[ind_j];
                const unsigned k_r = i_r * width + j_r;

            //! Number of similar patches
                const unsigned nSx_r = patch_table[k_r].size();

            //! Build of the 3D group
                vector<float> group_3D(chnls * nSx_r * kHard_2, 0.0f);
                for (unsigned c = 0; c < chnls; c++)
                    for (unsigned n = 0; n < nSx_r; n++)
                    {
                        const unsigned ind = patch_table[k_r][n] + (nHard - i_r) * width;
                        for (unsigned k = 0; k < kHard_2; k++)
                            group_3D[n + k * nSx_r + c * kHard_2 * nSx_r] =
                        table_2D[k + ind * kHard_2 + c * kHard_2 * (2 * nHard + 1) * width];
                    }

            //! HT filtering of the 3D group
                    vector<float> weight_table(chnls);
                    ht_filtering_hadamard(group_3D, hadamard_tmp, nSx_r, kHard, chnls, sigma_table,
                        lambdaHard3D, weight_table, !useSD);

            //! 3D weighting using Standard Deviation
                    if (useSD)
                        sd_weighting(group_3D, nSx_r, kHard, chnls, weight_table);

            //! Save the 3D group. The DCT 2D inverse will be done after.
                    for (unsigned c = 0; c < chnls; c++)
                        for (unsigned n = 0; n < nSx_r; n++)
                            for (unsigned k = 0; k < kHard_2; k++)
                                group_3D_table.push_back(group_3D[n + k * nSx_r +
                                    c * kHard_2 * nSx_r]);

            //! Save weighting
                            for (unsigned c = 0; c < chnls; c++)
                                wx_r_table.push_back(weight_table[c]);

        } //! End of loop on j_r

        //!  Apply 2D inverse transform
        if (tau_2D == DCT)
            dct_2d_inverse(group_3D_table, kHard, NHard * chnls * column_ind.size(),
             coef_norm_inv, plan_2d_inv);
        else if (tau_2D == BIOR)
            bior_2d_inverse(group_3D_table, kHard, lpr, hpr);

        //! Registration of the weighted estimation
        unsigned dec = 0;
        for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
        {
            const unsigned j_r   = column_ind[ind_j];
            const unsigned k_r   = i_r * width + j_r;
            const unsigned nSx_r = patch_table[k_r].size();
            for (unsigned c = 0; c < chnls; c++)
            {
                for (unsigned n = 0; n < nSx_r; n++)
                {
                    const unsigned k = patch_table[k_r][n] + c * width * height;
                    for (unsigned p = 0; p < kHard; p++)
                        for (unsigned q = 0; q < kHard; q++)
                        {
                            const unsigned ind = k + p * width + q;
                            numerator[ind] += kaiser_window[p * kHard + q]
                            * wx_r_table[c + ind_j * chnls]
                            * group_3D_table[p * kHard + q + n * kHard_2
                              + c * kHard_2 * nSx_r + dec];
                            denominator[ind] += kaiser_window[p * kHard + q]
                            * wx_r_table[c + ind_j * chnls];
                        }
                    }
                }
                dec += nSx_r * chnls * kHard_2;
            }

    } //! End of loop on i_r

    //! Final reconstruction
    switch(Kernel_type){
        case 1:
        for (unsigned k = 0; k < width * height * chnls; k++)
           img_basic_s[k] = numerator[k] / denominator[k];
       break;
       case 2:
        for (unsigned k = 0; k < width * height * chnls; k++)
           img_basic_c[k] = numerator[k] / denominator[k];
       break;
       case 3:
       for (unsigned k = 0; k < width * height * chnls; k++)
           img_basic_h[k] = numerator[k] / denominator[k];
       break;
       case 4:
       for (unsigned k = 0; k < width * height * chnls; k++)
           img_basic_v[k] = numerator[k] / denominator[k];
       break;
       case 5:
       for (unsigned k = 0; k < width * height * chnls; k++)
           img_basic_D135[k] = numerator[k] / denominator[k];
       break;
       case 6:
       for (unsigned k = 0; k < width * height * chnls; k++)
           img_basic_D45[k] = numerator[k] / denominator[k];
       break;
         default://case 0
         for (unsigned k = 0; k < width * height * chnls; k++)
           img_basic[k] = numerator[k] / denominator[k];

   }
}
}


/**
 * @brief Run the final process of BM3D (2nd step). The result
 *        is contained in img_denoised. The image has boundary, which
 *        are here only for block-matching and doesn't need to be
 *        denoised.
 *
 * @param sigma: value of assumed noise of the image to denoise;
 * @param img_noisy: noisy image;
 * @param img_basic: contains the denoised image after the 1st step;
 * @param img_denoised: will contain the final estimate of the denoised
 *        image after the second step;
 * @param width, height, chnls : size of img_noisy;
 * @param nWien: size of the boundary around img_noisy;
 * @param useSD: if true, use weight based on the standard variation
 *        of the 3D group for the second step, otherwise use the norm
 *        of Wiener coefficients of the 3D group;
 * @param tau_2D: DCT or BIOR.
 *
 * @return none.
 **/
void bm3d_2nd_step(
const float sigma
,   vector<float> const& img_noisy
,   vector<float> const& img_basic
,   vector<float> &img_denoised
,   vector<float> &img_denoised_s
,   vector<float> &img_denoised_c
,   vector<float> &img_denoised_hor
,   vector<float> &img_denoised_ver
,   vector<float> &img_denoised_D135
,   vector<float> &img_denoised_D45
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned nWien
,   const unsigned kWien
,   const unsigned NWien
,   const unsigned pWien
,   const bool     useSD
,   const unsigned color_space
,   const unsigned tau_2D
,   fftwf_plan *  plan_2d_for_1
,   fftwf_plan *  plan_2d_for_2
,   fftwf_plan *  plan_2d_inv
){
//! Estimatation of sigma on each channel
vector<float> sigma_table(chnls);
if (estimate_sigma(sigma, sigma_table, chnls, color_space) != EXIT_SUCCESS)
    return;

//! Parameters initialization
const float tauMatch = (sigma_table[0] < 35.0f ? 400 : 3500); //! threshold used to determinate similarity between patches

//! Initialization for convenience
vector<unsigned> row_ind;
ind_initialize(row_ind, height - kWien + 1, nWien, pWien);
vector<unsigned> column_ind;
ind_initialize(column_ind, width - kWien + 1, nWien, pWien);
const unsigned kWien_2 = kWien * kWien;
vector<float> group_3D_table(chnls * kWien_2 * NWien * column_ind.size());
vector<float> wx_r_table;
wx_r_table.reserve(chnls * column_ind.size());
vector<float> tmp(NWien);


//   ! Check allocation memory
if (img_denoised.size() != img_noisy.size())
    img_denoised.resize(img_noisy.size());
if(img_denoised_s.size() != img_noisy.size())
    img_denoised_s.resize(img_noisy.size());
if(img_denoised_c.size() != img_noisy.size())
    img_denoised_c.resize(img_noisy.size());
if(img_denoised_hor.size() != img_noisy.size())
    img_denoised_hor.resize(img_noisy.size());
if(img_denoised_ver.size() != img_noisy.size())
    img_denoised_ver.resize(img_noisy.size());
if(img_denoised_D135.size() != img_noisy.size())
    img_denoised_D135.resize(img_noisy.size());
if(img_denoised_D45.size() != img_noisy.size())
    img_denoised_D45.resize(img_noisy.size());


//! Preprocessing (KaiserWindow, Threshold, DCT normalization, ...)
vector<float> kaiser_window(kWien_2);
vector<float> coef_norm(kWien_2);
vector<float> coef_norm_inv(kWien_2);
for(int Kernel_type = 0 ; Kernel_type<=6 ; Kernel_type++){
preProcess(kaiser_window, coef_norm, coef_norm_inv, kWien, Kernel_type);

//! For aggregation part
vector<float> denominator(width * height * chnls, 0.0f);
vector<float> numerator  (width * height * chnls, 0.0f);

//! Precompute Bloc-Matching
vector<vector<unsigned> > patch_table;
precompute_BM(patch_table, img_basic, width, height, kWien, NWien, nWien, pWien, tauMatch);

//! Preprocessing of Bior table
vector<float> lpd, hpd, lpr, hpr;
bior15_coef(lpd, hpd, lpr, hpr);

//! DCT_table_2D[p * N + q + (i * width + j) * kWien_2 + c * (2 * ns + 1) * width * kWien_2]
vector<float> table_2D_img((2 * nWien + 1) * width * chnls * kWien_2, 0.0f);
vector<float> table_2D_est((2 * nWien + 1) * width * chnls * kWien_2, 0.0f);

//! Loop on i_r
for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
{
    const unsigned i_r = row_ind[ind_i];

    //! Update of DCT_table_2D
    if (tau_2D == DCT)
    {
        dct_2d_process(table_2D_img, img_noisy, plan_2d_for_1, plan_2d_for_2,
         nWien, width, height, chnls, kWien, i_r, pWien, coef_norm,
         row_ind[0], row_ind.back());
        dct_2d_process(table_2D_est, img_basic, plan_2d_for_1, plan_2d_for_2,
         nWien, width, height, chnls, kWien, i_r, pWien, coef_norm,
         row_ind[0], row_ind.back());
    }
    else if (tau_2D == BIOR)
    {
        bior_2d_process(table_2D_img, img_noisy, nWien, width, height,
            chnls, kWien, i_r, pWien, row_ind[0], row_ind.back(), lpd, hpd);
        bior_2d_process(table_2D_est, img_basic, nWien, width, height,
            chnls, kWien, i_r, pWien, row_ind[0], row_ind.back(), lpd, hpd);
    }

    wx_r_table.clear();
    group_3D_table.clear();

    //! Loop on j_r
    for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
    {
        //! Initialization
        const unsigned j_r = column_ind[ind_j];
        const unsigned k_r = i_r * width + j_r;

        //! Number of similar patches
        const unsigned nSx_r = patch_table[k_r].size();

        //! Build of the 3D group
        vector<float> group_3D_est(chnls * nSx_r * kWien_2, 0.0f);
        vector<float> group_3D_img(chnls * nSx_r * kWien_2, 0.0f);
        for (unsigned c = 0; c < chnls; c++)
            for (unsigned n = 0; n < nSx_r; n++)
            {
                const unsigned ind = patch_table[k_r][n] + (nWien - i_r) * width;
                for (unsigned k = 0; k < kWien_2; k++)
                {
                    group_3D_est[n + k * nSx_r + c * kWien_2 * nSx_r] =
                    table_2D_est[k + ind * kWien_2 + c * kWien_2 * (2 * nWien + 1) * width];
                    group_3D_img[n + k * nSx_r + c * kWien_2 * nSx_r] =
                    table_2D_img[k + ind * kWien_2 + c * kWien_2 * (2 * nWien + 1) * width];
                }
            }

        //! Wiener filtering of the 3D group
            vector<float> weight_table(chnls);
            wiener_filtering_hadamard(group_3D_img, group_3D_est, tmp, nSx_r, kWien,
                chnls, sigma_table, weight_table, !useSD);

        //! 3D weighting using Standard Deviation
            if (useSD)
                sd_weighting(group_3D_est, nSx_r, kWien, chnls, weight_table);

        //! Save the 3D group. The DCT 2D inverse will be done after.
            for (unsigned c = 0; c < chnls; c++)
                for (unsigned n = 0; n < nSx_r; n++)
                    for (unsigned k = 0; k < kWien_2; k++)
                        group_3D_table.push_back(group_3D_est[n + k * nSx_r + c * kWien_2 * nSx_r]);

        //! Save weighting
                    for (unsigned c = 0; c < chnls; c++)
                        wx_r_table.push_back(weight_table[c]);

    } //! End of loop on j_r

    //!  Apply 2D dct inverse
    if (tau_2D == DCT)
        dct_2d_inverse(group_3D_table, kWien, NWien * chnls * column_ind.size(),
         coef_norm_inv, plan_2d_inv);
    else if (tau_2D == BIOR)
        bior_2d_inverse(group_3D_table, kWien, lpr, hpr);

    //! Registration of the weighted estimation
    unsigned dec = 0;
    for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
    {
        const unsigned j_r   = column_ind[ind_j];
        const unsigned k_r   = i_r * width + j_r;
        const unsigned nSx_r = patch_table[k_r].size();
        for (unsigned c = 0; c < chnls; c++)
        {
            for (unsigned n = 0; n < nSx_r; n++)
            {
                const unsigned k = patch_table[k_r][n] + c * width * height;
                for (unsigned p = 0; p < kWien; p++)
                    for (unsigned q = 0; q < kWien; q++)
                    {
                        const unsigned ind = k + p * width + q;
                        numerator[ind] += kaiser_window[p * kWien + q]
                        * wx_r_table[c + ind_j * chnls]
                        * group_3D_table[p * kWien + q + n * kWien_2
                          + c * kWien_2 * nSx_r + dec];
                        denominator[ind] += kaiser_window[p * kWien + q]
                        * wx_r_table[c + ind_j * chnls];
                    }
                }
            }
            dec += nSx_r * chnls * kWien_2;
        }

} //! End of loop on i_r

//! Final reconstruction
// for (unsigned k = 0; k < width * height * chnls; k++)
//     img_denoised[k] = numerator[k] / denominator[k];

    //! Final reconstruction
    switch(Kernel_type){
        case 1:
        for (unsigned k = 0; k < width * height * chnls; k++)
           img_denoised_s[k] = numerator[k] / denominator[k];
       break;
       case 2:
        for (unsigned k = 0; k < width * height * chnls; k++)
           img_denoised_c[k] = numerator[k] / denominator[k];
       break;
       case 3:
       for (unsigned k = 0; k < width * height * chnls; k++)
           img_denoised_hor[k] = numerator[k] / denominator[k];
       break;
       case 4:
       for (unsigned k = 0; k < width * height * chnls; k++)
           img_denoised_ver[k] = numerator[k] / denominator[k];
       break;
       case 5:
       for (unsigned k = 0; k < width * height * chnls; k++)
           img_denoised_D135[k] = numerator[k] / denominator[k];
       break;
       case 6:
       for (unsigned k = 0; k < width * height * chnls; k++)
           img_denoised_D45[k] = numerator[k] / denominator[k];
       break;
         default://case 0
         for (unsigned k = 0; k < width * height * chnls; k++)
           img_denoised[k] = numerator[k] / denominator[k];

   }
}
}
/**
 * @brief Precompute a 2D DCT transform on all patches contained in
 *        a part of the image.
 *
 * @param DCT_table_2D : will contain the 2d DCT transform for all
 *        chosen patches;
 * @param img : image on which the 2d DCT will be processed;
 * @param plan_1, plan_2 : for convenience. Used by fftw;
 * @param nHW : size of the boundary around img;
 * @param width, height, chnls: size of img;
 * @param kHW : size of patches (kHW x kHW);
 * @param i_r: current index of the reference patches;
 * @param step: space in pixels between two references patches;
 * @param coef_norm : normalization coefficients of the 2D DCT;
 * @param i_min (resp. i_max) : minimum (resp. maximum) value
 *        for i_r. In this case the whole 2d transform is applied
 *        on every patches. Otherwise the precomputed 2d DCT is re-used
 *        without processing it.
 **/
void dct_2d_process(
vector<float> &DCT_table_2D
,   vector<float> const& img
,   fftwf_plan * plan_1
,   fftwf_plan * plan_2
,   const unsigned nHW
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned kHW
,   const unsigned i_r
,   const unsigned step
,   vector<float> const& coef_norm
,   const unsigned i_min
,   const unsigned i_max
){
//! Declarations
const unsigned kHW_2 = kHW * kHW;
const unsigned size = chnls * kHW_2 * width * (2 * nHW + 1);

//! If i_r == ns, then we have to process all DCT
if (i_r == i_min || i_r == i_max)
{
//! Allocating Memory
float* vec = (float*) fftwf_malloc(size * sizeof(float));
float* dct = (float*) fftwf_malloc(size * sizeof(float));

for (unsigned c = 0; c < chnls; c++)
{
const unsigned dc = c * width * height;
const unsigned dc_p = c * kHW_2 * width * (2 * nHW + 1);
for (unsigned i = 0; i < 2 * nHW + 1; i++)
for (unsigned j = 0; j < width - kHW; j++)
for (unsigned p = 0; p < kHW; p++)
for (unsigned q = 0; q < kHW; q++)
vec[p * kHW + q + dc_p + (i * width + j) * kHW_2] =
img[dc + (i_r + i - nHW + p) * width + j + q];
}

//! Process of all DCTs
fftwf_execute_r2r(*plan_1, vec, dct);
fftwf_free(vec);

//! Getting the result
for (unsigned c = 0; c < chnls; c++)
{
const unsigned dc   = c * kHW_2 * width * (2 * nHW + 1);
const unsigned dc_p = c * kHW_2 * width * (2 * nHW + 1);
for (unsigned i = 0; i < 2 * nHW + 1; i++)
for (unsigned j = 0; j < width - kHW; j++)
for (unsigned k = 0; k < kHW_2; k++)
DCT_table_2D[dc + (i * width + j) * kHW_2 + k] =
dct[dc_p + (i * width + j) * kHW_2 + k] * coef_norm[k];
}
fftwf_free(dct);
}
else
{
const unsigned ds = step * width * kHW_2;

//! Re-use of DCT already processed
for (unsigned c = 0; c < chnls; c++)
{
unsigned dc = c * width * (2 * nHW + 1) * kHW_2;
for (unsigned i = 0; i < 2 * nHW + 1 - step; i++)
for (unsigned j = 0; j < width - kHW; j++)
for (unsigned k = 0; k < kHW_2; k++)
DCT_table_2D[k + (i * width + j) * kHW_2 + dc] =
DCT_table_2D[k + (i * width + j) * kHW_2 + dc + ds];
}

//! Compute the new DCT
float* vec = (float*) fftwf_malloc(chnls * kHW_2 * step * width * sizeof(float));
float* dct = (float*) fftwf_malloc(chnls * kHW_2 * step * width * sizeof(float));

for (unsigned c = 0; c < chnls; c++)
{
const unsigned dc   = c * width * height;
const unsigned dc_p = c * kHW_2 * width * step;
for (unsigned i = 0; i < step; i++)
for (unsigned j = 0; j < width - kHW; j++)
for (unsigned p = 0; p < kHW; p++)
    for (unsigned q = 0; q < kHW; q++)
        vec[p * kHW + q + dc_p + (i * width + j) * kHW_2] =
    img[(p + i + 2 * nHW + 1 - step + i_r - nHW)
        * width + j + q + dc];
}

//! Process of all DCTs
fftwf_execute_r2r(*plan_2, vec, dct);
fftwf_free(vec);

//! Getting the result
for (unsigned c = 0; c < chnls; c++)
{
    const unsigned dc   = c * kHW_2 * width * (2 * nHW + 1);
    const unsigned dc_p = c * kHW_2 * width * step;
    for (unsigned i = 0; i < step; i++)
        for (unsigned j = 0; j < width - kHW; j++)
            for (unsigned k = 0; k < kHW_2; k++)
                DCT_table_2D[dc + ((i + 2 * nHW + 1 - step) * width + j) * kHW_2 + k] =
            dct[dc_p + (i * width + j) * kHW_2 + k] * coef_norm[k];
        }
        fftwf_free(dct);
    }
}

/**
 * @brief Precompute a 2D bior1.5 transform on all patches contained in
 *        a part of the image.
 *
 * @param bior_table_2D : will contain the 2d bior1.5 transform for all
 *        chosen patches;
 * @param img : image on which the 2d transform will be processed;
 * @param nHW : size of the boundary around img;
 * @param width, height, chnls: size of img;
 * @param kHW : size of patches (kHW x kHW). MUST BE A POWER OF 2 !!!
 * @param i_r: current index of the reference patches;
 * @param step: space in pixels between two references patches;
 * @param i_min (resp. i_max) : minimum (resp. maximum) value
 *        for i_r. In this case the whole 2d transform is applied
 *        on every patches. Otherwise the precomputed 2d DCT is re-used
 *        without processing it;
 * @param lpd : low pass filter of the forward bior1.5 2d transform;
 * @param hpd : high pass filter of the forward bior1.5 2d transform.
 **/
void bior_2d_process(
vector<float> &bior_table_2D
,   vector<float> const& img
,   const unsigned nHW
,   const unsigned width
,   const unsigned height
,   const unsigned chnls
,   const unsigned kHW
,   const unsigned i_r
,   const unsigned step
,   const unsigned i_min
,   const unsigned i_max
,   vector<float> &lpd
,   vector<float> &hpd
){
//! Declarations
const unsigned kHW_2 = kHW * kHW;

//! If i_r == ns, then we have to process all Bior1.5 transforms
if (i_r == i_min || i_r == i_max)
{
for (unsigned c = 0; c < chnls; c++)
{
const unsigned dc = c * width * height;
const unsigned dc_p = c * kHW_2 * width * (2 * nHW + 1);
for (unsigned i = 0; i < 2 * nHW + 1; i++)
    for (unsigned j = 0; j < width - kHW; j++)
    {
        bior_2d_forward(img, bior_table_2D, kHW, dc +
           (i_r + i - nHW) * width + j, width,
           dc_p + (i * width + j) * kHW_2, lpd, hpd);
    }
}
}
else
{
const unsigned ds = step * width * kHW_2;

//! Re-use of Bior1.5 already processed
for (unsigned c = 0; c < chnls; c++)
{
    unsigned dc = c * width * (2 * nHW + 1) * kHW_2;
    for (unsigned i = 0; i < 2 * nHW + 1 - step; i++)
        for (unsigned j = 0; j < width - kHW; j++)
            for (unsigned k = 0; k < kHW_2; k++)
                bior_table_2D[k + (i * width + j) * kHW_2 + dc] =
            bior_table_2D[k + (i * width + j) * kHW_2 + dc + ds];
        }

//! Compute the new Bior
        for (unsigned c = 0; c < chnls; c++)
        {
            const unsigned dc   = c * width * height;
            const unsigned dc_p = c * kHW_2 * width * (2 * nHW + 1);
            for (unsigned i = 0; i < step; i++)
                for (unsigned j = 0; j < width - kHW; j++)
                {
                    bior_2d_forward(img, bior_table_2D, kHW, dc +
                       (i + 2 * nHW + 1 - step + i_r - nHW) * width + j,
                       width, dc_p + ((i + 2 * nHW + 1 - step)
                           * width + j) * kHW_2, lpd, hpd);
                }
            }
        }
    }

/**
 * @brief HT filtering using Welsh-Hadamard transform (do only third
 *        dimension transform, Hard Thresholding and inverse transform).
 *
 * @param group_3D : contains the 3D block for a reference patch;
 * @param tmp: allocated vector used in Hadamard transform for convenience;
 * @param nSx_r : number of similar patches to a reference one;
 * @param kHW : size of patches (kHW x kHW);
 * @param chnls : number of channels of the image;
 * @param sigma_table : contains value of noise for each channel;
 * @param lambdaHard3D : value of thresholding;
 * @param weight_table: the weighting of this 3D group for each channel;
 * @param doWeight: if true process the weighting, do nothing
 *        otherwise.
 *
 * @return none.
 **/
void ht_filtering_hadamard(
vector<float> &group_3D
,   vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned kHard
,   const unsigned chnls
,   vector<float> const& sigma_table
,   const float lambdaHard3D
,   vector<float> &weight_table
,   const bool doWeight
){
//! Declarations
const unsigned kHard_2 = kHard * kHard;
for (unsigned c = 0; c < chnls; c++)
    weight_table[c] = 0.0f;
const float coef_norm = sqrtf((float) nSx_r);
const float coef = 1.0f / (float) nSx_r;

//! Process the Welsh-Hadamard transform on the 3rd dimension
for (unsigned n = 0; n < kHard_2 * chnls; n++)
    hadamard_transform(group_3D, tmp, nSx_r, n * nSx_r);

//! Hard Thresholding
for (unsigned c = 0; c < chnls; c++)
{
    const unsigned dc = c * nSx_r * kHard_2;
    const float T = lambdaHard3D * sigma_table[c] * coef_norm;
    for (unsigned k = 0; k < kHard_2 * nSx_r; k++)
    {
        if (fabs(group_3D[k + dc]) > T)
            weight_table[c]++;
        else
            group_3D[k + dc] = 0.0f;
    }
}

//! Process of the Welsh-Hadamard inverse transform
for (unsigned n = 0; n < kHard_2 * chnls; n++)
    hadamard_transform(group_3D, tmp, nSx_r, n * nSx_r);

for (unsigned k = 0; k < group_3D.size(); k++)
    group_3D[k] *= coef;

//! Weight for aggregation
if (doWeight)
    for (unsigned c = 0; c < chnls; c++)
        weight_table[c] = (weight_table[c] > 0.0f ? 1.0f / (float)
            (sigma_table[c] * sigma_table[c] * weight_table[c]) : 1.0f);
}

/**
 * @brief Wiener filtering using Hadamard transform.
 *
 * @param group_3D_img : contains the 3D block built on img_noisy;
 * @param group_3D_est : contains the 3D block built on img_basic;
 * @param tmp: allocated vector used in hadamard transform for convenience;
 * @param nSx_r : number of similar patches to a reference one;
 * @param kWien : size of patches (kWien x kWien);
 * @param chnls : number of channels of the image;
 * @param sigma_table : contains value of noise for each channel;
 * @param weight_table: the weighting of this 3D group for each channel;
 * @param doWeight: if true process the weighting, do nothing
 *        otherwise.
 *
 * @return none.
 **/
void wiener_filtering_hadamard(
vector<float> &group_3D_img
,   vector<float> &group_3D_est
,   vector<float> &tmp
,   const unsigned nSx_r
,   const unsigned kWien
,   const unsigned chnls
,   vector<float> const& sigma_table
,   vector<float> &weight_table
,   const bool doWeight
){
//! Declarations
const unsigned kWien_2 = kWien * kWien;
const float coef = 1.0f / (float) nSx_r;

for (unsigned c = 0; c < chnls; c++)
    weight_table[c] = 0.0f;

//! Process the Welsh-Hadamard transform on the 3rd dimension
for (unsigned n = 0; n < kWien_2 * chnls; n++)
{
    hadamard_transform(group_3D_img, tmp, nSx_r, n * nSx_r);
    hadamard_transform(group_3D_est, tmp, nSx_r, n * nSx_r);
}

//! Wiener Filtering
for (unsigned c = 0; c < chnls; c++)
{
    const unsigned dc = c * nSx_r * kWien_2;
    for (unsigned k = 0; k < kWien_2 * nSx_r; k++)
    {
        float value = group_3D_est[dc + k] * group_3D_est[dc + k] * coef;
        value /= (value + sigma_table[c] * sigma_table[c]);
        group_3D_est[k + dc] = group_3D_img[k + dc] * value * coef;
        weight_table[c] += value;
    }
}

//! Process of the Welsh-Hadamard inverse transform
for (unsigned n = 0; n < kWien_2 * chnls; n++)
    hadamard_transform(group_3D_est, tmp, nSx_r, n * nSx_r);

//! Weight for aggregation
if (doWeight)
    for (unsigned c = 0; c < chnls; c++)
        weight_table[c] = (weight_table[c] > 0.0f ? 1.0f / (float)
            (sigma_table[c] * sigma_table[c] * weight_table[c]) : 1.0f);
}

/**
 * @brief Apply 2D dct inverse to a lot of patches.
 *
 * @param group_3D_table: contains a huge number of patches;
 * @param kHW : size of patch;
 * @param coef_norm_inv: contains normalization coefficients;
 * @param plan : for convenience. Used by fftw.
 *
 * @return none.
 **/
void dct_2d_inverse(
vector<float> &group_3D_table
,   const unsigned kHW
,   const unsigned N
,   vector<float> const& coef_norm_inv
,   fftwf_plan * plan
){
//! Declarations
const unsigned kHW_2 = kHW * kHW;
const unsigned size = kHW_2 * N;
const unsigned Ns   = group_3D_table.size() / kHW_2;

//! Allocate Memory
float* vec = (float*) fftwf_malloc(size * sizeof(float));
float* dct = (float*) fftwf_malloc(size * sizeof(float));

//! Normalization
for (unsigned n = 0; n < Ns; n++)
for (unsigned k = 0; k < kHW_2; k++)
    dct[k + n * kHW_2] = group_3D_table[k + n * kHW_2] * coef_norm_inv[k];

//! 2D dct inverse
fftwf_execute_r2r(*plan, dct, vec);
fftwf_free(dct);

//! Getting the result + normalization
const float coef = 1.0f / (float)(kHW * 2);
for (unsigned k = 0; k < group_3D_table.size(); k++)
    group_3D_table[k] = coef * vec[k];

//! Free Memory
fftwf_free(vec);
}

void bior_2d_inverse(
vector<float> &group_3D_table
,   const unsigned kHW
,   vector<float> const& lpr
,   vector<float> const& hpr
){
//! Declarations
const unsigned kHW_2 = kHW * kHW;
const unsigned N = group_3D_table.size() / kHW_2;

//! Bior process
for (unsigned n = 0; n < N; n++)
    bior_2d_inverse(group_3D_table, kHW, n * kHW_2, lpr, hpr);
}

/** ----------------- **/
/** - Preprocessing - **/
/** ----------------- **/
/**
 * @brief Preprocess
 *
 * @param kaiser_window[kHW * kHW]: Will contain values of a Kaiser Window;
 * @param coef_norm: Will contain values used to normalize the 2D DCT;
 * @param coef_norm_inv: Will contain values used to normalize the 2D DCT;
 * @param bior1_5_for: will contain coefficients for the bior1.5 forward transform
 * @param bior1_5_inv: will contain coefficients for the bior1.5 inverse transform
 * @param kHW: size of patches (need to be 8 or 12).
 *
 * @return none.
 **/
void preProcess(
vector<float> &kaiserWindow
,   vector<float> &coef_norm
,   vector<float> &coef_norm_inv
,   const unsigned kHW
,   int Kernel_type
){
//! Kaiser Window coefficients
if(Kernel_type == 0){
//! Kaiser Window coefficients
if (kHW == 8)
{
//! First quarter of the matrix
kaiserWindow[0 + kHW * 0] = 0.1924f; kaiserWindow[0 + kHW * 1] = 0.2989f; kaiserWindow[0 + kHW * 2] = 0.3846f; kaiserWindow[0 + kHW * 3] = 0.4325f;
kaiserWindow[1 + kHW * 0] = 0.2989f; kaiserWindow[1 + kHW * 1] = 0.4642f; kaiserWindow[1 + kHW * 2] = 0.5974f; kaiserWindow[1 + kHW * 3] = 0.6717f;
kaiserWindow[2 + kHW * 0] = 0.3846f; kaiserWindow[2 + kHW * 1] = 0.5974f; kaiserWindow[2 + kHW * 2] = 0.7688f; kaiserWindow[2 + kHW * 3] = 0.8644f;
kaiserWindow[3 + kHW * 0] = 0.4325f; kaiserWindow[3 + kHW * 1] = 0.6717f; kaiserWindow[3 + kHW * 2] = 0.8644f; kaiserWindow[3 + kHW * 3] = 0.9718f;

//! Completing the rest of the matrix by symmetry
for(unsigned i = 0; i < kHW / 2; i++)
for (unsigned j = kHW / 2; j < kHW; j++)
    kaiserWindow[i + kHW * j] = kaiserWindow[i + kHW * (kHW - j - 1)];

for (unsigned i = kHW / 2; i < kHW; i++)
    for (unsigned j = 0; j < kHW; j++)
        kaiserWindow[i + kHW * j] = kaiserWindow[kHW - i - 1 + kHW * j];
}

else if (kHW == 4){
    kaiserWindow[0 + kHW * 0]=0.1924f;kaiserWindow[0 + kHW * 1]=0.4055f; kaiserWindow[0 + kHW * 2]=0.4055f;
    kaiserWindow[1 + kHW * 0]=0.4055f;kaiserWindow[1 + kHW * 1]=0.8544f;kaiserWindow[1 + kHW * 2]=0.8544f;
    kaiserWindow[2 + kHW * 0]=0.4055f;kaiserWindow[2 + kHW * 1]=0.8544f;kaiserWindow[2 + kHW * 2]=0.8544f;
    kaiserWindow[3 + kHW * 0]=0.1924f;kaiserWindow[3 + kHW * 1]=0.4055f;kaiserWindow[3 + kHW * 2]=0.4055f;


    kaiserWindow[0 + kHW * 3]=0.1924f;
    kaiserWindow[1 + kHW * 3]=0.4055f;
    kaiserWindow[2 + kHW * 3]=0.4055f;
    kaiserWindow[3 + kHW * 3]=0.1924f;

}
else if (kHW == 12)
{
//! First quarter of the matrix
    kaiserWindow[0 + kHW * 0] = 0.1924f; kaiserWindow[0 + kHW * 1] = 0.2615f; kaiserWindow[0 + kHW * 2] = 0.3251f; kaiserWindow[0 + kHW * 3] = 0.3782f;  kaiserWindow[0 + kHW * 4] = 0.4163f;  kaiserWindow[0 + kHW * 5] = 0.4362f;
    kaiserWindow[1 + kHW * 0] = 0.2615f; kaiserWindow[1 + kHW * 1] = 0.3554f; kaiserWindow[1 + kHW * 2] = 0.4419f; kaiserWindow[1 + kHW * 3] = 0.5139f;  kaiserWindow[1 + kHW * 4] = 0.5657f;  kaiserWindow[1 + kHW * 5] = 0.5927f;
    kaiserWindow[2 + kHW * 0] = 0.3251f; kaiserWindow[2 + kHW * 1] = 0.4419f; kaiserWindow[2 + kHW * 2] = 0.5494f; kaiserWindow[2 + kHW * 3] = 0.6390f;  kaiserWindow[2 + kHW * 4] = 0.7033f;  kaiserWindow[2 + kHW * 5] = 0.7369f;
    kaiserWindow[3 + kHW * 0] = 0.3782f; kaiserWindow[3 + kHW * 1] = 0.5139f; kaiserWindow[3 + kHW * 2] = 0.6390f; kaiserWindow[3 + kHW * 3] = 0.7433f;  kaiserWindow[3 + kHW * 4] = 0.8181f;  kaiserWindow[3 + kHW * 5] = 0.8572f;
    kaiserWindow[4 + kHW * 0] = 0.4163f; kaiserWindow[4 + kHW * 1] = 0.5657f; kaiserWindow[4 + kHW * 2] = 0.7033f; kaiserWindow[4 + kHW * 3] = 0.8181f;  kaiserWindow[4 + kHW * 4] = 0.9005f;  kaiserWindow[4 + kHW * 5] = 0.9435f;
    kaiserWindow[5 + kHW * 0] = 0.4362f; kaiserWindow[5 + kHW * 1] = 0.5927f; kaiserWindow[5 + kHW * 2] = 0.7369f; kaiserWindow[5 + kHW * 3] = 0.8572f;  kaiserWindow[5 + kHW * 4] = 0.9435f;  kaiserWindow[5 + kHW * 5] = 0.9885f;

//! Completing the rest of the matrix by symmetry
for(unsigned i = 0; i < kHW / 2; i++)
for (unsigned j = kHW / 2; j < kHW; j++)
    kaiserWindow[i + kHW * j] = kaiserWindow[i + kHW * (kHW - j - 1)];

for (unsigned i = kHW / 2; i < kHW; i++)
    for (unsigned j = 0; j < kHW; j++)
        kaiserWindow[i + kHW * j] = kaiserWindow[kHW - i - 1 + kHW * j];
}
else
    for (unsigned k = 0; k < kHW * kHW; k++)
        kaiserWindow[k] = 1.0f;

                                        }
else if(Kernel_type == 1){// square flat kernel
//  cout << endl;
    if (kHW == 8)
    {
        for (int i = 0; i < kHW ; i++){
            for (int j = 0; j < kHW; j++){
                kaiserWindow[j+ kHW * i] = (float)1/(float)64;
//    cout << kaiserWindow[j+ kHW * i] <<" ";
            }
//   cout<<endl;
        }


    }

//  cout << endl;
    else if (kHW == 4)
    {
        for (int i = 0; i < kHW ; i++){
            for (int j = 0; j < kHW; j++){
                kaiserWindow[j+ kHW * i] = (float)1/(float)16;
//    cout << kaiserWindow[j+ kHW * i] <<" ";
            }
//   cout<<endl;
        }


    }
    else if (kHW == 12)
    {

        for (int i = 0; i < kHW ; i++){
            for (int j = 0; j < kHW; j++){
                kaiserWindow[j+ kHW * i] = (float)1/(float)144;
//    cout << kaiserWindow[j+ kHW * i] <<" ";
            }
//  cout<<endl;
        }

    }
    else
        for (unsigned k = 0; k < kHW * kHW; k++)
            kaiserWindow[k] = 1.0f;

    }

else if (Kernel_type == 2){ //horizontal kernel
    if (kHW == 8){
        kaiserWindow[0 + kHW * 0] = 0.000001f; kaiserWindow[0 + kHW * 1] = 0.000001f; kaiserWindow[0 + kHW * 2] = 0.000001f; kaiserWindow[0 + kHW * 3] = 0.000001f;kaiserWindow[0 + kHW * 4] = 0.000001f;kaiserWindow[0 + kHW * 5] = 0.000001f;kaiserWindow[0 + kHW * 6] = 0.000001f;kaiserWindow[0 + kHW * 7] = 0.000001f;
        kaiserWindow[1 + kHW * 0] = 0.000001f; kaiserWindow[1 + kHW * 1] = 0.000001f; kaiserWindow[1 + kHW * 2] = 0.000001f; kaiserWindow[1 + kHW * 3] = 0.000001f;kaiserWindow[1 + kHW * 4] = 0.000001f;kaiserWindow[1 + kHW * 5] = 0.000001f;kaiserWindow[1 + kHW * 6] = 0.000001f;kaiserWindow[1 + kHW * 7] = 0.000001f;
        kaiserWindow[2 + kHW * 0] = 0.0313f; kaiserWindow[2 + kHW * 1] = 0.0313f; kaiserWindow[2 + kHW * 2] = 0.0313f; kaiserWindow[2 + kHW * 3] = 0.0313f;kaiserWindow[2 + kHW * 4] = 0.0313f;kaiserWindow[2 + kHW * 5] = 0.0313f;kaiserWindow[2 + kHW * 6] = 0.0313f;kaiserWindow[2 + kHW * 7] = 0.0313f;
        kaiserWindow[3 + kHW * 0] = 0.0313f; kaiserWindow[3 + kHW * 1] = 0.0313f; kaiserWindow[3 + kHW * 2] = 0.0313f; kaiserWindow[3 + kHW * 3] = 0.0313f;kaiserWindow[3 + kHW * 4] = 0.0313f;kaiserWindow[3 + kHW * 5] = 0.0313f;kaiserWindow[3 + kHW * 6] = 0.0313f;kaiserWindow[3 + kHW * 7] = 0.0313f;
        kaiserWindow[4 + kHW * 0] = 0.0313f; kaiserWindow[4 + kHW * 1] = 0.0313f; kaiserWindow[4 + kHW * 2] = 0.0313f; kaiserWindow[4 + kHW * 3] = 0.0313f;kaiserWindow[4 + kHW * 4] = 0.0313f;kaiserWindow[4 + kHW * 5] = 0.0313f;kaiserWindow[4 + kHW * 6] = 0.0313f;kaiserWindow[4 + kHW * 7] = 0.0313f;
        kaiserWindow[5 + kHW * 0] = 0.0313f; kaiserWindow[5 + kHW * 1] = 0.0313f; kaiserWindow[5 + kHW * 2] = 0.0313f; kaiserWindow[5 + kHW * 3] = 0.0313f;kaiserWindow[5 + kHW * 4] = 0.0313f;kaiserWindow[5 + kHW * 5] = 00.0313f;kaiserWindow[5 + kHW * 6] = 0.0313f;kaiserWindow[5 + kHW * 7] = 0.0313f;
        kaiserWindow[6 + kHW * 0] = 0.000001f; kaiserWindow[6 + kHW * 1] = 0.000001f; kaiserWindow[6 + kHW * 2] = 0.000001f; kaiserWindow[6 + kHW * 3] = 0.000001f;kaiserWindow[6 + kHW * 4] = 0.000001f;kaiserWindow[6 + kHW * 5] = 0.000001f;kaiserWindow[6 + kHW * 6] = 0.000001f;kaiserWindow[6 + kHW * 7] = 0.000001f;
        kaiserWindow[7 + kHW * 0] = 0.000001f; kaiserWindow[7 + kHW * 1] = 0.000001f; kaiserWindow[7 + kHW * 2] = 0.000001f; kaiserWindow[7 + kHW * 3] = 0.000001f;kaiserWindow[7 + kHW * 4] = 0.000001f;kaiserWindow[7 + kHW * 5] = 0.000001f;kaiserWindow[7 + kHW * 6] = 0.000001f;kaiserWindow[7 + kHW * 7] = 0.000001f;
    }
    else if(kHW == 4){
        kaiserWindow[0 + kHW * 0]=0.0001f;kaiserWindow[0 + kHW * 1]=0.0001f; kaiserWindow[0 + kHW * 2]=0.0001f;
        kaiserWindow[1 + kHW * 0]=0.1250f;kaiserWindow[1 + kHW * 1]=0.1250f;kaiserWindow[1 + kHW * 2]=0.1250f;
        kaiserWindow[2 + kHW * 0]=0.1250f;kaiserWindow[2 + kHW * 1]=0.1250f;kaiserWindow[2 + kHW * 2]=0.1250f;
        kaiserWindow[3 + kHW * 0]=0.0001f;kaiserWindow[3 + kHW * 1]=0.0001f;kaiserWindow[3 + kHW * 2]=0.0001f;


        kaiserWindow[0 + kHW * 3]=0.0001f;
        kaiserWindow[1 + kHW * 3]=0.1250f;
        kaiserWindow[2 + kHW * 3]=0.1250f;
        kaiserWindow[3 + kHW * 3]=0.0001f;
    }
    else if (kHW == 12)
    {
     kaiserWindow[0 + kHW * 0] = 0.0000001f; kaiserWindow[0 + kHW * 1] = 0.0000001f; kaiserWindow[0 + kHW * 2] = 0.0000001f; kaiserWindow[0 + kHW * 3] = 0.0000001f;  kaiserWindow[0 + kHW * 4] = 0.0000001f;  kaiserWindow[0 + kHW * 5] = 0.0000001f;kaiserWindow[0 + kHW * 6] = 0.0000001f;kaiserWindow[0 + kHW * 7] = 0.0000001f;kaiserWindow[0 + kHW * 8] = 0.0000001f;kaiserWindow[0 + kHW * 9] = 0.0000001f;kaiserWindow[0 + kHW * 10] = 0.0000001f;kaiserWindow[0 + kHW * 11] = 0.0000001f;
     kaiserWindow[1 + kHW * 0] = 0.0000001f; kaiserWindow[1 + kHW * 1] = 0.0000001f; kaiserWindow[1 + kHW * 2] = 0.0000001f; kaiserWindow[1 + kHW * 3] = 0.0000001f;  kaiserWindow[1 + kHW * 4] = 0.0000001f;  kaiserWindow[1 + kHW * 5] = 0.0000001f;kaiserWindow[1 + kHW * 6] = 0.0000001f;kaiserWindow[1 + kHW * 7] = 0.0000001f;kaiserWindow[1 + kHW * 8] = 0.0000001f;kaiserWindow[1 + kHW * 9] = 0.0000001f;kaiserWindow[1 + kHW * 10] = 0.0000001f;kaiserWindow[1 + kHW * 11] = 0.0000001f;
     kaiserWindow[2 + kHW * 0] = 0.0000001f; kaiserWindow[2 + kHW * 1] = 0.0000001f; kaiserWindow[2 + kHW * 2] = 0.0000001f; kaiserWindow[2 + kHW * 3] = 0.0000001f;  kaiserWindow[2 + kHW * 4] = 0.0000001f;  kaiserWindow[2 + kHW * 5] = 0.0000001f;kaiserWindow[2 + kHW * 6] = 0.0000001f;kaiserWindow[2 + kHW * 7] = 0.0000001f;kaiserWindow[2 + kHW * 8] = 0.0000001f;kaiserWindow[2 + kHW * 9] = 0.0000001f;kaiserWindow[2 + kHW * 10] = 0.0000001f;kaiserWindow[2 + kHW * 11] = 0.0000001f;
     kaiserWindow[3 + kHW * 0] = 0.0139f; kaiserWindow[3 + kHW * 1] = 0.0139f; kaiserWindow[3 + kHW * 2] = 0.0139f; kaiserWindow[3 + kHW * 3] = 0.0139f;  kaiserWindow[3 + kHW * 4] = 0.0139f;  kaiserWindow[3 + kHW * 5] = 0.0139f;kaiserWindow[3 + kHW * 6] = 0.0139f;kaiserWindow[3 + kHW * 7] = 0.0139f;kaiserWindow[3 + kHW * 8] = 0.0139f;kaiserWindow[3 + kHW * 9] = 0.0139f;kaiserWindow[3 + kHW * 10] = 0.0139f;kaiserWindow[3 + kHW * 11] = 0.0139f;
     kaiserWindow[4 + kHW * 0] = 0.0139f; kaiserWindow[4 + kHW * 1] = 0.0139f; kaiserWindow[4 + kHW * 2] = 0.0139f; kaiserWindow[4 + kHW * 3] = 0.0139f;  kaiserWindow[4 + kHW * 4] = 0.0139f;  kaiserWindow[4 + kHW * 5] = 0.0139f;kaiserWindow[4 + kHW * 6] = 0.0139f;kaiserWindow[4 + kHW * 7] = 0.0139f;kaiserWindow[4 + kHW * 8] = 0.0139f;kaiserWindow[4 + kHW * 9] = 0.0139f;kaiserWindow[4 + kHW * 10] = 0.0139f;kaiserWindow[4 + kHW * 11] = 0.0139f;
     kaiserWindow[5 + kHW * 0] = 0.0139f; kaiserWindow[5 + kHW * 1] = 0.0139f; kaiserWindow[5 + kHW * 2] = 0.0139f; kaiserWindow[5 + kHW * 3] = 0.0139f;  kaiserWindow[5 + kHW * 4] = 0.0139f;  kaiserWindow[5 + kHW * 5] = 0.0139f;kaiserWindow[5 + kHW * 6] = 0.0139f;kaiserWindow[5 + kHW * 7] = 0.0139f;kaiserWindow[5 + kHW * 8] = 0.0139f;kaiserWindow[5 + kHW * 9] = 0.0139f;kaiserWindow[5 + kHW * 10] = 0.0139f;kaiserWindow[5 + kHW * 11] = 0.0139f;
     kaiserWindow[6 + kHW * 0] = 0.0139f; kaiserWindow[6 + kHW * 1] = 0.0139f; kaiserWindow[6 + kHW * 2] = 0.0139f; kaiserWindow[6 + kHW * 3] = 0.0139f;  kaiserWindow[6 + kHW * 4] = 0.0139f;  kaiserWindow[6 + kHW * 5] = 0.0139f;kaiserWindow[6 + kHW * 6] = 0.0139f;kaiserWindow[6 + kHW * 7] = 0.0139f;kaiserWindow[6 + kHW * 8] = 0.0139f;kaiserWindow[6 + kHW * 9] = 0.0139f;kaiserWindow[6 + kHW * 10] = 0.0139f;kaiserWindow[6 + kHW * 11] = 0.0139f;
     kaiserWindow[7 + kHW * 0] = 0.0139f; kaiserWindow[7 + kHW * 1] = 0.0139f; kaiserWindow[7 + kHW * 2] = 0.0139f; kaiserWindow[7 + kHW * 3] = 0.0139f;  kaiserWindow[7 + kHW * 4] = 0.0139f;  kaiserWindow[7 + kHW * 5] = 0.0139f;kaiserWindow[7 + kHW * 6] = 0.0139f;kaiserWindow[7 + kHW * 7] = 0.0139f;kaiserWindow[7 + kHW * 8] = 0.0139f;kaiserWindow[7 + kHW * 9] = 0.0139f;kaiserWindow[7 + kHW * 10] = 0.0139f;kaiserWindow[7 + kHW * 11] = 0.0139f;
     kaiserWindow[8 + kHW * 0] = 0.0139f; kaiserWindow[8 + kHW * 1] = 0.0139f; kaiserWindow[8 + kHW * 2] = 0.0139f; kaiserWindow[8 + kHW * 3] = 0.0139f;  kaiserWindow[8 + kHW * 4] = 0.0139f;  kaiserWindow[8 + kHW * 5] = 0.0139f;kaiserWindow[8 + kHW * 6] = 0.0139f;kaiserWindow[8 + kHW * 7] = 0.0139f;kaiserWindow[8 + kHW * 8] = 0.0139f;kaiserWindow[8 + kHW * 9] = 0.0139f;kaiserWindow[8 + kHW * 10] = 0.0139f;kaiserWindow[8 + kHW * 11] = 0.0139f;
     kaiserWindow[9 + kHW * 0] = 0.0000001f; kaiserWindow[9 + kHW * 1] = 0.0000001f; kaiserWindow[9 + kHW * 2] = 0.0000001f; kaiserWindow[9 + kHW * 3] = 0.0000001f;  kaiserWindow[9 + kHW * 4] = 0.0000001f;  kaiserWindow[9 + kHW * 5] = 0.0000001f;kaiserWindow[9 + kHW * 6] = 0.0000001f;kaiserWindow[9 + kHW * 7] = 0.0000001f;kaiserWindow[9 + kHW * 8] = 0.0000001f;kaiserWindow[9 + kHW * 9] = 0.0000001f;kaiserWindow[9 + kHW * 10] = 0.0000001f;kaiserWindow[9 + kHW * 11] = 0.0000001f;
     kaiserWindow[10 + kHW * 0] = 0.0000001f; kaiserWindow[10 + kHW * 1] = 0.0000001f; kaiserWindow[10 + kHW * 2] = 0.0000001f; kaiserWindow[10 + kHW * 3] = 0.0000001f;  kaiserWindow[10 + kHW * 4] = 0.0000001f;  kaiserWindow[10 + kHW * 5] = 0.0000001f;kaiserWindow[10 + kHW * 6] = 0.0000001f;kaiserWindow[10 + kHW * 7] = 0.0000001f;kaiserWindow[10 + kHW * 8] = 0.0000001f;kaiserWindow[10 + kHW * 9] = 0.0000001f;kaiserWindow[10 + kHW * 10] = 0.0000001f;kaiserWindow[10 + kHW * 11] = 0.0000001f;
     kaiserWindow[11 + kHW * 0] = 0.0000001f; kaiserWindow[11 + kHW * 1] = 0.0000001f; kaiserWindow[11 + kHW * 2] = 0.0000001f; kaiserWindow[11 + kHW * 3] = 0.0000001f;  kaiserWindow[11 + kHW * 4] = 0.0000001f;  kaiserWindow[11 + kHW * 5] = 0.0000001f;kaiserWindow[11 + kHW * 6] = 0.0000001f;kaiserWindow[11 + kHW * 7] = 0.0000001f;kaiserWindow[11 + kHW * 8] = 0.0000001f;kaiserWindow[11 + kHW * 9] = 0.0000001f;kaiserWindow[11 + kHW * 10] = 0.0000001f;kaiserWindow[11 + kHW * 11] = 0.0000001f;
 }
 else
    for (unsigned k = 0; k < kHW * kHW; k++)
     kaiserWindow[k] = 1.0f;
}

else if (Kernel_type == 3){ //vertical kernel%revise??
    if (kHW == 8){
        kaiserWindow[0 + kHW * 0] = 0.000001f; kaiserWindow[0 + kHW * 1] = 0.000001f; kaiserWindow[0 + kHW * 2] = 0.0313f; kaiserWindow[0 + kHW * 3] = 0.0313f;kaiserWindow[0 + kHW * 4] = 0.0313f;kaiserWindow[0 + kHW * 5] = 0.0313f;kaiserWindow[0 + kHW * 6] = 0.000001f;kaiserWindow[0 + kHW * 7] = 0.000001f;
        kaiserWindow[1 + kHW * 0] = 0.000001f; kaiserWindow[1 + kHW * 1] = 0.000001f; kaiserWindow[1 + kHW * 2] = 0.0313f; kaiserWindow[1 + kHW * 3] = 0.0313f;kaiserWindow[1 + kHW * 4] = 0.0313f;kaiserWindow[1 + kHW * 5] = 0.0313f;kaiserWindow[1 + kHW * 6] = 0.000001f;kaiserWindow[1 + kHW * 7] = 0.000001f;
        kaiserWindow[2 + kHW * 0] = 0.000001f; kaiserWindow[2 + kHW * 1] = 0.000001f; kaiserWindow[2 + kHW * 2] = 0.0313f; kaiserWindow[2 + kHW * 3] = 0.0313f;kaiserWindow[2 + kHW * 4] = 0.0313f;kaiserWindow[2 + kHW * 5] = 0.0313f;kaiserWindow[2 + kHW * 6] = 0.000001f;kaiserWindow[2 + kHW * 7] = 0.000001f;
        kaiserWindow[3 + kHW * 0] = 0.000001f; kaiserWindow[3 + kHW * 1] = 0.000001f; kaiserWindow[3 + kHW * 2] = 0.0313f; kaiserWindow[3 + kHW * 3] = 0.0313f;kaiserWindow[3 + kHW * 4] = 0.0313f;kaiserWindow[3 + kHW * 5] = 0.0313f;kaiserWindow[3 + kHW * 6] = 0.000001f;kaiserWindow[3 + kHW * 7] = 0.000001f;
        kaiserWindow[4 + kHW * 0] = 0.000001f; kaiserWindow[4 + kHW * 1] = 0.000001f; kaiserWindow[4 + kHW * 2] = 0.0313f; kaiserWindow[4 + kHW * 3] = 0.0313f;kaiserWindow[4 + kHW * 4] = 0.0313f;kaiserWindow[4 + kHW * 5] = 0.0313f;kaiserWindow[4 + kHW * 6] = 0.000001f;kaiserWindow[4 + kHW * 7] = 0.000001f;
        kaiserWindow[5 + kHW * 0] = 0.000001f; kaiserWindow[5 + kHW * 1] = 0.000001f; kaiserWindow[5 + kHW * 2] = 0.0313f; kaiserWindow[5 + kHW * 3] = 0.0313f;kaiserWindow[5 + kHW * 4] = 0.0313f;kaiserWindow[5 + kHW * 5] = 0.0313f;kaiserWindow[5 + kHW * 6] = 0.000001f;kaiserWindow[5 + kHW * 7] = 0.000001f;
        kaiserWindow[6 + kHW * 0] = 0.000001f; kaiserWindow[6 + kHW * 1] = 0.000001f; kaiserWindow[6 + kHW * 2] = 0.0313f; kaiserWindow[6 + kHW * 3] = 0.0313f;kaiserWindow[6 + kHW * 4] = 0.0313f;kaiserWindow[6 + kHW * 5] = 0.0313f;kaiserWindow[6 + kHW * 6] = 0.000001f;kaiserWindow[6 + kHW * 7] = 0.000001f;
        kaiserWindow[7 + kHW * 0] = 0.000001f; kaiserWindow[7 + kHW * 1] = 0.000001f; kaiserWindow[7 + kHW * 2] = 0.0313f; kaiserWindow[7 + kHW * 3] = 0.0313f;kaiserWindow[7 + kHW * 4] = 0.0313f;kaiserWindow[7 + kHW * 5] = 0.0313f;kaiserWindow[7 + kHW * 6] = 0.000001f;kaiserWindow[7 + kHW * 7] = 0.000001f;
    }
    else if(kHW == 4){
        kaiserWindow[0 + kHW * 0]=0.0001f;kaiserWindow[0 + kHW * 1]=0.1250f; kaiserWindow[0 + kHW * 2]=0.1250f;
        kaiserWindow[1 + kHW * 0]=0.0001f;kaiserWindow[1 + kHW * 1]=0.1250f;kaiserWindow[1 + kHW * 2]=0.1250f;
        kaiserWindow[2 + kHW * 0]=0.0001f;kaiserWindow[2 + kHW * 1]=0.1250f;kaiserWindow[2 + kHW * 2]=0.1250f;
        kaiserWindow[3 + kHW * 0]=0.0001f;kaiserWindow[3 + kHW * 1]=0.1250f;kaiserWindow[3 + kHW * 2]=0.1250f;


        kaiserWindow[0 + kHW * 3]=0.0001f;
        kaiserWindow[1 + kHW * 3]=0.0001f;
        kaiserWindow[2 + kHW * 3]=0.0001f;
        kaiserWindow[3 + kHW * 3]=0.0001f;
    }

    else if (kHW == 12)
    {
     kaiserWindow[0 + kHW * 0] = 0.0000001f; kaiserWindow[0 + kHW * 1] = 0.0000001f; kaiserWindow[0 + kHW * 2] = 0.0000001f; kaiserWindow[0 + kHW * 3] = 0.0139f;  kaiserWindow[0 + kHW * 4] = 0.0139f;  kaiserWindow[0 + kHW * 5] = 0.0139f;kaiserWindow[0 + kHW * 6] = 0.0139f;kaiserWindow[0 + kHW * 7] = 0.0139f;kaiserWindow[0 + kHW * 8] = 0.0139f;kaiserWindow[0 + kHW * 9] = 0.0000001f;kaiserWindow[0 + kHW * 10] = 0.0000001f;kaiserWindow[0 + kHW * 11] = 0.0000001f;
     kaiserWindow[1 + kHW * 0] = 0.0000001f; kaiserWindow[1 + kHW * 1] = 0.0000001f; kaiserWindow[1 + kHW * 2] = 0.0000001f; kaiserWindow[1 + kHW * 3] = 0.0139f;  kaiserWindow[1 + kHW * 4] = 0.0139f;  kaiserWindow[1 + kHW * 5] = 0.0139f;kaiserWindow[1 + kHW * 6] = 0.0139f;kaiserWindow[1 + kHW * 7] = 0.0139f;kaiserWindow[1 + kHW * 8] = 0.0139f;kaiserWindow[1 + kHW * 9] = 0.0000001f;kaiserWindow[1 + kHW * 10] = 0.0000001f;kaiserWindow[1 + kHW * 11] = 0.0000001f;
     kaiserWindow[2 + kHW * 0] = 0.0000001f; kaiserWindow[2 + kHW * 1] = 0.0000001f; kaiserWindow[2 + kHW * 2] = 0.0000001f; kaiserWindow[2 + kHW * 3] = 0.0139f;  kaiserWindow[2 + kHW * 4] = 0.0139f;  kaiserWindow[2 + kHW * 5] = 0.0139f;kaiserWindow[2 + kHW * 6] = 0.0139f;kaiserWindow[2 + kHW * 7] = 0.0139f;kaiserWindow[2 + kHW * 8] = 0.0139f;kaiserWindow[2 + kHW * 9] = 0.0000001f;kaiserWindow[2 + kHW * 10] = 0.0000001f;kaiserWindow[2 + kHW * 11] = 0.0000001f;
     kaiserWindow[3 + kHW * 0] = 0.0000001f; kaiserWindow[3 + kHW * 1] = 0.0000001f; kaiserWindow[3 + kHW * 2] = 0.0000001f; kaiserWindow[3 + kHW * 3] = 0.0139f;  kaiserWindow[3 + kHW * 4] = 0.0139f;  kaiserWindow[3 + kHW * 5] = 0.0139f;kaiserWindow[3 + kHW * 6] = 0.0139f;kaiserWindow[3 + kHW * 7] = 0.0139f;kaiserWindow[3 + kHW * 8] = 0.0139f;kaiserWindow[3 + kHW * 9] = 0.0000001f;kaiserWindow[3 + kHW * 10] = 0.0000001f;kaiserWindow[3 + kHW * 11] = 0.0000001f;
     kaiserWindow[4 + kHW * 0] = 0.0000001f; kaiserWindow[4 + kHW * 1] = 0.0000001f; kaiserWindow[4 + kHW * 2] = 0.0000001f; kaiserWindow[4 + kHW * 3] = 0.0139f;  kaiserWindow[4 + kHW * 4] = 0.0139f;  kaiserWindow[4 + kHW * 5] = 0.0139f;kaiserWindow[4 + kHW * 6] = 0.0139f;kaiserWindow[4 + kHW * 7] = 0.0139f;kaiserWindow[4 + kHW * 8] = 0.0139f;kaiserWindow[4 + kHW * 9] = 0.0000001f;kaiserWindow[4 + kHW * 10] = 0.0000001f;kaiserWindow[4 + kHW * 11] = 0.0000001f;
     kaiserWindow[5 + kHW * 0] = 0.0000001f; kaiserWindow[5 + kHW * 1] = 0.0000001f; kaiserWindow[5 + kHW * 2] = 0.0000001f; kaiserWindow[5 + kHW * 3] = 0.0139f;  kaiserWindow[5 + kHW * 4] = 0.0139f;  kaiserWindow[5 + kHW * 5] = 0.0139f;kaiserWindow[5 + kHW * 6] = 0.0139f;kaiserWindow[5 + kHW * 7] = 0.0139f;kaiserWindow[5 + kHW * 8] = 0.0139f;kaiserWindow[5 + kHW * 9] = 0.0000001f;kaiserWindow[5 + kHW * 10] = 0.0000001f;kaiserWindow[5 + kHW * 11] = 0.0000001f;
     kaiserWindow[6 + kHW * 0] = 0.0000001f; kaiserWindow[6 + kHW * 1] = 0.0000001f; kaiserWindow[6 + kHW * 2] = 0.0000001f; kaiserWindow[6 + kHW * 3] = 0.0139f;  kaiserWindow[6 + kHW * 4] = 0.0139f;  kaiserWindow[6 + kHW * 5] = 0.0139f;kaiserWindow[6 + kHW * 6] = 0.0139f;kaiserWindow[6 + kHW * 7] = 0.0139f;kaiserWindow[6 + kHW * 8] = 0.0139f;kaiserWindow[6 + kHW * 9] = 0.0000001f;kaiserWindow[6 + kHW * 10] = 0.0000001f;kaiserWindow[6 + kHW * 11] = 0.0000001f;
     kaiserWindow[7 + kHW * 0] = 0.0000001f; kaiserWindow[7 + kHW * 1] = 0.0000001f; kaiserWindow[7 + kHW * 2] = 0.0000001f; kaiserWindow[7 + kHW * 3] = 0.0139f;  kaiserWindow[7 + kHW * 4] = 0.0139f;  kaiserWindow[7 + kHW * 5] = 0.0139f;kaiserWindow[7 + kHW * 6] = 0.0139f;kaiserWindow[7 + kHW * 7] = 0.0139f;kaiserWindow[7 + kHW * 8] = 0.0139f;kaiserWindow[7 + kHW * 9] = 0.0000001f;kaiserWindow[7 + kHW * 10] = 0.0000001f;kaiserWindow[7 + kHW * 11] = 0.0000001f;
     kaiserWindow[8 + kHW * 0] = 0.0000001f; kaiserWindow[8 + kHW * 1] = 0.0000001f; kaiserWindow[8 + kHW * 2] = 0.0000001f; kaiserWindow[8 + kHW * 3] = 0.0139f;  kaiserWindow[8 + kHW * 4] = 0.0139f;  kaiserWindow[8 + kHW * 5] = 0.0139f;kaiserWindow[8 + kHW * 6] = 0.0139f;kaiserWindow[8 + kHW * 7] = 0.0139f;kaiserWindow[8 + kHW * 8] = 0.0139f;kaiserWindow[8 + kHW * 9] = 0.0000001f;kaiserWindow[8 + kHW * 10] = 0.0000001f;kaiserWindow[8 + kHW * 11] = 0.0000001f;
     kaiserWindow[9 + kHW * 0] = 0.0000001f; kaiserWindow[9 + kHW * 1] = 0.0000001f; kaiserWindow[9 + kHW * 2] = 0.0000001f; kaiserWindow[9 + kHW * 3] = 0.0139f;  kaiserWindow[9 + kHW * 4] = 0.0139f;  kaiserWindow[9 + kHW * 5] = 0.0139f;kaiserWindow[9 + kHW * 6] = 0.0139f;kaiserWindow[9 + kHW * 7] = 0.0139f;kaiserWindow[9 + kHW * 8] = 0.0139f;kaiserWindow[9 + kHW * 9] = 0.0000001f;kaiserWindow[9 + kHW * 10] = 0.0000001f;kaiserWindow[9 + kHW * 11] = 0.0000001f;
     kaiserWindow[10 + kHW * 0] = 0.0000001f; kaiserWindow[10 + kHW * 1] = 0.0000001f; kaiserWindow[10 + kHW * 2] = 0.0000001f; kaiserWindow[10 + kHW * 3] = 0.0139f;  kaiserWindow[10 + kHW * 4] = 0.0139f;  kaiserWindow[10 + kHW * 5] = 0.0139f;kaiserWindow[10 + kHW * 6] = 0.0139f;kaiserWindow[10 + kHW * 7] = 0.0139f;kaiserWindow[10 + kHW * 8] = 0.0139f;kaiserWindow[10 + kHW * 9] = 0.0000001f;kaiserWindow[10 + kHW * 10] = 0.0000001f;kaiserWindow[10 + kHW * 11] = 0.0000001f;
     kaiserWindow[11 + kHW * 0] = 0.0000001f; kaiserWindow[11 + kHW * 1] = 0.0000001f; kaiserWindow[11 + kHW * 2] = 0.0000001f; kaiserWindow[11 + kHW * 3] = 0.0139f;  kaiserWindow[11 + kHW * 4] = 0.02083f;  kaiserWindow[11 + kHW * 5] = 0.02083f;kaiserWindow[11 + kHW * 6] = 0.02083f;kaiserWindow[11 + kHW * 7] = 0.0139f;kaiserWindow[11 + kHW * 8] = 0.0139f;kaiserWindow[11 + kHW * 9] = 0.0000001f;kaiserWindow[11 + kHW * 10] = 0.0000001f;kaiserWindow[11 + kHW * 11] = 0.0000001f;
 }
 else
    for (unsigned k = 0; k < kHW * kHW; k++)
     kaiserWindow[k] = 1.0f;

}

else if (Kernel_type == 4){ //D135 kernel
    if (kHW == 8){
        kaiserWindow[0 + kHW * 0] = 0.0227f; kaiserWindow[0 + kHW * 1] = 0.0227f; kaiserWindow[0 + kHW * 2] = 0.0227f; kaiserWindow[0 + kHW * 3] = 0.0227f;kaiserWindow[0 + kHW * 4] = 0.000001f;kaiserWindow[0 + kHW * 5] = 0.000001f;kaiserWindow[0 + kHW * 6] = 0.000001f;kaiserWindow[0 + kHW * 7] = 0.000001f;
        kaiserWindow[1 + kHW * 0] = 0.0227f; kaiserWindow[1 + kHW * 1] = 0.0227f; kaiserWindow[1 + kHW * 2] = 0.0227f; kaiserWindow[1 + kHW * 3] = 0.0227f;kaiserWindow[1 + kHW * 4] = 0.0227f;kaiserWindow[1 + kHW * 5] = 0.000001f;kaiserWindow[1 + kHW * 6] = 0.000001f;kaiserWindow[1 + kHW * 7] = 0.000001f;
        kaiserWindow[2 + kHW * 0] = 0.0227f; kaiserWindow[2 + kHW * 1] = 0.0227f; kaiserWindow[2 + kHW * 2] = 0.0227f; kaiserWindow[2 + kHW * 3] = 0.0227f;kaiserWindow[2 + kHW * 4] = 0.0227f;kaiserWindow[2 + kHW * 5] = 0.0227f;kaiserWindow[2 + kHW * 6] = 0.000001f;kaiserWindow[2 + kHW * 7] = 0.000001f;
        kaiserWindow[3 + kHW * 0] = 0.0227f; kaiserWindow[3 + kHW * 1] = 0.0227f; kaiserWindow[3 + kHW * 2] = 0.0227f; kaiserWindow[3 + kHW * 3] = 0.0227f;kaiserWindow[3 + kHW * 4] = 0.0227f;kaiserWindow[3 + kHW * 5] = 0.0227f;kaiserWindow[3 + kHW * 6] = 0.0227f;kaiserWindow[3 + kHW * 7] = 0.000001f;
        kaiserWindow[4 + kHW * 0] = 0.000001f; kaiserWindow[4 + kHW * 1] = 0.0227f; kaiserWindow[4 + kHW * 2] = 0.0227f; kaiserWindow[4 + kHW * 3] = 0.0227f;kaiserWindow[4 + kHW * 4] = 0.0227f;kaiserWindow[4 + kHW * 5] = 0.0227f;kaiserWindow[4 + kHW * 6] = 0.0227f;kaiserWindow[4 + kHW * 7] = 0.0227f;
        kaiserWindow[5 + kHW * 0] = 0.000001f; kaiserWindow[5 + kHW * 1] = 0.000001f; kaiserWindow[5 + kHW * 2] = 0.0227f; kaiserWindow[5 + kHW * 3] = 0.0227f;kaiserWindow[5 + kHW * 4] = 0.0227f;kaiserWindow[5 + kHW * 5] = 0.0227f;kaiserWindow[5 + kHW * 6] = 0.0227f;kaiserWindow[5 + kHW * 7] = 0.0227f;
        kaiserWindow[6 + kHW * 0] = 0.000001f; kaiserWindow[6 + kHW * 1] = 0.000001f; kaiserWindow[6 + kHW * 2] = 0.000001f; kaiserWindow[6 + kHW * 3] = 0.0227f;kaiserWindow[6 + kHW * 4] = 0.0227f;kaiserWindow[6 + kHW * 5] = 0.0227f;kaiserWindow[6 + kHW * 6] = 0.0227f;kaiserWindow[6 + kHW * 7] = 0.0227f;
        kaiserWindow[7 + kHW * 0] = 0.000001f; kaiserWindow[7 + kHW * 1] = 0.000001f; kaiserWindow[7 + kHW * 2] = 0.000001f; kaiserWindow[7 + kHW * 3] = 0.000001f;kaiserWindow[7 + kHW * 4] = 0.0227f;kaiserWindow[7 + kHW * 5] = 0.0227f;kaiserWindow[7 + kHW * 6] = 0.0227f;kaiserWindow[7 + kHW * 7] = 0.0227f;
    }
    else if(kHW == 4){
        kaiserWindow[0 + kHW * 0]=0.1000f;kaiserWindow[0 + kHW * 1]=0.1000f; kaiserWindow[0 + kHW * 2]=0.0001f;
        kaiserWindow[1 + kHW * 0]=0.1000f;kaiserWindow[1 + kHW * 1]=0.1000f;kaiserWindow[1 + kHW * 2]=0.1000f;
        kaiserWindow[2 + kHW * 0]=0.0001f;kaiserWindow[2 + kHW * 1]=0.1000f;kaiserWindow[2 + kHW * 2]=0.1000f;
        kaiserWindow[3 + kHW * 0]=0.0001f;kaiserWindow[3 + kHW * 1]=0.0001f;kaiserWindow[3 + kHW * 2]=0.1000f;


        kaiserWindow[0 + kHW * 3]=0.0001f;
        kaiserWindow[1 + kHW * 3]=0.0001f;
        kaiserWindow[2 + kHW * 3]=0.1000f;
        kaiserWindow[3 + kHW * 3]=0.1000f;
    }
    else if (kHW == 12)
    {
     kaiserWindow[0 + kHW * 0] = 0.0098f; kaiserWindow[0 + kHW * 1] = 0.0098f; kaiserWindow[0 + kHW * 2] = 0.0098f; kaiserWindow[0 + kHW * 3] = 0.0098f;  kaiserWindow[0 + kHW * 4] = 0.0098f;  kaiserWindow[0 + kHW * 5] = 0.0098f;kaiserWindow[0 + kHW * 6] = 0.0000001f;kaiserWindow[0 + kHW * 7] = 0.0000001f;kaiserWindow[0 + kHW * 8] = 0.0000001f;kaiserWindow[0 + kHW * 9] = 0.0000001f;kaiserWindow[0 + kHW * 10] = 0.0000001f;kaiserWindow[0 + kHW * 11] = 0.0000001f;
     kaiserWindow[1 + kHW * 0] = 0.0098f; kaiserWindow[1 + kHW * 1] = 0.0098f; kaiserWindow[1 + kHW * 2] = 0.0098f; kaiserWindow[1 + kHW * 3] = 0.0098f;  kaiserWindow[1 + kHW * 4] = 0.0098f;  kaiserWindow[1 + kHW * 5] = 0.0098f;kaiserWindow[1 + kHW * 6] = 0.0098f;kaiserWindow[1 + kHW * 7] = 0.0000001f;kaiserWindow[1 + kHW * 8] = 0.0000001f;kaiserWindow[1 + kHW * 9] = 0.0000001f;kaiserWindow[1 + kHW * 10] = 0.0000001f;kaiserWindow[1 + kHW * 11] = 0.0000001f;
     kaiserWindow[2 + kHW * 0] = 0.0098f; kaiserWindow[2 + kHW * 1] = 0.0098f; kaiserWindow[2 + kHW * 2] = 0.0098f; kaiserWindow[2 + kHW * 3] = 0.0098f;  kaiserWindow[2 + kHW * 4] = 0.0098f;  kaiserWindow[2 + kHW * 5] = 0.0098f;kaiserWindow[2 + kHW * 6] = 0.0098f;kaiserWindow[2 + kHW * 7] = 0.0098f;kaiserWindow[2 + kHW * 8] = 0.0000001f;kaiserWindow[2 + kHW * 9] = 0.0000001f;kaiserWindow[2 + kHW * 10] = 0.0000001f;kaiserWindow[2 + kHW * 11] = 0.0000001f;
     kaiserWindow[3 + kHW * 0] = 0.0098f; kaiserWindow[3 + kHW * 1] = 0.0098f; kaiserWindow[3 + kHW * 2] = 0.0098f; kaiserWindow[3 + kHW * 3] = 0.0098f;  kaiserWindow[3 + kHW * 4] = 0.0098f;  kaiserWindow[3 + kHW * 5] = 0.0098f;kaiserWindow[3 + kHW * 6] = 0.0098f;kaiserWindow[3 + kHW * 7] = 0.0098f;kaiserWindow[3 + kHW * 8] = 0.0098f;kaiserWindow[3 + kHW * 9] = 0.0000001f;kaiserWindow[3 + kHW * 10] = 0.0000001f;kaiserWindow[3 + kHW * 11] = 0.0000001f;
     kaiserWindow[4 + kHW * 0] = 0.0098f; kaiserWindow[4 + kHW * 1] = 0.0098f; kaiserWindow[4 + kHW * 2] = 0.0098f; kaiserWindow[4 + kHW * 3] = 0.0098f;  kaiserWindow[4 + kHW * 4] = 0.0098f;  kaiserWindow[4 + kHW * 5] = 0.0098f;kaiserWindow[4 + kHW * 6] = 0.0098f;kaiserWindow[4 + kHW * 7] = 0.0098f;kaiserWindow[4 + kHW * 8] = 0.0098f;kaiserWindow[4 + kHW * 9] = 0.0098f;kaiserWindow[4 + kHW * 10] = 0.0000001f;kaiserWindow[4 + kHW * 11] = 0.0000001f;
     kaiserWindow[5 + kHW * 0] = 0.0098f; kaiserWindow[5 + kHW * 1] = 0.0098f; kaiserWindow[5 + kHW * 2] = 0.0098f; kaiserWindow[5 + kHW * 3] = 0.0098f;  kaiserWindow[5 + kHW * 4] = 0.0098f;  kaiserWindow[5 + kHW * 5] = 0.0098f;kaiserWindow[5 + kHW * 6] = 0.0098f;kaiserWindow[5 + kHW * 7] = 0.0098f;kaiserWindow[5 + kHW * 8] = 0.0098f;kaiserWindow[5 + kHW * 9] = 0.0098f;kaiserWindow[5 + kHW * 10] = 0.0098f;kaiserWindow[5 + kHW * 11] = 0.0000001f;
     kaiserWindow[6 + kHW * 0] = 0.0000001f; kaiserWindow[6 + kHW * 1] = 0.0098f; kaiserWindow[6 + kHW * 2] = 0.0098f; kaiserWindow[6 + kHW * 3] = 0.0098f;  kaiserWindow[6 + kHW * 4] = 0.0098f;  kaiserWindow[6 + kHW * 5] = 0.0098f;kaiserWindow[6 + kHW * 6] = 0.0098f;kaiserWindow[6 + kHW * 7] = 0.0098f;kaiserWindow[6 + kHW * 8] = 0.0098f;kaiserWindow[6 + kHW * 9] = 0.0098f;kaiserWindow[6 + kHW * 10] = 0.0098f;kaiserWindow[6 + kHW * 11] = 0.0098f;
     kaiserWindow[7 + kHW * 0] = 0.0000001f; kaiserWindow[7 + kHW * 1] = 0.0000001f; kaiserWindow[7 + kHW * 2] = 0.0098f; kaiserWindow[7 + kHW * 3] = 0.0098f;  kaiserWindow[7 + kHW * 4] = 0.0098f;  kaiserWindow[7 + kHW * 5] = 0.0098f;kaiserWindow[7 + kHW * 6] = 0.0098f;kaiserWindow[7 + kHW * 7] = 0.0098f;kaiserWindow[7 + kHW * 8] = 0.0098f;kaiserWindow[7 + kHW * 9] = 0.0098f;kaiserWindow[7 + kHW * 10] = 0.0098f;kaiserWindow[7 + kHW * 11] = 0.0098f;
     kaiserWindow[8 + kHW * 0] = 0.0000001f; kaiserWindow[8 + kHW * 1] = 0.0000001f; kaiserWindow[8 + kHW * 2] = 0.0000001f; kaiserWindow[8 + kHW * 3] = 0.0098f;  kaiserWindow[8 + kHW * 4] = 0.0098f;  kaiserWindow[8 + kHW * 5] = 0.0098f;kaiserWindow[8 + kHW * 6] = 0.0098f;kaiserWindow[8 + kHW * 7] = 0.0098f;kaiserWindow[8 + kHW * 8] = 0.0098f;kaiserWindow[8 + kHW * 9] = 0.0098f;kaiserWindow[8 + kHW * 10] = 0.0098f;kaiserWindow[8 + kHW * 11] = 0.0098f;
     kaiserWindow[9 + kHW * 0] = 0.0000001f; kaiserWindow[9 + kHW * 1] = 0.0000001f; kaiserWindow[9 + kHW * 2] = 0.0000001f; kaiserWindow[9 + kHW * 3] = 0.0000001f;  kaiserWindow[9 + kHW * 4] = 0.0098f;  kaiserWindow[9 + kHW * 5] = 0.0098f;kaiserWindow[9 + kHW * 6] = 0.0098f;kaiserWindow[9 + kHW * 7] = 0.0098f;kaiserWindow[9 + kHW * 8] = 0.0098f;kaiserWindow[9 + kHW * 9] = 0.0098f;kaiserWindow[9 + kHW * 10] = 0.0098f;kaiserWindow[9 + kHW * 11] = 0.0098f;
     kaiserWindow[10 + kHW * 0] = 0.0000001f; kaiserWindow[10 + kHW * 1] = 0.0000001f; kaiserWindow[10 + kHW * 2] = 0.0000001f; kaiserWindow[10 + kHW * 3] = 0.0000001f;  kaiserWindow[10 + kHW * 4] = 0.0000001f;  kaiserWindow[10 + kHW * 5] = 0.0098f;kaiserWindow[10 + kHW * 6] = 0.0098f;kaiserWindow[10 + kHW * 7] = 0.0098f;kaiserWindow[10 + kHW * 8] = 0.0098f;kaiserWindow[10 + kHW * 9] = 0.0098f;kaiserWindow[10 + kHW * 10] = 0.0098f;kaiserWindow[10 + kHW * 11] = 0.0098f;
     kaiserWindow[11 + kHW * 0] = 0.0000001f; kaiserWindow[11 + kHW * 1] = 0.0000001f; kaiserWindow[11 + kHW * 2] = 0.0000001f; kaiserWindow[11 + kHW * 3] = 0.0000001f;  kaiserWindow[11 + kHW * 4] = 0.0000001f;  kaiserWindow[11 + kHW * 5] =  0.0000001f;kaiserWindow[11 + kHW * 6] = 0.0098f;kaiserWindow[11 + kHW * 7] = 0.0098f;kaiserWindow[11 + kHW * 8] = 0.0098f;kaiserWindow[11 + kHW * 9] = 0.0098f;kaiserWindow[11 + kHW * 10] = 00.0098f;kaiserWindow[11 + kHW * 11] = 0.0098f;
 }
 else
    for (unsigned k = 0; k < kHW * kHW; k++)
     kaiserWindow[k] = 1.0f;

}
//D45 Kernel
else if(Kernel_type == 5){
    if (kHW == 8){
        kaiserWindow[0 + kHW * 0] = 0.000001f; kaiserWindow[0 + kHW * 1] = 0.000001f; kaiserWindow[0 + kHW * 2] = 0.000001f; kaiserWindow[0 + kHW * 3] = 0.000001f;kaiserWindow[0 + kHW * 4] = 0.0227f;kaiserWindow[0 + kHW * 5] = 0.0227f;kaiserWindow[0 + kHW * 6] = 0.0227f;kaiserWindow[0 + kHW * 7] = 0.0227f;
        kaiserWindow[1 + kHW * 0] = 0.000001f; kaiserWindow[1 + kHW * 1] = 0.000001f; kaiserWindow[1 + kHW * 2] = 0.000001f; kaiserWindow[1 + kHW * 3] = 0.0227f;kaiserWindow[1 + kHW * 4] = 0.0227f;kaiserWindow[1 + kHW * 5] = 0.0227f;kaiserWindow[1 + kHW * 6] = 0.0227f;kaiserWindow[1 + kHW * 7] = 0.0227f;
        kaiserWindow[2 + kHW * 0] = 0.000001f; kaiserWindow[2 + kHW * 1] = 0.000001f; kaiserWindow[2 + kHW * 2] = 0.0227f; kaiserWindow[2 + kHW * 3] = 0.0227f;kaiserWindow[2 + kHW * 4] = 0.0227f;kaiserWindow[2 + kHW * 5] = 0.0227f;kaiserWindow[2 + kHW * 6] = 0.0227f;kaiserWindow[2 + kHW * 7] = 0.0227f;
        kaiserWindow[3 + kHW * 0] = 0.000001f; kaiserWindow[3 + kHW * 1] = 0.0227f; kaiserWindow[3 + kHW * 2] = 0.0227f; kaiserWindow[3 + kHW * 3] = 0.0227f;kaiserWindow[3 + kHW * 4] = 0.0227f;kaiserWindow[3 + kHW * 5] = 0.0227f;kaiserWindow[3 + kHW * 6] = 0.0227f;kaiserWindow[3 + kHW * 7] = 0.0227f;
        kaiserWindow[4 + kHW * 0] = 0.0227f; kaiserWindow[4 + kHW * 1] = 0.0227f; kaiserWindow[4 + kHW * 2] = 0.0227f; kaiserWindow[4 + kHW * 3] = 0.0227f;kaiserWindow[4 + kHW * 4] = 0.0227f;kaiserWindow[4 + kHW * 5] = 0.0227f;kaiserWindow[4 + kHW * 6] = 0.0227f;kaiserWindow[4 + kHW * 7] = 0.000001f;
        kaiserWindow[5 + kHW * 0] = 0.0227f; kaiserWindow[5 + kHW * 1] = 0.0227f; kaiserWindow[5 + kHW * 2] = 0.0227f; kaiserWindow[5 + kHW * 3] = 0.0227f;kaiserWindow[5 + kHW * 4] = 0.0227f;kaiserWindow[5 + kHW * 5] = 0.0227f;kaiserWindow[5 + kHW * 6] = 0.000001f;kaiserWindow[5 + kHW * 7] = 0.000001f;
        kaiserWindow[6 + kHW * 0] = 0.0227f; kaiserWindow[6 + kHW * 1] = 0.0227f; kaiserWindow[6 + kHW * 2] = 0.0227f; kaiserWindow[6 + kHW * 3] = 0.0227f;kaiserWindow[6 + kHW * 4] = 0.0227f;kaiserWindow[6 + kHW * 5] = 0.000001f;kaiserWindow[6 + kHW * 6] = 0.000001f;kaiserWindow[6 + kHW * 7] = 0.000001f;
        kaiserWindow[7 + kHW * 0] = 0.0227f; kaiserWindow[7 + kHW * 1] = 0.0227f; kaiserWindow[7 + kHW * 2] = 0.0227f; kaiserWindow[7 + kHW * 3] = 0.0227f;kaiserWindow[7 + kHW * 4] = 0.000001f;kaiserWindow[7 + kHW * 5] = 0.000001f;kaiserWindow[7 + kHW * 6] = 0.000001f;kaiserWindow[7 + kHW * 7] = 0.000001f;
    }
    else if(kHW == 4){
        kaiserWindow[0 + kHW * 0]=0.0001f;kaiserWindow[0 + kHW * 1]=0.0001f; kaiserWindow[0 + kHW * 2]=0.1000f;
        kaiserWindow[1 + kHW * 0]=0.0001f;kaiserWindow[1 + kHW * 1]=0.0001f;kaiserWindow[1 + kHW * 2]=0.0001f;
        kaiserWindow[2 + kHW * 0]=0.1000f;kaiserWindow[2 + kHW * 1]=0.0001f;kaiserWindow[2 + kHW * 2]=0.0001f;
        kaiserWindow[3 + kHW * 0]=0.1000f;kaiserWindow[3 + kHW * 1]=0.1000f;kaiserWindow[3 + kHW * 2]=0.0001f;


        kaiserWindow[0 + kHW * 3]=0.1000f;
        kaiserWindow[1 + kHW * 3]=0.1000f;
        kaiserWindow[2 + kHW * 3]=0.0001f;
        kaiserWindow[3 + kHW * 3]=0.0001f;

    }
    else if (kHW == 12)
    {
        kaiserWindow[0 + kHW * 0] = 0.0000001f; kaiserWindow[0 + kHW * 1] = 0.0000001f; kaiserWindow[0 + kHW * 2] = 0.0000001f; kaiserWindow[0 + kHW * 3] = 0.0000001f; kaiserWindow[0 + kHW * 4] = 0.0000001f;  kaiserWindow[0 + kHW * 5] = 0.0000001f; kaiserWindow[0 + kHW * 6] = 0.0098f;kaiserWindow[0 + kHW * 7] = 0.0098f;kaiserWindow[0 + kHW * 8] = 0.0098f;kaiserWindow[0 + kHW * 9] = 0.0098f;kaiserWindow[0 + kHW * 10] = 0.0098f;kaiserWindow[0 + kHW * 11] =  0.0098f;
        kaiserWindow[1 + kHW * 0] = 0.0000001f; kaiserWindow[1 + kHW * 1] = 0.0000001f; kaiserWindow[1 + kHW * 2] = 0.0000001f; kaiserWindow[1 + kHW * 3] = 0.0000001f; kaiserWindow[1 + kHW * 4] = 0.0000001f;  kaiserWindow[1 + kHW * 5] = 0.0098f; kaiserWindow[1 + kHW * 6] = 0.0098f;kaiserWindow[1 + kHW * 7] = 0.0098f;kaiserWindow[1 + kHW * 8] = 0.0098f;kaiserWindow[1 + kHW * 9] = 0.0098f;kaiserWindow[1 + kHW * 10] = 0.0098f;kaiserWindow[1 + kHW * 11] =  0.0098f;
        kaiserWindow[2 + kHW * 0] = 0.0000001f; kaiserWindow[2 + kHW * 1] = 0.0000001f; kaiserWindow[2 + kHW * 2] = 0.0000001f; kaiserWindow[2 + kHW * 3] = 0.0000001f; kaiserWindow[2 + kHW * 4] = 0.0098f;  kaiserWindow[2 + kHW * 5] = 0.0098f; kaiserWindow[2 + kHW * 6] = 0.0098f;kaiserWindow[2 + kHW * 7] = 0.0098f;kaiserWindow[2 + kHW * 8] = 0.0098f;kaiserWindow[2 + kHW * 9] = 0.0098f;kaiserWindow[2 + kHW * 10] = 0.0098f;kaiserWindow[2 + kHW * 11] =  0.0098f;
        kaiserWindow[3 + kHW * 0] = 0.0000001f; kaiserWindow[3 + kHW * 1] = 0.0000001f; kaiserWindow[3 + kHW * 2] = 0.0000001f; kaiserWindow[3 + kHW * 3] = 0.0098f; kaiserWindow[3 + kHW * 4] = 0.0098f;  kaiserWindow[3 + kHW * 5] = 0.0098f; kaiserWindow[3 + kHW * 6] = 0.0098f;kaiserWindow[3 + kHW * 7] = 0.0098f;kaiserWindow[3 + kHW * 8] = 0.0098f;kaiserWindow[3 + kHW * 9] = 0.0098f;kaiserWindow[3 + kHW * 10] = 0.0098f;kaiserWindow[3 + kHW * 11] =  0.0098f;
        kaiserWindow[4 + kHW * 0] = 0.0000001f; kaiserWindow[4 + kHW * 1] = 0.0000001f; kaiserWindow[4 + kHW * 2] = 0.0098f; kaiserWindow[4 + kHW * 3] = 0.0098f; kaiserWindow[4 + kHW * 4] = 0.0098f;  kaiserWindow[4 + kHW * 5] = 0.0098f; kaiserWindow[4 + kHW * 6] = 0.0098f;kaiserWindow[4 + kHW * 7] = 0.0098f;kaiserWindow[4 + kHW * 8] = 0.0098f;kaiserWindow[4 + kHW * 9] = 0.0098f;kaiserWindow[4 + kHW * 10] = 0.0098f;kaiserWindow[4 + kHW * 11] =  0.0098f;
        kaiserWindow[5 + kHW * 0] = 0.0000001f; kaiserWindow[5 + kHW * 1] = 0.0098f; kaiserWindow[5 + kHW * 2] = 0.0098f; kaiserWindow[5 + kHW * 3] = 0.0098f; kaiserWindow[5 + kHW * 4] = 0.0098f;  kaiserWindow[5 + kHW * 5] = 0.0098f; kaiserWindow[5 + kHW * 6] = 0.0098f;kaiserWindow[5 + kHW * 7] = 0.0098f;kaiserWindow[5 + kHW * 8] = 0.0098f;kaiserWindow[5 + kHW * 9] = 0.0098f;kaiserWindow[5 + kHW * 10] = 0.0098f;kaiserWindow[5 + kHW * 11] =  0.0098f;
        kaiserWindow[6 + kHW * 0] = 0.0098f; kaiserWindow[6 + kHW * 1] = 0.0098f; kaiserWindow[6 + kHW * 2] = 0.0098f; kaiserWindow[6 + kHW * 3] = 0.0098f; kaiserWindow[6 + kHW * 4] = 0.0098f;  kaiserWindow[6 + kHW * 5] = 0.0098f; kaiserWindow[6 + kHW * 6] = 0.0098f;kaiserWindow[6 + kHW * 7] = 0.0098f;kaiserWindow[6 + kHW * 8] = 0.0098f;kaiserWindow[6 + kHW * 9] = 0.0098f;kaiserWindow[6 + kHW * 10] = 0.0098f;kaiserWindow[6 + kHW * 11] = 0.0000001f;
        kaiserWindow[7 + kHW * 0] = 0.0098f; kaiserWindow[7 + kHW * 1] = 0.0098f; kaiserWindow[7 + kHW * 2] = 0.0098f; kaiserWindow[7 + kHW * 3] = 0.0098f; kaiserWindow[7 + kHW * 4] = 0.0098f;  kaiserWindow[7 + kHW * 5] = 0.0098f; kaiserWindow[7 + kHW * 6] = 0.0098f;kaiserWindow[7 + kHW * 7] = 0.0098f;kaiserWindow[7 + kHW * 8] = 0.0098f;kaiserWindow[7 + kHW * 9] = 0.0098f;kaiserWindow[7 + kHW * 10] = 0.0000001f;kaiserWindow[7 + kHW * 11] = 0.0000001f;
        kaiserWindow[8 + kHW * 0] = 0.0098f; kaiserWindow[8 + kHW * 1] = 0.0098f; kaiserWindow[8 + kHW * 2] = 0.0000001f; kaiserWindow[8 + kHW * 3] = 0.0098f; kaiserWindow[8 + kHW * 4] = 0.0098f;  kaiserWindow[8 + kHW * 5] = 0.0098f; kaiserWindow[8 + kHW * 6] = 0.0098f;kaiserWindow[8 + kHW * 7] = 0.0098f;kaiserWindow[8 + kHW * 8] = 0.0098f;kaiserWindow[8 + kHW * 9] = 0.0000001f;kaiserWindow[8 + kHW * 10] = 0.0000001f;kaiserWindow[8 + kHW * 11] = 0.0000001f;
        kaiserWindow[9 + kHW * 0] = 0.0098f; kaiserWindow[9 + kHW * 1] = 0.0098f; kaiserWindow[9 + kHW * 2] = 0.0098f; kaiserWindow[9 + kHW * 3] = 0.0098f; kaiserWindow[9 + kHW * 4] = 0.0098f;  kaiserWindow[9 + kHW * 5] = 0.0098f; kaiserWindow[9 + kHW * 6] = 0.0098f;kaiserWindow[9 + kHW * 7] = 0.0098f;kaiserWindow[9 + kHW * 8] = 0.0000001f;kaiserWindow[9 + kHW * 9] = 0.0000001f;kaiserWindow[9 + kHW * 10] = 0.0000001f;kaiserWindow[9 + kHW * 11] = 0.0000001f;
        kaiserWindow[10 + kHW * 0] = 0.0098f; kaiserWindow[10 + kHW * 1] = 0.0098f; kaiserWindow[10 + kHW * 2] = 0.0098f; kaiserWindow[10 + kHW * 3] = 0.0098f; kaiserWindow[10 + kHW * 4] = 0.0098f;  kaiserWindow[10 + kHW * 5] = 0.0098f; kaiserWindow[10 + kHW * 6] = 0.0098f;kaiserWindow[10 + kHW * 7] = 0.0000001f;kaiserWindow[10 + kHW * 8] = 0.0000001f;kaiserWindow[10 + kHW * 9] = 0.0000001f;kaiserWindow[10 + kHW * 10] = 0.0000001f;kaiserWindow[10 + kHW * 11] = 0.0000001f;
        kaiserWindow[11 + kHW * 0] = 0.0098f; kaiserWindow[11 + kHW * 1] = 0.0098f; kaiserWindow[11 + kHW * 2] = 0.0098f; kaiserWindow[11 + kHW * 3] = 0.0098f; kaiserWindow[11 + kHW * 4] = 0.0098f;  kaiserWindow[11 + kHW * 5] = 0.0098f; kaiserWindow[11 + kHW * 6] = 0.0000001f;kaiserWindow[11 + kHW * 7] = 0.0000001f;kaiserWindow[11 + kHW * 8] = 0.0000001f;kaiserWindow[11 + kHW * 9] = 0.0000001f;kaiserWindow[11 + kHW * 10] = 0.0000001f;kaiserWindow[11 + kHW * 11] = 0.0000001f;
    }
    else
        for (unsigned k = 0; k < kHW * kHW; k++)
         kaiserWindow[k] = 1.0f;

 }

 else{
    //circle  kernel
    if (kHW == 8){
             kaiserWindow[0 + kHW * 0] = 0.0001f; kaiserWindow[0 + kHW * 1] = 0.0001f; kaiserWindow[0 + kHW * 2] = 0.0001f; kaiserWindow[0 + kHW * 3] = 0.0001f;
        kaiserWindow[1 + kHW * 0] = 0.0001f; kaiserWindow[1 + kHW * 1] = 0.0001f; kaiserWindow[1 + kHW * 2] = 0.0001f; kaiserWindow[1 + kHW * 3] = 0.0417f;
        kaiserWindow[2 + kHW * 0] = 0.0001f; kaiserWindow[2 + kHW * 1] = 0.0001f; kaiserWindow[2 + kHW * 2] = 0.0417f; kaiserWindow[2 + kHW * 3] = 0.0417f;
        kaiserWindow[3 + kHW * 0] = 0.0001f; kaiserWindow[3 + kHW * 1] = 0.0417f; kaiserWindow[3 + kHW * 2] = 0.0417f; kaiserWindow[3 + kHW * 3] = 0.0417f; 
        kaiserWindow[4 + kHW * 0] = 0.0001f; kaiserWindow[4 + kHW * 1] = 0.0417f; kaiserWindow[4 + kHW * 2] = 0.0417f; kaiserWindow[4 + kHW * 3] = 0.0417f;
        kaiserWindow[5 + kHW * 0] = 0.0001f; kaiserWindow[5 + kHW * 1] = 0.0001f; kaiserWindow[5 + kHW * 2] = 0.0417f; kaiserWindow[5 + kHW * 3] = 0.0417f;
        kaiserWindow[6 + kHW * 0] = 0.0001f; kaiserWindow[6 + kHW * 1] = 0.0001f; kaiserWindow[6 + kHW * 2] = 0.0001f; kaiserWindow[6 + kHW * 3] = 0.0417f;
        kaiserWindow[7 + kHW * 0] = 0.0001f; kaiserWindow[7 + kHW * 1] = 0.0001f; kaiserWindow[7 + kHW * 2] = 0.0001f; kaiserWindow[7 + kHW * 3] = 0.0001f;

        kaiserWindow[0 + kHW * 4] = 0.0001f;kaiserWindow[0 + kHW * 5] = 0.0001f;kaiserWindow[0 + kHW * 6] = 0.0001f;kaiserWindow[0 + kHW * 7] = 0.0001f;
        kaiserWindow[1 + kHW * 4] = 0.0417f;kaiserWindow[1 + kHW * 5] = 0.0001f;kaiserWindow[1 + kHW * 6] = 0.0001f;kaiserWindow[1 + kHW * 7] = 0.0001f;
        kaiserWindow[2 + kHW * 4] = 0.0417f;kaiserWindow[2 + kHW * 5] = 0.0417f;kaiserWindow[2 + kHW * 6] = 0.0001f;kaiserWindow[2 + kHW * 7] =0.0001f;
        kaiserWindow[3 + kHW * 4] = 0.0417f;kaiserWindow[3 + kHW * 5] = 0.0417f;kaiserWindow[3 + kHW * 6] = 0.0417f;kaiserWindow[3 + kHW * 7] = 0.0001f; 
        kaiserWindow[4 + kHW * 4] = 0.0417f;kaiserWindow[4 + kHW * 5] = 0.0417f;kaiserWindow[4 + kHW * 6] = 0.0417f;kaiserWindow[4 + kHW * 7] = 0.0001f;
        kaiserWindow[5 + kHW * 4] = 0.0417f;kaiserWindow[5 + kHW * 5] = 0.0417f;kaiserWindow[5 + kHW * 6] = 0.0001f;kaiserWindow[5 + kHW * 7] = 0.0001f;
        kaiserWindow[6 + kHW * 4] = 0.0417f;kaiserWindow[6 + kHW * 5] = 0.0001f;kaiserWindow[6 + kHW * 6] = 0.0001f; kaiserWindow[6 + kHW * 7] = 0.0001f;
        kaiserWindow[7 + kHW * 4] = 0.0001f;kaiserWindow[7 + kHW * 5] = 0.0001f;; kaiserWindow[7 + kHW * 6] = 0.0001f;kaiserWindow[7 + kHW * 7] =0.0001f;
    }
    else if(kHW == 4){
        //Not correct for a circle
        kaiserWindow[0 + kHW * 0]=0.0001f;kaiserWindow[0 + kHW * 1]=0.0001f; kaiserWindow[0 + kHW * 2]=0.0001f;
        kaiserWindow[1 + kHW * 0]=0.1250f;kaiserWindow[1 + kHW * 1]=0.1250f;kaiserWindow[1 + kHW * 2]=0.1250f;
        kaiserWindow[2 + kHW * 0]=0.1250f;kaiserWindow[2 + kHW * 1]=0.1250f;kaiserWindow[2 + kHW * 2]=0.1250f;
        kaiserWindow[3 + kHW * 0]=0.0001f;kaiserWindow[3 + kHW * 1]=0.0001f;kaiserWindow[3 + kHW * 2]=0.0001f;


        kaiserWindow[0 + kHW * 3]=0.0001f;
        kaiserWindow[1 + kHW * 3]=0.1250f;
        kaiserWindow[2 + kHW * 3]=0.1250f;
        kaiserWindow[3 + kHW * 3]=0.0001f;
    }
    else if (kHW == 12)
    {
    kaiserWindow[0 + kHW * 0] = 0.0001f;kaiserWindow[0 + kHW * 1] = 0.0001f;kaiserWindow[0 + kHW * 2] = 0.0001f;kaiserWindow[0 + kHW * 3] = 0.0001f;
kaiserWindow[1 + kHW * 0] = 0.0001f;kaiserWindow[1 + kHW * 1] = 0.0001f;kaiserWindow[1 + kHW * 2] = 0.0001f;kaiserWindow[1 + kHW * 3] = 0.0001f;
kaiserWindow[2 + kHW * 0] = 0.0001f;kaiserWindow[2 + kHW * 1] = 0.0001f;kaiserWindow[2 + kHW * 2] = 0.0001f;kaiserWindow[2 + kHW * 3] = 0.0001f;
kaiserWindow[3 + kHW * 0] = 0.0001f;kaiserWindow[3 + kHW * 1] = 0.0001f;kaiserWindow[3 + kHW * 2] = 0.0001f;kaiserWindow[3 + kHW * 3] = 0.0455f;
kaiserWindow[4 + kHW * 0] = 0.0001f;kaiserWindow[4 + kHW * 1] = 0.0001f;kaiserWindow[4 + kHW * 2] = 0.0455f;kaiserWindow[4 + kHW * 3] = 0.0455f;
kaiserWindow[5 + kHW * 0] = 0.0001f;kaiserWindow[5 + kHW * 1] = 0.0455f;kaiserWindow[5 + kHW * 2] = 0.0455f;kaiserWindow[5 + kHW * 3] = 0.0455f;
kaiserWindow[6 + kHW * 0] = 0.0001f;kaiserWindow[6 + kHW * 1] = 0.0455f;kaiserWindow[6 + kHW * 2] = 0.0455f;kaiserWindow[6 + kHW * 3] = 0.0455f;
kaiserWindow[7 + kHW * 0] = 0.0001f;kaiserWindow[7 + kHW * 1] = 0.0001f;kaiserWindow[7 + kHW * 2] = 0.0455f;kaiserWindow[7 + kHW * 3] = 0.0455f;
kaiserWindow[8 + kHW * 0] = 0.0001f;kaiserWindow[8 + kHW * 1] = 0.0001f;kaiserWindow[8 + kHW * 2] = 0.0001f;kaiserWindow[8 + kHW * 3] = 0.0455f;
kaiserWindow[9 + kHW * 0] = 0.0001f;kaiserWindow[9 + kHW * 1] = 0.0001f;kaiserWindow[9 + kHW * 2] = 0.0001f;kaiserWindow[9 + kHW * 3] = 0.0001f;
kaiserWindow[10 + kHW * 0] = 0.0001f;kaiserWindow[10 + kHW * 1] = 0.0001f;kaiserWindow[10 + kHW * 2] = 0.0001f;kaiserWindow[10 + kHW * 3] = 0.0001f;
kaiserWindow[11 + kHW * 0] = 0.0001f;kaiserWindow[11 + kHW * 1] = 0.0001f;kaiserWindow[11 + kHW * 2] = 0.0001f;kaiserWindow[11 + kHW * 3] = 0.0001f;

kaiserWindow[0 + kHW * 4] = 0.0001f;kaiserWindow[0 + kHW * 5] = 0.0001f;kaiserWindow[0 + kHW * 6] = 0.0001f;kaiserWindow[0 + kHW * 7] = 0.0001f;
kaiserWindow[1 + kHW * 4] = 0.0001f;kaiserWindow[1 + kHW * 5] = 0.0455f;kaiserWindow[1 + kHW * 6] = 0.0455f;kaiserWindow[1 + kHW * 7] = 0.0001f;
kaiserWindow[2 + kHW * 4] = 0.0455f;kaiserWindow[2 + kHW * 5] = 0.0455f;kaiserWindow[2 + kHW * 6] = 0.0455f;kaiserWindow[2 + kHW * 7] = 0.0455f;
kaiserWindow[3 + kHW * 4] = 0.0455f;kaiserWindow[3 + kHW * 5] = 0.0455f;kaiserWindow[3 + kHW * 6] = 0.0455f;kaiserWindow[3 + kHW * 7] = 0.0455f;
kaiserWindow[4 + kHW * 4] = 0.0455f;kaiserWindow[4 + kHW * 5] = 0.0455f;kaiserWindow[4 + kHW * 6] = 0.0455f;kaiserWindow[4 + kHW * 7] = 0.0455f;
kaiserWindow[5 + kHW * 4] = 0.0455f;kaiserWindow[5 + kHW * 5] = 0.0455f;kaiserWindow[5 + kHW * 6] = 0.0455f;kaiserWindow[5 + kHW * 7] = 0.0455f;
kaiserWindow[6 + kHW * 4] = 0.0455f;kaiserWindow[6 + kHW * 5] = 0.0455f;kaiserWindow[6 + kHW * 6] = 0.0455f;kaiserWindow[6 + kHW * 7] = 0.0455f;
kaiserWindow[7 + kHW * 4] = 0.0455f;kaiserWindow[7 + kHW * 5] = 0.0455f;kaiserWindow[7 + kHW * 6] = 0.0455f;kaiserWindow[7 + kHW * 7] = 0.0455f;
kaiserWindow[8 + kHW * 4] = 0.0455f;kaiserWindow[8 + kHW * 5] = 0.0455f;kaiserWindow[8 + kHW * 6] = 0.0455f;kaiserWindow[8 + kHW * 7] = 0.0455f;
kaiserWindow[9 + kHW * 4] = 0.0455f;kaiserWindow[9 + kHW * 5] = 0.0455f;kaiserWindow[9 + kHW * 6] = 0.0455f;kaiserWindow[9 + kHW * 7] = 0.0455f;
kaiserWindow[10 + kHW * 4] = 0.0001f;kaiserWindow[10 + kHW * 5] = 0.0455f;kaiserWindow[10 + kHW * 6] = 0.0455f;kaiserWindow[10 + kHW * 7] = 0.0001f;
kaiserWindow[11 + kHW * 4] = 0.0001f;kaiserWindow[11 + kHW * 5] = 0.0001f;kaiserWindow[11 + kHW * 6] = 0.0001f;kaiserWindow[11 + kHW * 7] = 0.0001f;

kaiserWindow[0 + kHW * 8] = 0.0001f;kaiserWindow[0 + kHW * 9] = 0.0001f;kaiserWindow[0 + kHW * 10] = 0.0001f;kaiserWindow[0 + kHW * 11] = 0.0001f;
kaiserWindow[1 + kHW * 8] = 0.0001f;kaiserWindow[1 + kHW * 9] = 0.0001f;kaiserWindow[1 + kHW * 10] = 0.0001f;kaiserWindow[1 + kHW * 11] = 0.0001f;
kaiserWindow[2 + kHW * 8] = 0.0001f;kaiserWindow[2 + kHW * 9] = 0.0001f;kaiserWindow[2 + kHW * 10] = 0.0001f;kaiserWindow[2 + kHW * 11] = 0.0001f;
kaiserWindow[3 + kHW * 8] = 0.0455f;kaiserWindow[3 + kHW * 9] = 0.0001f;kaiserWindow[3 + kHW * 10] = 0.0001f;kaiserWindow[3 + kHW * 11] = 0.0001f;
kaiserWindow[4 + kHW * 8] = 0.0455f;kaiserWindow[4 + kHW * 9] = 0.0455f;kaiserWindow[4 + kHW * 10] = 0.0001f;kaiserWindow[4 + kHW * 11] = 0.0001f;
kaiserWindow[5 + kHW * 8] = 0.0455f;kaiserWindow[5 + kHW * 9] = 0.0455f;kaiserWindow[5 + kHW * 10] = 0.0455f;kaiserWindow[5 + kHW * 11] = 0.0001f;
kaiserWindow[6 + kHW * 8] = 0.0455f;kaiserWindow[6 + kHW * 9] = 0.0455f;kaiserWindow[6 + kHW * 10] = 0.0455f;kaiserWindow[6 + kHW * 11] = 0.0001f;
kaiserWindow[7 + kHW * 8] = 0.0455f;kaiserWindow[7 + kHW * 9] = 0.0455f;kaiserWindow[7 + kHW * 10] = 0.0001f;kaiserWindow[7 + kHW * 11] = 0.0001f;
kaiserWindow[8 + kHW * 8] = 0.0455f;kaiserWindow[8 + kHW * 9] = 0.0001f;kaiserWindow[8 + kHW * 10] = 0.0001f;kaiserWindow[8 + kHW * 11] = 0.0001f;
kaiserWindow[9 + kHW * 8] = 0.0001f;kaiserWindow[9 + kHW * 9] = 0.0001f;kaiserWindow[9 + kHW * 10] = 0.0001f;kaiserWindow[9 + kHW * 11] = 0.0001f;
kaiserWindow[10 + kHW * 8] = 0.0001f;kaiserWindow[10 + kHW * 9] = 0.0001f;kaiserWindow[10 + kHW * 10] = 0.0001f;kaiserWindow[10 + kHW * 11] = 0.0001f;
kaiserWindow[11 + kHW * 8] = 0.0001f;kaiserWindow[11 + kHW * 9] = 0.0001f;kaiserWindow[11 + kHW * 10] = 0.0001f;kaiserWindow[11 + kHW * 11] = 0.0001f;
 }
 else
    for (unsigned k = 0; k < kHW * kHW; k++)
     kaiserWindow[k] = 1.0f;
}
 

// ! Coefficient of normalization for DCT II and DCT II inverse
 const float coef = 0.5f / ((float) (kHW));
 for (unsigned i = 0; i < kHW; i++)
    for (unsigned j = 0; j < kHW; j++)
    {
        if (i == 0 && j == 0)
        {
            coef_norm    [i * kHW + j] = 0.5f * coef;
            coef_norm_inv[i * kHW + j] = 2.0f;
        }
        else if (i * j == 0)
        {
            coef_norm    [i * kHW + j] = SQRT2_INV * coef;
            coef_norm_inv[i * kHW + j] = SQRT2;
        }
        else
        {
            coef_norm    [i * kHW + j] = 1.0f * coef;
            coef_norm_inv[i * kHW + j] = 1.0f;
        }
    }
}


/**
 * @brief Precompute Bloc Matching (distance inter-patches)
 *
 * @param patch_table: for each patch in the image, will contain
 * all coordonnate of its similar patches
 * @param img: noisy image on which the distance is computed
 * @param width, height: size of img
 * @param kHW: size of patch
 * @param NHW: maximum similar patches wanted
 * @param nHW: size of the boundary of img
 * @param tauMatch: threshold used to determinate similarity between
 *        patches
 *
 * @return none.
 **/
void precompute_BM(
    vector<vector<unsigned> > &patch_table
    ,   const vector<float> &img
    ,   const unsigned width
    ,   const unsigned height
    ,   const unsigned kHW
    ,   const unsigned NHW
    ,   const unsigned nHW
    ,   const unsigned pHW
    ,   const float    tauMatch
    ){
    //! Declarations
    const unsigned Ns = 2 * nHW + 1;
    const float threshold = tauMatch * kHW * kHW;
    vector<float> diff_table(width * height);
    vector<vector<float> > sum_table((nHW + 1) * Ns, vector<float> (width * height, 2 * threshold));
    if (patch_table.size() != width * height)
        patch_table.resize(width * height);
    vector<unsigned> row_ind;
    ind_initialize(row_ind, height - kHW + 1, nHW, pHW);
    vector<unsigned> column_ind;
    ind_initialize(column_ind, width - kHW + 1, nHW, pHW);

    //! For each possible distance, precompute inter-patches distance
    for (unsigned di = 0; di <= nHW; di++)
        for (unsigned dj = 0; dj < Ns; dj++)
        {
            const int dk = (int) (di * width + dj) - (int) nHW;
            const unsigned ddk = di * Ns + dj;

            //! Process the image containing the square distance between pixels
            for (unsigned i = nHW; i < height - nHW; i++)
            {
                unsigned k = i * width + nHW;
                for (unsigned j = nHW; j < width - nHW; j++, k++)
                    diff_table[k] = (img[k + dk] - img[k]) * (img[k + dk] - img[k]);
            }

            //! Compute the sum for each patches, using the method of the integral images
            const unsigned dn = nHW * width + nHW;
            //! 1st patch, top left corner
            float value = 0.0f;
            for (unsigned p = 0; p < kHW; p++)
            {
                unsigned pq = p * width + dn;
                for (unsigned q = 0; q < kHW; q++, pq++)
                    value += diff_table[pq];
            }
            sum_table[ddk][dn] = value;

            //! 1st row, top
            for (unsigned j = nHW + 1; j < width - nHW; j++)
            {
                const unsigned ind = nHW * width + j - 1;
                float sum = sum_table[ddk][ind];
                for (unsigned p = 0; p < kHW; p++)
                    sum += diff_table[ind + p * width + kHW] - diff_table[ind + p * width];
                sum_table[ddk][ind + 1] = sum;
            }

            //! General case
            for (unsigned i = nHW + 1; i < height - nHW; i++)
            {
                const unsigned ind = (i - 1) * width + nHW;
                float sum = sum_table[ddk][ind];
                //! 1st column, left
                for (unsigned q = 0; q < kHW; q++)
                    sum += diff_table[ind + kHW * width + q] - diff_table[ind + q];
                sum_table[ddk][ind + width] = sum;

                //! Other columns
                unsigned k = i * width + nHW + 1;
                unsigned pq = (i + kHW - 1) * width + kHW - 1 + nHW + 1;
                for (unsigned j = nHW + 1; j < width - nHW; j++, k++, pq++)
                {
                    sum_table[ddk][k] =
                    sum_table[ddk][k - 1]
                    + sum_table[ddk][k - width]
                    - sum_table[ddk][k - 1 - width]
                    + diff_table[pq]
                    - diff_table[pq - kHW]
                    - diff_table[pq - kHW * width]
                    + diff_table[pq - kHW - kHW * width];
                }

            }
        }

    //! Precompute Bloc Matching
        vector<pair<float, unsigned> > table_distance;
    //! To avoid reallocation
        table_distance.reserve(Ns * Ns);

        for (unsigned ind_i = 0; ind_i < row_ind.size(); ind_i++)
        {
            for (unsigned ind_j = 0; ind_j < column_ind.size(); ind_j++)
            {
            //! Initialization
                const unsigned k_r = row_ind[ind_i] * width + column_ind[ind_j];
                table_distance.clear();
                patch_table[k_r].clear();

            //! Threshold distances in order to keep similar patches
                for (int dj = -(int) nHW; dj <= (int) nHW; dj++)
                {
                    for (int di = 0; di <= (int) nHW; di++)
                        if (sum_table[dj + nHW + di * Ns][k_r] < threshold)
                            table_distance.push_back(make_pair(
                                sum_table[dj + nHW + di * Ns][k_r]
                                , k_r + di * width + dj));

                        for (int di = - (int) nHW; di < 0; di++)
                            if (sum_table[-dj + nHW + (-di) * Ns][k_r] < threshold)
                                table_distance.push_back(make_pair(
                                    sum_table[-dj + nHW + (-di) * Ns][k_r + di * width + dj]
                                    , k_r + di * width + dj));
                        }

            //! We need a power of 2 for the number of similar patches,
            //! because of the Welsh-Hadamard transform on the third dimension.
            //! We assume that NHW is already a power of 2
                        const unsigned nSx_r = (NHW > table_distance.size() ?
                            closest_power_of_2(table_distance.size()) : NHW);

			//! To avoid problem
                        if (nSx_r == 1 && table_distance.size() == 0)
                        {
                            cout << "problem size" << endl;
                            table_distance.push_back(make_pair(0, k_r));
                        }

            //! Sort patches according to their distance to the reference one
                        partial_sort(table_distance.begin(), table_distance.begin() + nSx_r,
                            table_distance.end(), ComparaisonFirst);

            //! Keep a maximum of NHW similar patches
                        for (unsigned n = 0; n < nSx_r; n++)
                            patch_table[k_r].push_back(table_distance[n].second);

			//! To avoid problem
                        if (nSx_r == 1)
                            patch_table[k_r].push_back(table_distance[0].second);
                    }
                }
            }

/**
 * @brief Process of a weight dependent on the standard
 *        deviation, used during the weighted aggregation.
 *
 * @param group_3D : 3D group
 * @param nSx_r : number of similar patches in the 3D group
 * @param kHW: size of patches
 * @param chnls: number of channels in the image
 * @param weight_table: will contain the weighting for each
 *        channel.
 *
 * @return none.
 **/
            void sd_weighting(
                std::vector<float> const& group_3D
                ,   const unsigned nSx_r
                ,   const unsigned kHW
                ,   const unsigned chnls
                ,   std::vector<float> &weight_table
                ){
                const unsigned N = nSx_r * kHW * kHW;

                for (unsigned c = 0; c < chnls; c++)
                {
        //! Initialization
                    float mean = 0.0f;
                    float std  = 0.0f;

        //! Compute the sum and the square sum
                    for (unsigned k = 0; k < N; k++)
                    {
                        mean += group_3D[k];
                        std  += group_3D[k] * group_3D[k];
                    }

        //! Sample standard deviation (Bessel's correction)
                    float res = (std - mean * mean / (float) N) / (float) (N - 1);

        //! Return the weight as used in the aggregation
                    weight_table[c] = (res > 0.0f ? 1.0f / sqrtf(res) : 0.0f);
                }
            }






