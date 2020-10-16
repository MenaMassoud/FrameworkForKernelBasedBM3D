
    //apply patchSNR on the basic Image
    //add boundary
   // cout<<img_basic_s.size()<<endl;
    // vector<float> img_basic2, img_basic_s2,img_basic_h2,
    // img_basic_v2,img_basic_D1352,img_basic_D452;
    //   symetrize(img_basic, img_basic2, width, height, chnls, 4);
    //   symetrize(img_basic_s, img_basic_s2, width, height, chnls, 4);
    //   symetrize(img_basic_h, img_basic_h2, width, height, chnls, 4);
    //   symetrize(img_basic_v, img_basic_v2, width, height, chnls, 4);
    //   symetrize(img_basic_D135, img_basic_D1352, width, height, chnls,4);
    //   symetrize(img_basic_D45, img_basic_D452, width, height, chnls, 4);

    // cout<<img_basic_s2.size()<<endl;

    //combine based on PSNR
// int p = 3;
// int patch_width = 7;
// int count = 0;

// float sum_basic = 0.0;
// float sum_s = 0.0;
// float sum_h = 0.0;
// float sum_v = 0.0;
// float sum_D135 = 0.0;
// float sum_D45 = 0.0;

// float avg_basic = 0.0;
// float avg_s = 0.0;
// float avg_h = 0.0;
// float avg_v = 0.0;
// float avg_D135 = 0.0;
// float avg_D45 = 0.0;

// float var_basic = 0.0;
// float var_s = 0.0;
// float var_h = 0.0;
// float var_v = 0.0;
// float var_D135 = 0.0;
// float var_D45 = 0.0;

// float patSNR_basic= 0.0;
// float patSNR_s= 0.0;
// float patSNR_h= 0.0;
// float patSNR_v= 0.0;
// float patSNR_D135= 0.0;
// float patSNR_D45= 0.0;

// int cmin = 0;
// int cmax = 0;
// int cs = 0;
// int cb = 0;
// int cd135 = 0;
// int cd45 = 0;
// int ch = 0;
// int cv = 0;

// for (int i = p ; i < width-p ; i++){
// for (int j = p ; j <height-p ; j++){
//      float patch_basic[patch_width*patch_width] = {}; 
//      float patch_s[patch_width*patch_width] = {}; 
//      float patch_h[patch_width*patch_width] = {}; 
//      float patch_v[patch_width*patch_width] = {}; 
//      float patch_D135[patch_width*patch_width] = {}; 
//      float patch_D45[patch_width*patch_width] = {}; 
//      count = 0;   
//         //generate the pixel neighbours
//         //int loc = i + j*patch_width;
//     for(int m = i-p ; m <= i+p ; m++){               
//         for (int n = j-p ; n <= j+p ; n++){
//                 //to get the pixel loction in the array
//                 //patch[count] = m+n*(width); //index
//             patch_basic[count] = img_basic[m+n*(width)];
//             patch_s[count] = img_basic_s[m+n*(width)];
//             patch_v[count] = img_basic_v[m+n*(width)];
//             patch_h[count] = img_basic_h[m+n*(width)];
//             patch_D135[count] = img_basic_D135[m+n*(width)];
//             patch_D45[count] = img_basic_D45[m+n*(width)];
//         //     cout<<patch_basic[count]<<" ";
//               count = count + 1;
             
//           }
//      //         cout << endl;
//            //   cout<<"count = "<<count<<endl;
//       }
//     //   cout << endl;
//     //   cout << endl;
//        // Find the sum 
//     sum_basic  =  std::accumulate(patch_basic,patch_basic + (patch_width*patch_width), 0.0);
//   //  cout<<"sum" << sum_basic<<endl;
//     sum_s  =  std::accumulate(patch_s,patch_s + (patch_width*patch_width), 0.0);
//     sum_h  =  std::accumulate(patch_h,patch_h + (patch_width*patch_width), 0.0);
//     sum_v  =  std::accumulate(patch_v,patch_v + (patch_width*patch_width), 0.0);
//     sum_D135  =  std::accumulate(patch_D135,patch_D135 + (patch_width*patch_width), 0.0);
//     sum_D45  =  std::accumulate(patch_D45,patch_D45 + (patch_width*patch_width), 0.0);
//    // cout<<sum_basic<<endl;
//      avg_basic = sum_basic / (float)(patch_width*patch_width);
//    //  cout<<"avg "<< avg_basic<<endl;
//      avg_s = sum_s / (float)(patch_width*patch_width);
//      avg_h = sum_h / (float)(patch_width*patch_width);
//      avg_v = sum_v / (float)(patch_width*patch_width);
//      avg_D135 = sum_D135 / (float)(patch_width*patch_width);
//      avg_D45 = sum_D45 / (float)(patch_width*patch_width);

//      for (int i = 0 ; i < (patch_width*patch_width) ; i++)  {
//       var_basic += (patch_basic[i] - avg_basic) * (patch_basic[i] - avg_basic); 
//       var_s += (patch_s[i] - avg_s) * (patch_s[i] - avg_s); 
//       var_h += (patch_h[i] - avg_h) * (patch_h[i] - avg_h); 
//       var_v += (patch_v[i] - avg_v) * (patch_v[i] - avg_v); 
//       var_D135 += (patch_D135[i] - avg_D135) * (patch_D135[i] - avg_D135); 
//       var_D45 += (patch_D45[i] - avg_D45) * (patch_D45[i] - avg_D45); 
//   }
//   // cout<<"var basic" << var_basic<<endl;
//    var_basic = var_basic / (float)(patch_width*patch_width);
//   // cout<<"var basic normalized" << var_basic<<endl;
//    var_s = var_s / (float)(patch_width*patch_width);
//    var_h = var_h / (float)(patch_width*patch_width);
//    var_v = var_v / (float)(patch_width*patch_width); 
//    var_D135 = var_D135 / (float)(patch_width*patch_width);
//    var_D45 = var_D45 / (float)(patch_width*patch_width);
//  // cout<<"varb = "<<var_basic<<" var_s = "<<var_s<<" varh = "<<var_h<<" varv = ";
//  // cout<<var_v<<" vard135 = "<<var_D135<<" varD45 = "<<var_D45<<endl;
  

//     ResPatSNR_basic[i+j*width] = sqrt(var_basic / (float)pow(sigma,2));
//     ResPatSNR_s[i+j*width] = sqrt(var_s / (float)pow(sigma,2));
//     ResPatSNR_h[i+j*width] = sqrt(var_h / (float)pow(sigma,2));
//     ResPatSNR_v[i+j*width] = sqrt(var_v / (float)pow(sigma,2));
//     ResPatSNR_D135[i+j*width] = sqrt(var_D135 / (float)pow(sigma,2));
//     ResPatSNR_D45[i+j*width] = sqrt(var_D45 / (float)pow(sigma,2));
//    // cout<<"res_b = "<<ResPatSNR_basic[i+j*width]<<" res_s = ";
//    // cout<<ResPatSNR_s[i+j*width]<<" res_h = "<<ResPatSNR_h[i+j*width];
//    // cout<<" res_v =  "<<ResPatSNR_v[i+j*width]<<" D135 = "<<ResPatSNR_D135[i+j*width];
//    // cout<<" D45 =  "<<ResPatSNR_D45[i+j*width]<<endl;


// }
// }
  
//     cout << "\nMin Element = "
//          << *min_element(ResPatSNR_basic.begin(), ResPatSNR_basic.end())<<endl; 
  
//     // Find the max element 
//     cout << "\nMax Element = "
//          << *max_element(ResPatSNR_basic.begin(),ResPatSNR_basic.end())<<endl; 
//          float max = *max_element(ResPatSNR_basic.begin(),ResPatSNR_basic.end());
// float t = (*min_element(ResPatSNR_basic.begin(), ResPatSNR_basic.end())+
// *max_element(ResPatSNR_basic.begin(), ResPatSNR_basic.end()))/2;

// //Compute Histogram
// int MAX_INTENSITY = 255;
// long int SizeImg = width * height ;
// int f = 0;
// unsigned hist[MAX_INTENSITY + 1];
// for(f = 0; f<=MAX_INTENSITY ; f++)hist[f] = 0;
// for(f = 0; f< SizeImg ; f++){
//     int value = (int)ResPatSNR_basic[f];
//     hist[value]++;
// }

// //compute threshold
// float sum = 0;
// float sumB = 0;
// int q1 =  0;
// int q2 = 0;
// float varMax = 0;
// float threshold;
// for(int f = 0; f<=MAX_INTENSITY; f++){
//     //update q1
//     q1 += hist[f];
//     if(q1 == 0)
//         continue;
//     //update q2
//     q2 = SizeImg - q1;
//     if(q2 == 0)
//         break;

//     sumB += (float)(f * ((int)hist[f]));
//     float m1 = sumB / q1;
//     float m2 = (sum - sumB) / q2;
// // Update the between class variance
// float varBetween = (float) q1 * (float) q2 * (m1 - m2) * (m1 - m2);
// // Update the threshold if necessary
//       if (varBetween > varMax) {
//         varMax = varBetween;
//         threshold = f;
//       }
// }
// cout<<"Threshold = "<< threshold <<endl;

// for(int x = p; x< width-p ; x++){
//     for(int y = p; y< height-p ; y++){

//  if(ResPatSNR_basic[x+y*width] < threshold){
//     std::vector<float> arr = {ResPatSNR_basic[x+y*width],ResPatSNR_s[x+y*width],
//     ResPatSNR_h[x+y*width],ResPatSNR_v[x+y*width],
//     ResPatSNR_D135[x+y*width], ResPatSNR_D45[x+y*width]};

//      int minElementIndex = std::min_element(arr.begin(),arr.end()) - arr.begin();
//     // cout<<"max ind " << maxElementIndex<<endl;
//      cmin = cmin + 1;
//     switch(minElementIndex){
//             case 0:
//             img_basic[x+y*width] = img_basic[x+y*width];
//             cb = cb + 1;
//             break;
//             case 1:
//             img_basic[x+y*width] = img_basic_s[x+y*width];
//             cs = cs + 1;
//             break;
//             case 2:
//             img_basic[x+y*width] = img_basic_h[x+y*width];
//             ch = ch + 1;
//             break;
//             case 3:
//             img_basic[x+y*width] = img_basic_v[x+y*width];
//             cv = cv + 1;
//             break;
//             case 4:
//             img_basic[x+y*width] = img_basic_D135[x+y*width];
//             cd135 = cd135+1;
//             break;
//             default:
//             img_basic[x+y*width] = img_basic_D45[x+y*width];
//             cd45 = cd45+1;
//          }

   //  float arr[] = {img_basic[x+y*width], img_basic_s[x+y*width],
   //  img_basic_h[x+y*width],img_basic_v[x+y*width],
   //  img_basic_D135[x+y*width], img_basic_D45[x+y*width]};
   //  int n = 6;
   //  sort(arr, arr+n); 
   //      // check for even case 
   //  if (n % 2 != 0)    
   //     img_basic[x+y*width] = (double)arr[n / 2]; 
   // else
   //     img_basic[x+y*width] =  (double)(arr[(n - 1) / 2] + arr[n / 2]) / 2.0; 
// }
// }
// } 
// if(compute_psnr(img, img_basic, &psnr_basic, &rmse_basic) != EXIT_SUCCESS)
//          cout<<"Error calculating psnr";

//     cout << "(basic image) :" << endl;
//     cout << "PSNR basic2: " << psnr_basic << endl;
//     cout << "RMSE basic2: " << rmse_basic << endl << endl;

//  cout<<"min "<<cmin<<endl;
//  cout<<"max "<<cmax<<endl;
//  cout<<"cb "<<cb << endl;
//  cout<<"cs "<<cs<<endl;
//  cout<<"ch "<<ch<<endl;
//  cout<<"cv "<<cv<<endl;
//  cout<<"cd135 "<<cd135<<endl;
//  cout<<"cd45 "<<cd45<<endl;
// if(compute_psnr(img, img_basic, &psnr_basic, &rmse_basic) != EXIT_SUCCESS)
//          cout<<"Error calculating psnr";

//     cout << "(basic image) :" << endl;
//     cout << "PSNR basic2: " << psnr_basic << endl;
//     cout << "RMSE basic2: " << rmse_basic << endl << endl;