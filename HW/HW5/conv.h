/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_

#include <algorithm>
#include "cfu.h"
#include "models/my_cycles.h"
#include "perf.h"
#include "playground_util/print_params.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

extern long long unsigned my_cycles;

namespace tflite {
namespace reference_integer_ops {

// Fixed-point per-channel-quantization convolution reference kernel.
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const int32_t* bias_data, const RuntimeShape& output_shape,
    int8_t* output_data) {

  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  // const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  // const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  

  /* Modify below to im2col */  //need concern stride
  for (int batch = 0; batch < batches; ++batch){
    // Store filter
    int32_t filter_matrix[300][300];  // w*h*c, batch
    // int8_t filter_matrix[300][300];  // w*h*c, batch
    
    int row_idx_filter = 0;
    int col_idx_filter = 0;
    // int filter_size_x = filter_input_depth * filter_height * filter_width;
    // printf("\n ************ Start filter batch ********************\n");
    for (int out_channel = 0; out_channel < output_depth; ++out_channel){
        for (int in_channel = 0; in_channel < filter_input_depth; ++in_channel){
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                    filter_matrix[row_idx_filter][col_idx_filter] = filter_data[Offset(filter_shape, out_channel, filter_y, filter_x, in_channel)];
                row_idx_filter++;
                }
            }
        }
        row_idx_filter = 0;
        col_idx_filter++;
    }
    // Store input 
    printf("\n ************ Store Filter Done ********************\n");
    col_idx_filter = 0;
    int32_t input_matrix[900][300];  //#filter*batch, w*h*c
    int32_t input_matrix_for_HW[900][300];  //#filter*batch, w*h*c
    int input_inside_image_map_HW[900][300];
    // int8_t input_matrix[900][300];  //#filter*batch, w*h*c
    
        for (int idx_y = 0; idx_y < output_height; ++idx_y) {
            const int in_y_origin = (idx_y * stride_height) - pad_height;
            for (int idx_x = 0; idx_x < output_width; ++idx_x) {
                const int in_x_origin = (idx_x * stride_width) - pad_width;

                for (int in_channel = 0; in_channel < input_depth; ++in_channel){

                    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
                        const int in_y = in_y_origin + dilation_height_factor * filter_y;
                        for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                            const int in_x = in_x_origin + dilation_width_factor * filter_x;

                            const bool is_point_inside_image = (in_x >= 0) && (in_x < input_width) && (in_y >= 0) && (in_y < input_height);
                            
                            int32_t input_val = 0;
                            if (is_point_inside_image) {
                                input_val = input_data[Offset(input_shape, batch, in_y, in_x, in_channel)] + input_offset;     
                            }
                            input_matrix[row_idx_filter][col_idx_filter] = input_val;
                            
                            if (is_point_inside_image){
                                input_matrix_for_HW[row_idx_filter][col_idx_filter] = input_data[Offset(input_shape, batch, in_y, in_x, in_channel)];
                                input_inside_image_map_HW[row_idx_filter][col_idx_filter] = 1;
                            }
                            else {
                                input_matrix_for_HW[row_idx_filter][col_idx_filter] = 0;
                                input_inside_image_map_HW[row_idx_filter][col_idx_filter] = 0;
                            }

                            col_idx_filter++;
                        }
                    }
                }
                row_idx_filter++;
                col_idx_filter = 0;
            }
        }
        printf("\n ************ Store input Done ********************\n");



        //Compute output
        int32_t output_matrix[300][900];
        int32_t HW_ans_matrix[300][900];
        int32_t HW_ans_x = 0;
        int32_t HW_ans_y = 0;
        

        for (int out_y = 0; out_y < output_depth; ++out_y){
            for (int out_x = 0; out_x < output_width * output_height; ++out_x){
                output_matrix[out_y][out_x] = 0;
                for (int idx = 0; idx < filter_height * filter_width * filter_input_depth; ++idx){
                    output_matrix[out_y][out_x] += input_matrix[out_x][idx] * filter_matrix[idx][out_y];
                }
            }
        }

        /* Move matrix to hardware*/
        unsigned my_start = perf_get_mcycle();
        // Store filter matrix
        for (int j = 0; j < output_depth; j=j+4) {
            for (int i = 0; i < filter_height * filter_width * filter_input_depth; i++) {
                cfu_op1(0, filter_matrix[i][j], filter_matrix[i][j+1]);
                // printf("\nfilter_matrix[%d][%d] = %ld, filter_matrix[%d][%d] = %ld", i,j,filter_matrix[i][j], i, j+1, filter_matrix[i][j+1]);
                // printf("\nfilter_matrix[%d][%d] = %ld, filter_matrix[%d][%d] = %ld", i,j,filter_matrix[i][j], i, j+1, filter_matrix[i][j+1]);
                printf(" ");
                // printf("\n ================== CFU op1 Done %d =====================\n", i);
                cfu_op1(0, filter_matrix[i][j+2], filter_matrix[i][j+3]);
                // printf("filter_matrix[%d][%d] = %ld, filter_matrix[%d][%d] = %ld\n", i,j+2,filter_matrix[i][j+2], i, j+3, filter_matrix[i][j+3]);
                // printf("filter_matrix[%d][%d] = %ld, filter_matrix[%d][%d] = %ld\n", i,j+2,filter_matrix[i][j+2], i, j+3, filter_matrix[i][j+3]);
                printf(" ");
                // printf("\n ================== CFU op1 Done %d =====================\n", i);
                /*For Debug*/
                // int8_t val = cfu_op5(0,0,0);
                // if (val != filter_matrix[i][j])
                //     printf("\n!!!! Filter matrix store Gbuff_B Wrong !!!!, cfu_op5 = %d, filter_matrix = %ld\n", val, filter_matrix[i][j]);
                
                // val = cfu_op5(0,0,0);
                // if (val != filter_matrix[i][j+1])
                //     printf("\n!!!! Filter matrix store Gbuff_B Wrong !!!!, cfu_op5 = %d, filter_matrix = %ld\n", val, filter_matrix[i][j+1]);
                
                // val = cfu_op5(0,0,0);
                // if (val != filter_matrix[i][j+2])
                //     printf("\n!!!! Filter matrix store Gbuff_B Wrong !!!!, cfu_op5 = %d, filter_matrix = %ld\n", val, filter_matrix[i][j+2]);
                
                // val = cfu_op5(0,0,0);
                // if (val != filter_matrix[i][j+3])
                //     printf("\n!!!! Filter matrix store Gbuff_B Wrong !!!!, cfu_op5 = %d, filter_matrix = %ld\n", val, filter_matrix[i][j+3]);
                // else
                //     printf("\n ++++ Filter matrix stroe Correct ++++\n");
                /*Debug end*/
            }
            // printf("\n ================== CFU op1 Done %d / %d (output_depth) =====================\n", j, output_depth);
        // Store transopse filter matrix
            for (int k = 0; k < output_height * output_width; k = k+4) {
                int inner_compute_flag = 0;
                for (int r = 0; r < filter_height * filter_width * filter_input_depth; r++) {
                    // printf("\n ================== CFU op0 Start %d / %d (filter_height * filter_width * filter_input_depth) =====================\n", r, filter_height * filter_width * filter_input_depth);
                    if (k+1 == output_height * output_width) {
                        if (input_inside_image_map_HW[k][r] == 1) {
                            cfu_op0(1, input_matrix_for_HW[k][r], 0);
                            printf(".");
                        }else { 
                            cfu_op0(0, input_matrix_for_HW[k][r], 0);
                            printf(".");
                        }
                        
                        // printf("\n ================== CFU op0 Done %d =====================\n", k);
                        cfu_op0(0, 0, 0);
                        printf(".");
                    }
                    else if (k+2 == output_height * output_width) {
                        if (input_inside_image_map_HW[k][r] == 0 && input_inside_image_map_HW[k+1][r] == 0){
                            cfu_op0(0, input_matrix_for_HW[k][r], input_matrix_for_HW[k+1][r]);
                            printf(".");
                        }else if (input_inside_image_map_HW[k][r] == 1 && input_inside_image_map_HW[k+1][r] == 0){
                            cfu_op0(1, input_matrix_for_HW[k][r], input_matrix_for_HW[k+1][r]);
                            printf(".");
                        }else if (input_inside_image_map_HW[k][r] == 0 && input_inside_image_map_HW[k+1][r] == 1){
                            cfu_op0(2, input_matrix_for_HW[k][r], input_matrix_for_HW[k+1][r]);
                            printf(".");
                        }else if (input_inside_image_map_HW[k][r] == 1 && input_inside_image_map_HW[k+1][r] == 1){
                            cfu_op0(3, input_matrix_for_HW[k][r], input_matrix_for_HW[k+1][r]);
                            printf(".");
                        }
                        
                        // printf("\n ================== CFU op0 Done %d =====================\n", k);
                        cfu_op0(0, 0, 0);
                        printf(".");
                    }
                    else if (k+3 == output_height * output_width) {
                        if (input_inside_image_map_HW[k][r] == 0 && input_inside_image_map_HW[k+1][r] == 0){
                            cfu_op0(0, input_matrix_for_HW[k][r], input_matrix_for_HW[k+1][r]);
                            // printf(".");
                        }else if (input_inside_image_map_HW[k][r] == 1 && input_inside_image_map_HW[k+1][r] == 0){
                            cfu_op0(1, input_matrix_for_HW[k][r], input_matrix_for_HW[k+1][r]);
                            // printf(".");
                        }else if (input_inside_image_map_HW[k][r] == 0 && input_inside_image_map_HW[k+1][r] == 1){
                            cfu_op0(2, input_matrix_for_HW[k][r], input_matrix_for_HW[k+1][r]);
                            // printf(".");
                        }else if (input_inside_image_map_HW[k][r] == 1 && input_inside_image_map_HW[k+1][r] == 1){
                            cfu_op0(3, input_matrix_for_HW[k][r], input_matrix_for_HW[k+1][r]);
                            // printf(".");
                        }
                        // printf("\n ================== CFU op0 Done %d =====================\n", k);
                        if (input_inside_image_map_HW[k+2][r] == 0){
                            cfu_op0(0, input_matrix_for_HW[k+2][r], 0);
                            // printf(".");
                        }
                        else {
                            cfu_op0(1, input_matrix_for_HW[k+2][r], 0);
                            // printf(".");
                        }
                    }
                    else {
                        if (input_inside_image_map_HW[k][r] == 0 && input_inside_image_map_HW[k+1][r] == 0){
                            cfu_op0(0, input_matrix_for_HW[k][r], input_matrix_for_HW[k+1][r]);
                            // printf("\ninput_matrix[%d][%d] = %ld, input_matrix[%d][%d] = %ld ", k, r, input_matrix_for_HW[k][r], k+1, r, input_matrix_for_HW[k+1][r]);
                            // printf("\ninput_matrix[%d][%d] = %ld, input_matrix[%d][%d] = %ld ", k, r, input_matrix_for_HW[k][r], k+1, r, input_matrix_for_HW[k+1][r]);
                            printf(".");
                        }else if (input_inside_image_map_HW[k][r] == 1 && input_inside_image_map_HW[k+1][r] == 0){
                            cfu_op0(1, input_matrix_for_HW[k][r], input_matrix_for_HW[k+1][r]);
                            // printf("\ninput_matrix[%d][%d] = %ld, input_matrix[%d][%d] = %ld ", k, r, input_matrix_for_HW[k][r]+input_offset, k+1, r, input_matrix_for_HW[k+1][r]);
                            // printf("\ninput_matrix[%d][%d] = %ld, input_matrix[%d][%d] = %ld ", k, r, input_matrix_for_HW[k][r]+input_offset, k+1, r, input_matrix_for_HW[k+1][r]);
                            printf(".");
                        }else if (input_inside_image_map_HW[k][r] == 0 && input_inside_image_map_HW[k+1][r] == 1){
                            cfu_op0(2, input_matrix_for_HW[k][r], input_matrix_for_HW[k+1][r]);
                            // printf("\ninput_matrix[%d][%d] = %ld, input_matrix[%d][%d] = %ld ", k, r, input_matrix_for_HW[k][r], k+1, r, input_matrix_for_HW[k+1][r]+input_offset);
                            // printf("\ninput_matrix[%d][%d] = %ld, input_matrix[%d][%d] = %ld ", k, r, input_matrix_for_HW[k][r], k+1, r, input_matrix_for_HW[k+1][r]+input_offset);
                            printf(".");
                        }else if (input_inside_image_map_HW[k][r] == 1 && input_inside_image_map_HW[k+1][r] == 1){
                            cfu_op0(3, input_matrix_for_HW[k][r], input_matrix_for_HW[k+1][r]);
                            // printf("\ninput_matrix[%d][%d] = %ld, input_matrix[%d][%d] = %ld ", k, r, input_matrix_for_HW[k][r]+input_offset, k+1, r, input_matrix_for_HW[k+1][r]+input_offset);
                            // printf("\ninput_matrix[%d][%d] = %ld, input_matrix[%d][%d] = %ld ", k, r, input_matrix_for_HW[k][r]+input_offset, k+1, r, input_matrix_for_HW[k+1][r]+input_offset);
                            printf(".");
                        }

                        // printf("\n ================== CFU op0 Done %d =====================\n", k);
                        if (input_inside_image_map_HW[k+2][r] == 0 && input_inside_image_map_HW[k+3][r] == 0){
                            cfu_op0(0, input_matrix_for_HW[k+2][r], input_matrix_for_HW[k+3][r]);
                            // printf("input_matrix[%d][%d] = %ld, input_matrix[%d][%d] = %ld", k+2, r, input_matrix_for_HW[k+2][r], k+3, r, input_matrix_for_HW[k+3][r]);
                            // printf("input_matrix[%d][%d] = %ld, input_matrix[%d][%d] = %ld", k+2, r, input_matrix_for_HW[k+2][r], k+3, r, input_matrix_for_HW[k+3][r]);
                            printf(".");
                        }else if (input_inside_image_map_HW[k+2][r] == 1 && input_inside_image_map_HW[k+3][r] == 0){
                            cfu_op0(1, input_matrix_for_HW[k+2][r], input_matrix_for_HW[k+3][r]);
                            // printf("input_matrix[%d][%d] = %ld, input_matrix[%d][%d] = %ld", k+2, r, input_matrix_for_HW[k+2][r]+input_offset, k+3, r, input_matrix_for_HW[k+3][r]);
                            // printf("input_matrix[%d][%d] = %ld, input_matrix[%d][%d] = %ld", k+2, r, input_matrix_for_HW[k+2][r]+input_offset, k+3, r, input_matrix_for_HW[k+3][r]);
                            printf(".");
                        }else if (input_inside_image_map_HW[k+2][r] == 0 && input_inside_image_map_HW[k+3][r] == 1){
                            cfu_op0(2, input_matrix_for_HW[k+2][r], input_matrix_for_HW[k+3][r]);
                            // printf("input_matrix[%d][%d] = %ld, input_matrix[%d][%d] = %ld", k+2, r, input_matrix_for_HW[k+2][r], k+3, r, input_matrix_for_HW[k+3][r]+input_offset);
                            // printf("input_matrix[%d][%d] = %ld, input_matrix[%d][%d] = %ld", k+2, r, input_matrix_for_HW[k+2][r], k+3, r, input_matrix_for_HW[k+3][r]+input_offset);
                            printf(".");
                        }else if (input_inside_image_map_HW[k+2][r] == 1 && input_inside_image_map_HW[k+3][r] == 1){
                            cfu_op0(3, input_matrix_for_HW[k+2][r], input_matrix_for_HW[k+3][r]);
                            // printf("input_matrix[%d][%d] = %ld, input_matrix[%d][%d] = %ld", k+2, r, input_matrix_for_HW[k+2][r]+input_offset, k+3, r, input_matrix_for_HW[k+3][r]+input_offset);
                            // printf("input_matrix[%d][%d] = %ld, input_matrix[%d][%d] = %ld", k+2, r, input_matrix_for_HW[k+2][r]+input_offset, k+3, r, input_matrix_for_HW[k+3][r]+input_offset);
                            printf(".");
                        }  
                    }

                    /*For debug*/
                    // int8_t val = cfu_op4(0,0,0);
                    // int flag = 0;
                    // if (val != input_matrix_for_HW[k][r])
                    //     printf("\n !!!! Input matrix stroe Gbuff_A Wrong !!!!, k = %d, r = %d, cfu_op4 = %d, input_matrix_for_HW = %ld\n", k, r, val, input_matrix[k][r]);
                    
                    // val = cfu_op4(0,0,0);
                    // if (val != input_matrix_for_HW[k+1][r] && (k+1 != output_height * output_width))
                    //     printf("\n !!!! Input matrix stroe Gbuff_A Wrong !!!!, k = %d, r = %d, cfu_op4 = %d, input_matrix_for_HW = %ld\n", k+1, r, val, input_matrix[k+1][r]);
                    // else if (k+1 == output_height * output_width)
                    //     flag = 1;

                    // val = cfu_op4(0,0,0);
                    // if (val != input_matrix_for_HW[k+2][r] && (k+2 != output_height * output_width) && flag == 0)
                    //     printf("\n !!!! Input matrix stroe Gbuff_A Wrong !!!!, k = %d, r = %d, cfu_op4 = %d, input_matrix_for_HW = %ld\n", k+2, r, val, input_matrix[k+2][r]);
                    // else if (k+2 == output_height * output_width)
                    //     flag = 1;

                    // val = cfu_op4(0,0,0);
                    // if (val != input_matrix_for_HW[k+3][r] && (k+3 != output_height * output_width) && flag == 0)
                    //     printf("\n !!!! Input matrix stroe Gbuff_A Wrong !!!!, k = %d, r = %d, cfu_op4 = %d, input_matrix_for_HW = %ld\n", k+3, r, val, input_matrix[k+3][r]);
                    
                    // else
                    //     printf("\n ++++ Input matrix store Correct ++++\n");
                    /*End debug*/
                }
                // printf("\n ========== Finish cfu_op0&2 %d / %d (output_height * output_width)iterations ========== \n", k, output_height * output_width);
                // Compute here, and assign value to output matrix
                // Send K_in to HW, (M*K*N)
                if (inner_compute_flag == 0) {
                    cfu_op2(0, filter_height * filter_width * filter_input_depth, input_offset); //Start compute trigger!
                    printf(" ");
                }
                // printf("\n ========== Finish cfu_op2 %d / %d (output_height * output_width)iterations ========== \n", k, output_height * output_width);
                // May have zero padding (check)
                for (int x = 0; x < 4; x++) {
                    for (int y = 0; y < 4; y++){
                        HW_ans_matrix[y + HW_ans_y][x + HW_ans_x] = cfu_op3(0,0,0); // Just get the return value
                        printf(" ");
                        if (output_matrix[y + HW_ans_y][x + HW_ans_x] != HW_ans_matrix[y + HW_ans_y][x + HW_ans_x]){
                            printf(" ");
                            // printf("\n !!!! Miss Match !!!!, x = %ld, y = %ld, HW = %ld, Ans = %ld\n",y + HW_ans_y, x + HW_ans_x, HW_ans_matrix[y + HW_ans_y][x + HW_ans_x], output_matrix[y + HW_ans_y][x + HW_ans_x]);
                            // printf("\n !!!! Miss Match !!!!, x = %ld, y = %ld, HW = %ld, Ans = %ld\n",y + HW_ans_y, x + HW_ans_x, HW_ans_matrix[y + HW_ans_y][x + HW_ans_x], output_matrix[y + HW_ans_y][x + HW_ans_x]);
                        
                        }
                        // else
                            // printf ("\n !!!! Answer Match !!!!\n");
                    }
                }
                HW_ans_x = HW_ans_x + 4;
                // printf("\n ========== Finish %d / %d (output_height * output_width)iterations ========== \n", k, output_height * output_width);
            }
            HW_ans_x = 0;
            HW_ans_y = HW_ans_y + 4;
            // printf("\n ========== Finish %d / %d (output depth)iterations ========== \n", j, output_depth);
        }
        printf("\n ************ HW matrix computation Done ********************\n");
        unsigned my_finish = perf_get_mcycle();
        my_cycles += (my_finish - my_start);

        // Checking answer
        for (int x = 0; x < output_depth; x++) {
            for (int y = 0; y < output_width * output_height; y++) {
                if (output_matrix[x][y] != HW_ans_matrix[x][y]) {
                    printf("\nAnswer Not Equal x = %d, y = %d, output_matrix = %ld, HW_ans_matrix = %ld\n", x, y, output_matrix[x][y], HW_ans_matrix[x][y]);
                    break;
                }
            }
        }

        /**/
        

        // output write back
        for (int out_channel = 0; out_channel < output_depth; ++out_channel){
            for (int out_y = 0; out_y < output_height; ++out_y){
                for (int out_x = 0; out_x < output_width; ++out_x){
                    const int idx = out_y * output_width + out_x;
                    // int32_t acc = output_matrix[out_channel][idx];
                    int32_t acc = HW_ans_matrix[out_channel][idx];
                    
                    if(bias_data){
                        acc += bias_data[out_channel];
                    }
                    acc = MultiplyByQuantizedMultiplier(acc, output_multiplier[out_channel], output_shift[out_channel]);
                    acc += output_offset;
                    acc = std::max(acc, output_activation_min);
                    acc = std::min(acc, output_activation_max);
                    output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] = static_cast<int8_t>(acc);
                }
            }
        }
        printf("\n ************ One Batch Done ********************\n");
    }
}

inline void ConvPerChannelWithPackedInt4Weights(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int8_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_input, int8_t* unpacked_filter_data,
    const RuntimeShape& bias_shape, const int32_t* bias_data,
    const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK(unpacked_filter_data != nullptr);
  tflite::tensor_utils::UnpackDenseInt4IntoInt8(
      filter_input, filter_shape.FlatSize(), unpacked_filter_data);
  ConvPerChannel(params, output_multiplier, output_shift, input_shape,
                 input_data, filter_shape, unpacked_filter_data, bias_shape,
                 bias_data, output_shape, output_data);
}

// Fixed-point per-channel-quantization convolution reference kernel.
// 16-bit data and 8-bit filter
template <typename AccumScalar>
inline void ConvPerChannel(
    const ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const RuntimeShape& input_shape,
    const int16_t* input_data, const RuntimeShape& filter_shape,
    const int8_t* filter_data, const RuntimeShape& bias_shape,
    const AccumScalar* bias_data, const RuntimeShape& output_shape,
    int16_t* output_data) {
  // Get parameters.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          AccumScalar acc = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;

              // Zero padding by omitting the areas outside the image.
              const bool is_point_inside_image =
                  (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                  (in_y < input_height);

              if (!is_point_inside_image) {
                continue;
              }

              for (int in_channel = 0; in_channel < filter_input_depth;
                   ++in_channel) {
                int32_t input_val =
                    input_data[Offset(input_shape, batch, in_y, in_x,
                                      in_channel + group * filter_input_depth)];
                int32_t filter_val = filter_data[Offset(
                    filter_shape, out_channel, filter_y, filter_x, in_channel)];
                // Accumulate with 64 bits accumulator.
                // int64_t += int8_t * int16_t so the highest value we can
                // get from each accumulation is [-127, 127] * ([-32768,
                // 32767] -
                // [-32768, 32767]), which is [-8322945, 8322945].
                // log2(8322945) = 22.99.
                acc += filter_val * input_val;
              }
            }
          }
          if (bias_data) {
            acc += bias_data[out_channel];
          }
          int32_t scaled_acc = MultiplyByQuantizedMultiplier(
              acc, output_multiplier[out_channel], output_shift[out_channel]);
          scaled_acc = std::max(scaled_acc, output_activation_min);
          scaled_acc = std::min(scaled_acc, output_activation_max);
          output_data[Offset(output_shape, batch, out_y, out_x, out_channel)] =
              static_cast<int16_t>(scaled_acc);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_CONV_H_
