module Cfu (
  input               cmd_valid,
  output              cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,
  input      [31:0]   cmd_payload_inputs_1,
  output reg          rsp_valid,
  input               rsp_ready,
  output reg [31:0]   rsp_payload_outputs_0,
  input               reset,
  input               clk
);
  /* SPEC: funct3 = 0, For write A buffer; funct3 = 1, For write B buffer;
           funct3 = 2, Start compute flag; funct3 = 3, Get results
           funct3 = 4, For debug A buffer; funct4 = 5, For debug B buffer
  */
  reg signed [7:0] data_in_0, data_in_1;
  reg [1:0] input_offset_enable;
  reg store_gbuff_A_enable, store_gbuff_B_enable; //C is triggered by TPU, (check)
  reg store_done_flag;

  /* For TPU */ 
  reg start_compute_flag;  // Map to busy signal
  reg comupte_done_flag;
  reg [8:0] K_in;
  reg signed [31:0] input_offset;
  reg [20:0] cycle_cnt;
  reg [15:0] A_index, B_index, C_index;  // for computation
  reg [15:0] A_index_dbg, B_index_dbg;  // for debug
  reg signed [31:0] C_Matrix[0:15];

  integer i;

  /* Handshake control start */
  assign cmd_ready = 1;
  always @(posedge clk) begin
    if (reset) begin  //reset is high active (check)
      //cmd input
      store_gbuff_A_enable <= 0;
      store_gbuff_B_enable <= 0;
      data_in_0 <= 'd0;
      data_in_1 <= 'd0;
      // Compute signal
      K_in <= 'd0;
      //response
      rsp_payload_outputs_0 <= 32'd0;
      rsp_valid <= 1'b0;
      C_index <= 'd0;
      // debug
      A_index_dbg <= 'd0;
      B_index_dbg <= 'd0;
    end else if (cmd_valid) begin
      if (cmd_payload_function_id[2:0] == 'd0) begin
        store_gbuff_A_enable <= 1;
        input_offset_enable <= cmd_payload_function_id[4:3];
        data_in_0 <= $signed(cmd_payload_inputs_0[7:0]);
        data_in_1 <= $signed(cmd_payload_inputs_1[7:0]);
      end
      else if (cmd_payload_function_id[2:0] == 'd1) begin
        store_gbuff_B_enable <= 1;
        data_in_0 <= $signed(cmd_payload_inputs_0[7:0]);
        data_in_1 <= $signed(cmd_payload_inputs_1[7:0]);
      end
      else if (cmd_payload_function_id[2:0] == 'd2) begin  // Start compute
        // start_compute_flag <= 1;
        K_in <= cmd_payload_inputs_0[8:0];
        input_offset <= cmd_payload_inputs_1;
      end
      else if (cmd_payload_function_id[2:0] == 'd3) begin
        rsp_valid <= 'd1;
        rsp_payload_outputs_0 <= C_Matrix[C_index];
        C_index <= C_index + 'd1;
      end
      else if (cmd_payload_function_id[2:0] == 'd4) begin
        rsp_valid <= 'd1;
        rsp_payload_outputs_0 <= {{24{gbuff_A[7]}}, gbuff_A[A_index_dbg]};  // sign extension
        A_index_dbg <= A_index_dbg + 'd1;
      end
      else if (cmd_payload_function_id[2:0] == 'd5) begin
        rsp_valid <= 'd1;
        rsp_payload_outputs_0 <= {{24{gbuff_B[7]}}, gbuff_B[B_index_dbg]};
        B_index_dbg <= B_index_dbg + 'd1;
      end
    end else if (store_done_flag) begin // (check), need a complete signal
      rsp_valid <= 1;
      store_gbuff_A_enable <= 'd0;
      store_gbuff_B_enable <= 'd0;
      rsp_payload_outputs_0 <= 'd0;
      C_index <= 'd0;
    end else if (comupte_done_flag) begin
      rsp_valid <= 1;
      A_index_dbg <= 'd0;
      B_index_dbg <= 'd0;
    end
    else begin
      rsp_valid <= 'd0;
      rsp_payload_outputs_0 <= 'd0;
      // Write done control
    end
  end

  always @(posedge clk) begin
    if (reset)
      store_done_flag <= 'd0;
    else if (cmd_valid && (cmd_payload_function_id[2:0] == 'd0 || cmd_payload_function_id[2:0] == 'd1))
      store_done_flag <= 'd1;
    else 
      store_done_flag <= 'd0;
  end
  /* Handshake control end */


  /* Global Buffer start */

  // Size limits to 1000000 bits
  // Global buffer A for Input matrix, ADDR_BITS=16, DATA_BITS=32
  parameter ADDR_BITS_A = 11;  //Up to 14
  parameter DATA_BITS_A = 8;
  parameter DEPTH_A = 1200;
  // parameter DEPTH_A = 1200;
  reg signed [DATA_BITS_A-1:0] gbuff_A [DEPTH_A-1:0];  //2048 is too large (check), 1200 is enough
  reg gbuff_offset_map [DEPTH_A-1:0];  //check
  reg [ADDR_BITS_A-1:0] index_A;
  always @ (posedge clk) begin
    if(reset) begin
      index_A <= 'd0;
      // for(i = 0; i < DEPTH_A; i = i+1)
      //   gbuff_A[i] <= 'd0;      
    end
    else begin
      if(store_gbuff_A_enable) begin  // (check)
        gbuff_A[index_A]   <= data_in_0;
        gbuff_A[index_A+1] <= data_in_1;

        if(input_offset_enable == 'd0) begin
          gbuff_offset_map[index_A]   <= 1'b0;
          gbuff_offset_map[index_A+1] <= 1'b0;
        end
        else if(input_offset_enable == 'd1) begin
          gbuff_offset_map[index_A]   <= 1'b1;
          gbuff_offset_map[index_A+1] <= 1'b0;
        end
        else if(input_offset_enable == 'd2) begin
          gbuff_offset_map[index_A]   <= 1'b0;
          gbuff_offset_map[index_A+1] <= 1'b1;
        end
        else if(input_offset_enable == 'd3) begin
          gbuff_offset_map[index_A]   <= 1'b1;
          gbuff_offset_map[index_A+1] <= 1'b1;
        end
          
        index_A <= index_A + 2;
      end
      else if(comupte_done_flag) begin
        index_A <= 'd0;
      end
    end
  end

  // Global buffer B for Filter matrix, ADDR_BITS=16, DATA_BITS=32
  parameter ADDR_BITS_B = 11; //Up to 14
  parameter DATA_BITS_B = 8;
  parameter DEPTH_B = 1200;  // 2048 is too large (check)
  reg signed [DATA_BITS_B-1:0] gbuff_B [DEPTH_B-1:0];
  reg [ADDR_BITS_B-1:0] index_B;
  always @ (posedge clk) begin
    if(reset) begin
      index_B <= 'd0;
      // for(i = 0; i < DEPTH_B; i = i+1) begin
      //   gbuff_B[i] <= 'd0;          
      // end
    end
    else begin
      if(store_gbuff_B_enable) begin // (check)
        gbuff_B[index_B]   <= data_in_0;
        gbuff_B[index_B+1] <= data_in_1;
        index_B <= index_B + 2;
      end
      else if(comupte_done_flag) begin  // (May Need modify)
        index_B <= 'd0;
      end
    end
  end

  /* Global Buffer end */

  /* TPU start */
  always @(posedge clk) begin //start_compte_flag map to busy signal
    if (reset)
      start_compute_flag <= 'd0;
    else if (cmd_valid && cmd_payload_function_id[2:0] == 'd2)
      start_compute_flag <= 'd1;
    else if (cycle_cnt == (K_in) + 1 + 1) //(check)
      start_compute_flag <= 'd0;
  end

  always @(posedge clk) begin
    if(reset)
      cycle_cnt <= 'd0;
    else if(start_compute_flag)
      cycle_cnt <= cycle_cnt + 'd1;
    else 
      cycle_cnt <= 'd0;
  end

  always @(posedge clk) begin
    if (reset)
      comupte_done_flag <= 'd0;
    else if (cycle_cnt == (K_in) + 1 + 1) //(check) cycle +1 +1
      comupte_done_flag <= 'd1;
    else 
      comupte_done_flag <= 'd0;
  end

  /* Get A,B Matrix*/
  always @ (posedge clk) begin
    if(reset)
      A_index <= 'd0;
    else if(start_compute_flag && cycle_cnt < (K_in - 1))
      A_index <= A_index + 'd4;
    else if (cmd_payload_function_id[2:0] == 'd3 && cmd_valid) //(check)
      A_index <= 'd0;
  end

  always @(posedge clk) begin
    if (reset)
      B_index <= 'd0;
    else if (start_compute_flag && cycle_cnt < (K_in -1))
      B_index <= B_index + 'd4;
    else if (cmd_payload_function_id[2:0] == 'd3 && cmd_valid) // Only after three times computation reset to zero
      B_index <= 'd0;
  end

  /* PEs Calculate */
  integer c;
  reg signed [31:0] pipeline_buffer[0:15];

  // Pipeline for multiply
  reg signed [31:0] tmp_gbuff_A_0, tmp_gbuff_A_1, tmp_gbuff_A_2, tmp_gbuff_A_3;
  reg signed [7:0] tmp_gbuff_B_0, tmp_gbuff_B_1, tmp_gbuff_B_2, tmp_gbuff_B_3;

  always @(posedge clk) begin
    if (reset) begin
      tmp_gbuff_A_0 <= 'd0;
      tmp_gbuff_A_1 <= 'd0;
      tmp_gbuff_A_2 <= 'd0;
      tmp_gbuff_A_3 <= 'd0;
    end
    else begin
      if (gbuff_offset_map[A_index] == 1'b1) begin
        tmp_gbuff_A_0 <= gbuff_A[A_index] + input_offset;
      end
      else begin
        tmp_gbuff_A_0 <= gbuff_A[A_index];
      end

      if (gbuff_offset_map[A_index + 1] == 1'b1) begin
        tmp_gbuff_A_1 <= gbuff_A[A_index + 1] + input_offset;
      end
      else begin
        tmp_gbuff_A_1 <= gbuff_A[A_index + 1];
      end

      if (gbuff_offset_map[A_index +2] == 1'b1) begin
        tmp_gbuff_A_2 <= gbuff_A[A_index + 2] + input_offset;
      end
      else begin
        tmp_gbuff_A_2 <= gbuff_A[A_index + 2];
      end

      if (gbuff_offset_map[A_index + 3] == 1'b1) begin
        tmp_gbuff_A_3 <= gbuff_A[A_index + 3] + input_offset;
      end
      else begin
        tmp_gbuff_A_3 <= gbuff_A[A_index + 3];
      end

    end
  end
  always @(posedge clk) begin
    if (reset) begin
      tmp_gbuff_B_0 <= 'd0;
      tmp_gbuff_B_1 <= 'd0;
      tmp_gbuff_B_2 <= 'd0;
      tmp_gbuff_B_3 <= 'd0;
    end
    else begin
      tmp_gbuff_B_0 <= gbuff_B[B_index];
      tmp_gbuff_B_1 <= gbuff_B[B_index + 1];
      tmp_gbuff_B_2 <= gbuff_B[B_index + 2];
      tmp_gbuff_B_3 <= gbuff_B[B_index + 3];
    end
  end

  always @(posedge clk) begin
    if (reset) begin
      for (c = 0; c < 16; c = c+1)
        pipeline_buffer[c] <= 'd0;
    end
    // else if (cycle_cnt >= 1 && cycle_cnt <= K_in) begin
    else if (cycle_cnt >= 1 && cycle_cnt <= K_in) begin //Add one cycle
      pipeline_buffer[0]  <= tmp_gbuff_A_0 * tmp_gbuff_B_0;
      pipeline_buffer[1]  <= tmp_gbuff_A_0 * tmp_gbuff_B_1;
      pipeline_buffer[2]  <= tmp_gbuff_A_0 * tmp_gbuff_B_2;
      pipeline_buffer[3]  <= tmp_gbuff_A_0 * tmp_gbuff_B_3;
      pipeline_buffer[4]  <= tmp_gbuff_A_1 * tmp_gbuff_B_0;
      pipeline_buffer[5]  <= tmp_gbuff_A_1 * tmp_gbuff_B_1;
      pipeline_buffer[6]  <= tmp_gbuff_A_1 * tmp_gbuff_B_2;
      pipeline_buffer[7]  <= tmp_gbuff_A_1 * tmp_gbuff_B_3;
      pipeline_buffer[8]  <= tmp_gbuff_A_2 * tmp_gbuff_B_0;
      pipeline_buffer[9]  <= tmp_gbuff_A_2 * tmp_gbuff_B_1;
      pipeline_buffer[10] <= tmp_gbuff_A_2 * tmp_gbuff_B_2;
      pipeline_buffer[11] <= tmp_gbuff_A_2 * tmp_gbuff_B_3;
      pipeline_buffer[12] <= tmp_gbuff_A_3 * tmp_gbuff_B_0;
      pipeline_buffer[13] <= tmp_gbuff_A_3 * tmp_gbuff_B_1;
      pipeline_buffer[14] <= tmp_gbuff_A_3 * tmp_gbuff_B_2;
      pipeline_buffer[15] <= tmp_gbuff_A_3 * tmp_gbuff_B_3;
    end
    else if ((cmd_payload_function_id[2:0] == 'd0 || cmd_payload_function_id[2:0] == 'd1) && cmd_valid) begin  //Wrong (check)
      for (c = 0; c < 16; c = c+1)
        pipeline_buffer[c] <= 'd0;
    end
  end

  always @(posedge clk) begin
    if (reset) begin
      for (c = 0; c < 16; c = c+1)
        C_Matrix[c] <= 'd0;
    end
    // else if (cycle_cnt >= 1+1 && cycle_cnt <= K_in + 1) begin // cycle + 1
    else if (cycle_cnt >= 1+1 && cycle_cnt <= K_in + 1) begin // cycle + 1 + 1
      C_Matrix[0]  <= C_Matrix[0]  + pipeline_buffer[0];
      C_Matrix[1]  <= C_Matrix[1]  + pipeline_buffer[1];
      C_Matrix[2]  <= C_Matrix[2]  + pipeline_buffer[2];
      C_Matrix[3]  <= C_Matrix[3]  + pipeline_buffer[3];
      C_Matrix[4]  <= C_Matrix[4]  + pipeline_buffer[4];
      C_Matrix[5]  <= C_Matrix[5]  + pipeline_buffer[5];
      C_Matrix[6]  <= C_Matrix[6]  + pipeline_buffer[6];
      C_Matrix[7]  <= C_Matrix[7]  + pipeline_buffer[7];
      C_Matrix[8]  <= C_Matrix[8]  + pipeline_buffer[8];
      C_Matrix[9]  <= C_Matrix[9]  + pipeline_buffer[9];
      C_Matrix[10] <= C_Matrix[10] + pipeline_buffer[10];
      C_Matrix[11] <= C_Matrix[11] + pipeline_buffer[11];
      C_Matrix[12] <= C_Matrix[12] + pipeline_buffer[12];
      C_Matrix[13] <= C_Matrix[13] + pipeline_buffer[13];
      C_Matrix[14] <= C_Matrix[14] + pipeline_buffer[14];
      C_Matrix[15] <= C_Matrix[15] + pipeline_buffer[15];
    end
    else if ((cmd_payload_function_id[2:0] == 'd0 || cmd_payload_function_id[2:0] == 'd1) && cmd_valid) begin  //Wrong (check) would be reset to zero
      for (c = 0; c < 16; c = c+1)
        C_Matrix[c] <= 'd0;
    end
  end


  /* TPU end */




endmodule