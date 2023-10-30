
module TPU(
    clk,
    rst_n,

    in_valid,
    K,
    M,
    N,
    busy,

    A_wr_en,
    A_index,
    A_data_in,
    A_data_out,

    B_wr_en,
    B_index,
    B_data_in,
    B_data_out,

    C_wr_en,
    C_index,
    C_data_in,
    C_data_out
);


input clk;
input rst_n;
input            in_valid;
input [7:0]      K;
input [7:0]      M;
input [7:0]      N;
output  reg      busy;

output           A_wr_en;
output reg [15:0]    A_index;
output reg [31:0]    A_data_in;
input  [31:0]    A_data_out;

output           B_wr_en;
output reg [15:0]    B_index;
output reg [31:0]    B_data_in;
input  [31:0]    B_data_out;

output reg           C_wr_en;
output reg [15:0]    C_index;
output reg [127:0]   C_data_in;
input  [127:0]   C_data_out;



//* Implement your design here
reg [1:0] mode;  //4 modes: (2*2, 2*2), (4*4, 4*4), (4*k,k*4), (M*k, k*N)
reg [7:0] K_in, M_in, N_in;

reg [7:0] idx;
reg [31:0] A_buffer;
reg [31:0] B_buffer;
reg [127:0] C_Matrix[0:4];

reg [15:0] cycle_cnt; //not sure the size


always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        cycle_cnt <= 'd0;
    else if (busy)
        cycle_cnt <= cycle_cnt + 'd1;
    else
        cycle_cnt <= 'd0;
end

/* Input KMN */
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        K_in <= 'd0;
        M_in <= 'd0;
        N_in <= 'd0;
    end
    else if (in_valid) begin
        K_in <= K;
        M_in <= M;
        N_in <= N; 
    end
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) 
        mode <= 'd0;
    else if (in_valid) begin
        if (K == 2 && M == 2 && N == 2)
            mode <= 'd0;
        else if (K == 4 && M == 4 && N == 4)
            mode <= 'd1;
        else if (M == 4 && N == 4)
            mode <= 'd2;
        else
            mode <= 'd3;
    end
end


/* Get A,B Matrix */
integer i;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        A_index <= 'd0;
    else if (busy && cycle_cnt < 1 && mode == 'd0)
        A_index <= A_index + 1;
    else if (busy && cycle_cnt < 3 && mode == 'd1)
        A_index <= A_index + 1;
    else if (busy && cycle_cnt < (K_in -1) && mode == 'd2)
        A_index <= A_index + 1;
    else 
        A_index <= 'd0;
end


always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        B_index <= 'd0;
    else if (busy && cycle_cnt < 1 && mode == 'd0)
        B_index <= B_index + 1;
    else if (busy && cycle_cnt < 3 && mode == 'd1)
        B_index <= B_index + 1;
    else if (busy && cycle_cnt < (K_in - 1) && mode == 'd2)
        B_index <= B_index + 1;
    else
        B_index <= 'd0;
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        idx <= 'd0;
    else if (busy && cycle_cnt < 1 && mode == 'd0)      //need reset to 0
        idx <= idx + 'd1;
    else
        idx <= 'd0;
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        A_buffer <= 'd0;
    else 
        A_buffer <= A_data_out;    
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        B_buffer <= 'd0;
    else
        B_buffer <= B_data_out;    
end


/* PEs Calculate */
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        for (i = 0; i < 4; i = i+1)
            C_Matrix[i] <= 'd0;
    end
    else if (cycle_cnt >= 1 && cycle_cnt <= 2 && mode == 'd0) begin
        C_Matrix[0] <= {C_Matrix[0][127:96] + A_buffer[31:24]*B_buffer[31:24],
                        C_Matrix[0][95:64]  + A_buffer[31:24]*B_buffer[23:16], {64{1'b0}}};
        C_Matrix[1] <= {C_Matrix[1][127:96] + A_buffer[23:16]*B_buffer[31:24],
                        C_Matrix[1][95:64]  + A_buffer[23:16]*B_buffer[23:16], {64{1'b0}}};
    end
    else if (cycle_cnt >= 1 && cycle_cnt <= K_in && (mode == 'd1 || mode == 'd2)) begin
        C_Matrix[0] <= {C_Matrix[0][127:96] + A_buffer[31:24]*B_buffer[31:24],
                        C_Matrix[0][95:64]  + A_buffer[31:24]*B_buffer[23:16],
                        C_Matrix[0][63:32]  + A_buffer[31:24]*B_buffer[15:8],
                        C_Matrix[0][31:0]   + A_buffer[31:24]*B_buffer[7:0]};

        C_Matrix[1] <= {C_Matrix[1][127:96] + A_buffer[23:16]*B_buffer[31:24],
                        C_Matrix[1][95:64]  + A_buffer[23:16]*B_buffer[23:16],
                        C_Matrix[1][63:32]  + A_buffer[23:16]*B_buffer[15:8],
                        C_Matrix[1][31:0]   + A_buffer[23:16]*B_buffer[7:0]};

        C_Matrix[2] <= {C_Matrix[2][127:96] + A_buffer[15:8]*B_buffer[31:24],
                        C_Matrix[2][95:64]  + A_buffer[15:8]*B_buffer[23:16],
                        C_Matrix[2][63:32]  + A_buffer[15:8]*B_buffer[15:8],
                        C_Matrix[2][31:0]   + A_buffer[15:8]*B_buffer[7:0]};

        C_Matrix[3] <= {C_Matrix[3][127:96] + A_buffer[7:0]*B_buffer[31:24],
                        C_Matrix[3][95:64]  + A_buffer[7:0]*B_buffer[23:16],
                        C_Matrix[3][63:32]  + A_buffer[7:0]*B_buffer[15:8],
                        C_Matrix[3][31:0]   + A_buffer[7:0]*B_buffer[7:0]};                
    end
    else if (!busy) begin
        for (i = 0; i < 4; i = i+1)
            C_Matrix[i] <= 'd0;
    end
end

/* output control */
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        busy <= 0;
    end
    else if (in_valid)
        busy <= 1;
    else if (cycle_cnt == 5 && mode == 'd0)
        busy <= 0;
    else if (cycle_cnt == 9 && mode == 'd1)
        busy <= 0;
    else if (cycle_cnt == (K_in + 5) && mode == 'd2)
        busy <= 0;
end

always @(posedge clk or negedge rst_n) begin 
    if (!rst_n)
        C_wr_en  <= 'd0;
    else if (cycle_cnt >= 3 && cycle_cnt <= 4 && mode == 'd0)
        C_wr_en <= 'd1;
    else if (cycle_cnt >= 5 && cycle_cnt <= 8 && mode == 'd1)
        C_wr_en <= 'd1;
    else if (cycle_cnt >= (K_in + 1) && cycle_cnt <= (K_in + 4) && mode == 'd2)
        C_wr_en <= 'd1;
    else 
        C_wr_en <= 'd0;
end

always @(posedge clk or negedge rst_n) begin 
    if (!rst_n)
        C_index  <= 'd0;
    else if (cycle_cnt == 4 && mode == 'd0)
        C_index <= C_index + 1;
    else if (cycle_cnt >= 6 && cycle_cnt <= 8 && mode == 'd1)
        C_index <= C_index + 1;
    else if (cycle_cnt >= (K_in + 2) && cycle_cnt <= (K_in + 4) && mode == 'd2)
        C_index <= C_index + 1;
    else 
        C_index <= 'd0;
end

always @(posedge clk or negedge rst_n) begin
    if (!rst_n)
        C_data_in  <= 'd0;
    else if (cycle_cnt >= 3 && cycle_cnt <= 4 && mode == 'd0)
        C_data_in <= C_Matrix[cycle_cnt - 3];
    else if (cycle_cnt >= 5 && cycle_cnt <= 8 && mode == 'd1)
        C_data_in <= C_Matrix[cycle_cnt - 5];
    else if (cycle_cnt >= (K_in + 1) && cycle_cnt <= (K_in + 4) && mode == 'd2)
        C_data_in <= C_Matrix[cycle_cnt - (K_in + 1)];
    else 
        C_data_in <= 'd0;
end

endmodule