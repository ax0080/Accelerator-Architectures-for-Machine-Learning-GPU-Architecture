
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
output [15:0]    A_index;  //* select this
output [31:0]    A_data_in;
input  [31:0]    A_data_out;  //* to get this in the next cycle

output           B_wr_en;
output [15:0]    B_index;  //* select this
output [31:0]    B_data_in;
input  [31:0]    B_data_out;  //* to get this in the next cycle

output           C_wr_en;
output [15:0]    C_index;    //* select this
output [127:0]   C_data_in;  //* to write the results to here in the same cycle
input  [127:0]   C_data_out;



//* Implement your design here

assign A_wr_en = 0;
assign B_wr_en = 0;
assign C_wr_en = 1;

reg [7:0] k;
reg [7:0] m;
reg [7:0] n;

output [7:0] cur_block_B;
output [31:0] delayed_A_data_out;
output [31:0] delayed_B_data_out;

output busy_c;
output block_over;  // for sysArr to tell controller that the current block pair is finished
output finish;  // for controller to tell sysArr that the data feeding is finished

controller controller(
    .clk (clk),
    .reset (rst_n),
    .block_over (block_over),
    .finish (finish),
    .K   (k),
    .M   (m),
    .N   (n),
    .cur_block_B (cur_block_B),
    .A_index (A_index),
    .B_index (B_index),
    .busy (busy_c)
);

buffer bufferA(
    .clk (clk),
    .reset (rst_n),
    .busy (busy),
    .block_over (block_over),
    .datain (A_data_out),
    .dataout (delayed_A_data_out)
);

buffer bufferB(
    .clk (clk),
    .reset (rst_n),
    .busy (busy),
    .block_over (block_over),
    .datain (B_data_out),
    .dataout (delayed_B_data_out)
);

sysArr sysArr(
    .clk (clk),
    .reset (rst_n),
    .block_over (block_over),
    .finish (finish),
    .K (k),
    .M (m),
    .cur_block_B (cur_block_B),
    .datain_h (delayed_A_data_out),
    .datain_v (delayed_B_data_out),
    .C_index (C_index),
    .C_data_in (C_data_in),
    .busy (busy_c)
);


initial begin
    busy = 0;
end

always @(negedge clk) begin
    busy <= busy_c;
end

always @(negedge clk) begin
    if(K > 0) begin
        k = K;
        m = M;
        n = N;
    end
end

endmodule


module controller(
    clk,
    reset,
    busy,
    block_over,
    finish,
    K,
    M,
    N,

    cur_block_B,
    A_index,
    B_index,
);
    input clk;
    input reset;
    input busy;
    input block_over;
    output reg finish;
    input [7:0] K;
    input [7:0] M;
    input [7:0] N;

    output [15:0] A_index;
    output [15:0] B_index;
    reg signed [15:0] count_A;
    reg signed [15:0] count_B;
    reg [7:0] num_block_A;
    reg [7:0] cur_block_A;
    reg [7:0] num_block_B;
    output reg [7:0] cur_block_B;
    
    assign A_index = count_A;
    assign B_index = count_B;

    always @(negedge reset or negedge busy) begin
        if(finish >= 0) begin
            repeat(10) @(negedge clk);
        end
        else begin
            repeat(3) @(negedge clk);
        end
        #1
        finish = 0;

        count_A = -1;
        count_B = -1;

        num_block_A = $ceil(M/4.0);
        cur_block_A = 0;

        num_block_B = $ceil(N/4.0);
        cur_block_B = 0;
    end

    always @(posedge block_over) begin
        if(cur_block_A < num_block_A - 1) begin
            cur_block_A += 1;

            count_B = cur_block_B*K-1;
            count_A = cur_block_A*K-1;
        end
        else if (cur_block_B < num_block_B - 1) begin
            cur_block_A = 0;
            cur_block_B += 1;

            count_B = cur_block_B*K-1;
            count_A = cur_block_A*K-1;
        end
        else begin
            finish = 1;
        end
    end

    always @(negedge clk) begin
        if(busy && !block_over) begin
            if(count_A == (cur_block_A+1)*K-1 || count_A == 16320) begin
                count_A = 16320;
            end
            else begin
                count_A = count_A + 1;
            end

            if(count_B == (cur_block_B+1)*K-1 || count_B == 16320) begin
                count_B = 16320;
            end
            else begin
                count_B = count_B + 1;
            end
        end
    end

endmodule


module BE(
    clk,
    reset,
    busy,
    block_over,
    datain,
    dataout
);
    input clk;
    input reset;
    input busy;
    input block_over;
    input [7:0] datain;

    reg [7:0] datain_c;

    output reg [7:0] dataout;

    always @(negedge reset or negedge busy or posedge block_over) begin
        datain_c = 0;
    end

    always @(posedge clk) begin
        datain_c <= datain;
    end
    
    always @(negedge clk) begin
        dataout <= datain_c;
    end
    
endmodule


module buffer(
    clk,
    reset,
    busy,
    block_over,
    datain,
    dataout
);
    parameter wh = 4;
    input clk;
    input reset;
    input busy;
    input block_over;

    input [31:0] datain;
    output [31:0] dataout;
    wire [((wh-1) * wh * 8)-1:0] data_inter;

    genvar i, j;
    generate
        for(i=0; i<4; i=i+1) begin
            for(j=0; j<4; j=j+1) begin
                if(i+j == 3) begin
                    if(j == 3) begin
                        BE be(
                            .clk (clk),
                            .reset (reset),
                            .busy (busy),
                            .block_over (block_over),
                            .datain (datain[(j+1)*8-1:j*8]),
                            .dataout (dataout[31:24])
                        );
                    end
                    else begin
                        BE be(
                            .clk (clk),
                            .reset (reset),
                            .busy (busy),
                            .block_over (block_over),
                            .datain (datain[(j+1)*8-1:j*8]),
                            .dataout (data_inter[(3*i+j+1)*8-1:(3*i+j)*8])
                        );
                    end
                end
                else begin
                    if(j == 3) begin
                        BE be(
                            .clk (clk),
                            .reset (reset),
                            .busy (busy),
                            .block_over (block_over),
                            .datain (data_inter[(3*i+j)*8-1:(3*i+j-1)*8]),
                            .dataout (dataout[(3-i+1)*8-1:(3-i)*8])
                        );
                    end
                    else if(j == 0) begin
                        BE be(
                            .clk (clk),
                            .reset (reset),
                            .busy (busy),
                            .block_over (block_over),
                            .datain (8'd0),
                            .dataout (data_inter[(3*i+j+1)*8-1:(3*i+j)*8])
                        );
                    end
                    else begin
                        BE be(
                            .clk (clk),
                            .reset (reset),
                            .busy (busy),
                            .block_over (block_over),
                            .datain (data_inter[(3*i+j)*8-1:(3*i+j-1)*8]),
                            .dataout (data_inter[(3*i+j+1)*8-1:(3*i+j)*8])
                        );
                    end
                end
            end
        end
    endgenerate


endmodule

module PE(
    i,
    j,
    clk,
    reset,
    busy,
    block_over,
    datain_h,
    datain_v,
    rd_macc_en,

    dataout_h,
    dataout_v,
    maccout,
);
    input [2:0] i;
    input [2:0] j;
    input clk;
    input reset;
    input busy;
    input block_over;
    input [7:0] datain_h;
    input [7:0] datain_v;
    input rd_macc_en;

    output reg [7:0] dataout_h;
    output reg [7:0] dataout_v;
    output reg [31:0] maccout;

    reg [31:0] mul_res;
    reg [31:0] macc;
    reg [7:0] datain_h_c;
    reg [7:0] datain_v_c;

    always @(negedge reset or negedge busy or posedge block_over) begin
        macc = 0;
    end

    always @(negedge clk) begin
        if(datain_h >= 0) begin
            mul_res = datain_h * datain_v;
            macc += mul_res;
            datain_h_c = datain_h;
            datain_v_c = datain_v;
        end
    end

    always @(posedge clk) begin
        dataout_h <= datain_h_c;
        dataout_v <= datain_v_c;
    end

    always @(posedge rd_macc_en) begin
        maccout <= macc;
    end

endmodule


module sysArr(
    clk,
    reset,
    busy,
    block_over,,
    finish,
    K,
    M,
    cur_block_B,
    datain_h,
    datain_v,
    C_index,
    C_data_in
);
    parameter wh = 4;
    input clk;
    input reset;
    input finish;
    input [7:0] K;
    input [7:0] M;
    input [7:0] cur_block_B;
    input [8*wh-1:0] datain_h;
    input [8*wh-1:0] datain_v;

    // Interconnection
    wire [((wh-1) * wh * 8)-1:0] datain_h_inter;
    wire [((wh-1) * wh * 8)-1:0] datain_v_inter;

    output reg busy;
    output reg block_over;
    output reg [15:0] macc_wr;
    output [15:0] C_index;
    output reg [127:0] C_data_in;
    output [511:0] C_data_in_c;

    reg signed [15:0] count;
    reg signed [15:0] accumu_index;
    reg [7:0] row;
    assign C_index = accumu_index;

    genvar i, j;
    generate
        for(i=0; i<4; i=i+1) begin
            for(j=0; j<4; j=j+1) begin
                if(i > 0 && i < 3 && j > 0 && j < 3) begin
                    PE pe(
                        .i (i[2:0]),
                        .j (j[2:0]),
                        .clk (clk),
                        .reset (reset),
                        .busy (busy),
                        .block_over (block_over),
                        .datain_h (datain_h_inter[(3*i+j)*8-1:(3*i+j-1)*8]),
                        .datain_v (datain_v_inter[(4*i+j-3)*8-1:(4*i+j-4)*8]),
                        .rd_macc_en (macc_wr[4*i+j]),
                        .dataout_h (datain_h_inter[(3*i+j+1)*8-1:(3*i+j)*8]),
                        .dataout_v (datain_v_inter[(4*i+j+1)*8-1:(4*i+j)*8]),
                        .maccout (C_data_in_c[128*i-32*j+127:128*i-32*j+96])
                    );
                end
                else if(i == 0) begin
                    if(j == 0) begin
                        PE pe(
                            .i (i[2:0]),
                            .j (j[2:0]),
                            .clk (clk),
                            .reset (reset),
                            .busy (busy),
                            .block_over (block_over),
                            .datain_h (datain_h[31:24]),
                            .datain_v (datain_v[31:24]),
                            .rd_macc_en (macc_wr[4*i+j]),
                            .dataout_h (datain_h_inter[(3*i+j+1)*8-1:(3*i+j)*8]),
                            .dataout_v (datain_v_inter[(4*i+j+1)*8-1:(4*i+j)*8]),
                            .maccout (C_data_in_c[128*i-32*j+127:128*i-32*j+96])
                        );
                    end
                    else if(j == 3) begin
                        PE pe(
                            .i (i[2:0]),
                            .j (j[2:0]),
                            .clk (clk),
                            .reset (reset),
                            .busy (busy),
                            .block_over (block_over),
                            .datain_h (datain_h_inter[(3*i+j)*8-1:(3*i+j-1)*8]),
                            .datain_v (datain_v[7:0]),
                            .rd_macc_en (macc_wr[4*i+j]),
                            .dataout_h (),
                            .dataout_v (datain_v_inter[(4*i+j+1)*8-1:(4*i+j)*8]),
                            .maccout (C_data_in_c[128*i-32*j+127:128*i-32*j+96])
                        );
                    end
                    else begin
                        PE pe(
                            .i (i[2:0]),
                            .j (j[2:0]),
                            .clk (clk),
                            .reset (reset),
                            .busy (busy),
                            .block_over (block_over),
                            .datain_h (datain_h_inter[(3*i+j)*8-1:(3*i+j-1)*8]),
                            .datain_v (datain_v[(3-j+1)*8-1:(3-j)*8]),
                            .rd_macc_en (macc_wr[4*i+j]),
                            .dataout_h (datain_h_inter[(3*i+j+1)*8-1:(3*i+j)*8]),
                            .dataout_v (datain_v_inter[(4*i+j+1)*8-1:(4*i+j)*8]),
                            .maccout (C_data_in_c[128*i-32*j+127:128*i-32*j+96])
                        );
                    end
                end
                else if(i == 3) begin
                    if(j == 0) begin
                        PE pe(
                            .i (i[2:0]),
                            .j (j[2:0]),
                            .clk (clk),
                            .reset (reset),
                            .busy (busy),
                            .block_over (block_over),
                            .datain_h (datain_h[7:0]),
                            .datain_v (datain_v_inter[(4*i+j-3)*8-1:(4*i+j-4)*8]),
                            .rd_macc_en (macc_wr[4*i+j]),
                            .dataout_h (datain_h_inter[(3*i+j+1)*8-1:(3*i+j)*8]),
                            .dataout_v (),
                            .maccout (C_data_in_c[128*i-32*j+127:128*i-32*j+96])
                        );
                    end
                    else if(j == 3) begin
                        PE pe(
                            .i (i[2:0]),
                            .j (j[2:0]),
                            .clk (clk),
                            .reset (reset),
                            .busy (busy),
                            .block_over (block_over),
                            .datain_h (datain_h_inter[(3*i+j)*8-1:(3*i+j-1)*8]),
                            .datain_v (datain_v_inter[(4*i+j-3)*8-1:(4*i+j-4)*8]),
                            .rd_macc_en (macc_wr[4*i+j]),
                            .dataout_h (),
                            .dataout_v (),
                            .maccout (C_data_in_c[128*i-32*j+127:128*i-32*j+96])
                        );
                    end
                    else begin
                        PE pe(
                            .i (i[2:0]),
                            .j (j[2:0]),
                            .clk (clk),
                            .reset (reset),
                            .busy (busy),
                            .block_over (block_over),
                            .datain_h (datain_h_inter[(3*i+j)*8-1:(3*i+j-1)*8]),
                            .datain_v (datain_v_inter[(4*i+j-3)*8-1:(4*i+j-4)*8]),
                            .rd_macc_en (macc_wr[4*i+j]),
                            .dataout_h (datain_h_inter[(3*i+j+1)*8-1:(3*i+j)*8]),
                            .dataout_v (),
                            .maccout (C_data_in_c[128*i-32*j+127:128*i-32*j+96])
                        );
                    end
                end
                else if(j == 0) begin
                    PE pe(
                        .i (i[2:0]),
                        .j (j[2:0]),
                        .clk (clk),
                        .reset (reset),
                        .busy (busy),
                        .block_over (block_over),
                        .datain_h (datain_h[(3-i+1)*8-1:(3-i)*8]),
                        .datain_v (datain_v_inter[(4*i+j-3)*8-1:(4*i+j-4)*8]),
                        .rd_macc_en (macc_wr[4*i+j]),
                        .dataout_h (datain_h_inter[(3*i+j+1)*8-1:(3*i+j)*8]),
                        .dataout_v (datain_v_inter[(4*i+j+1)*8-1:(4*i+j)*8]),
                        .maccout (C_data_in_c[128*i-32*j+127:128*i-32*j+96])
                    );
                end
                else begin
                    PE pe(
                        .i (i[2:0]),
                        .j (j[2:0]),
                        .clk (clk),
                        .reset (reset),
                        .busy (busy),
                        .block_over (block_over),
                        .datain_h (datain_h_inter[(3*i+j)*8-1:(3*i+j-1)*8]),
                        .datain_v (datain_v_inter[(4*i+j-3)*8-1:(4*i+j-4)*8]),
                        .rd_macc_en (macc_wr[4*i+j]),
                        .dataout_h (),
                        .dataout_v (datain_v_inter[(4*i+j+1)*8-1:(4*i+j)*8]),
                        .maccout (C_data_in_c[128*i-32*j+127:128*i-32*j+96])
                    );
                end
            end
        end
    endgenerate


    always @(negedge reset or negedge busy) begin
        count = 0;
        macc_wr = 0;
        if(accumu_index > 0) begin
            repeat(8) @(negedge clk);
        end
        else begin
            repeat(3) @(negedge clk);
        end
        accumu_index = -1;
        busy = 1;  // Here I do set busy to high immediately after in_valid fall from high to low
        block_over = 0;
    end
    
    always @(posedge clk) begin
        if(busy) begin
            count = count + 1;
        end
        if(count == K+12) begin
            count = 0;
            macc_wr = 0;
            block_over = 1;
            #1
            block_over = 0;
            if(finish) begin
                busy = 0;
            end
        end
        if(count >= K+8) begin
            if(accumu_index < 0 || accumu_index < M * (cur_block_B+1) - 1) begin
                accumu_index += 1;
                if((accumu_index%M)%4 == 0) begin
                    macc_wr = macc_wr + 4'b1111;
                    #1
                    C_data_in <= C_data_in_c[127:0];
                end
                else begin
                    macc_wr = macc_wr << 4;
                    #1
                    if((accumu_index%M)%4 == 1) begin
                        C_data_in <= C_data_in_c[255:128];
                    end
                    else if((accumu_index%M)%4 == 2) begin
                        C_data_in <= C_data_in_c[383:256];
                    end
                    else if((accumu_index%M)%4 == 3) begin
                        C_data_in <= C_data_in_c[511:384];
                    end
                end
            end
            
        end
    end

endmodule
