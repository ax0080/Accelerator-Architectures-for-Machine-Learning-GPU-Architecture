//P = X*Y + Z
module mult_add#(
	parameter DATAWIDTH = 8
)
(
	input  [ DATAWIDTH - 1 : 0 ] X,
	input  [ DATAWIDTH - 1 : 0 ] Y,
	input  [ DATAWIDTH - 1 : 0 ] Z,
	output [ DATAWIDTH * 2  : 0 ] P
);

wire [ DATAWIDTH * 2  : 0 ] mult = X * Y;
wire [ DATAWIDTH * 2  : 0 ] P_w = mult + Z;
assign P = P_w;

endmodule


//PE(Processing element)
module PE#(
	parameter DATAWIDTH = 8
)
(
	input                        CLK,
	input                        RSTn,
	input  [ DATAWIDTH - 1 : 0 ] A,
	input  [ DATAWIDTH - 1 : 0 ] B,
	output [ DATAWIDTH - 1 : 0 ] Next_A,
	output [ DATAWIDTH - 1 : 0 ] Next_B,
	output [ DATAWIDTH * 2  : 0 ] PE_out
);

reg [ DATAWIDTH - 1 : 0 ] Next_A_reg;
reg [ DATAWIDTH - 1 : 0 ] Next_B_reg;
reg  [ DATAWIDTH * 2  : 0 ] PE_reg;
wire [ DATAWIDTH * 2 : 0 ] PE_net;

mult_add#(.DATAWIDTH(DATAWIDTH)) multadd(.X(A), .Y(B), .Z(PE_reg), .P(PE_net));

always @ ( posedge CLK or negedge RSTn )
begin
	if (!RSTn)
	begin
		Next_A_reg <= 0;
		Next_B_reg <= 0;
		PE_reg <= 0;
		
	end
	else 
	begin
		Next_A_reg <= A;
		Next_B_reg <= B;
		PE_reg <= PE_net;
	end
end

assign PE_out = PE_reg;
assign Next_A = Next_A_reg;
assign Next_B = Next_B_reg;
endmodule


module TPU(
    input clk,
    input rst_n,
    input in_valid,
    input [7:0] K,
    input [7:0] M,
    input [7:0] N,
    output reg busy,

    output A_wr_en,
    output [15:0] A_index,
    output [31:0] A_data_in,
    input [31:0] A_data_out,

    output B_wr_en,
    output [15:0] B_index,
    output [31:0] B_data_in,
    input [31:0] B_data_out,

    output C_wr_en,
    output [15:0] C_index,
    output [127:0] C_data_in,
    input [127:0] C_data_out
);

// PE instances in a 4x4 Systolic Array
wire [31:0] PE_inputs [3:0][3:0];
wire [31:0] PE_outputs [3:0][3:0];

genvar i, j;
generate
    for (i = 0; i < 4; i=i+1) begin
        for (j = 0; j < 4; j=j+1) begin
            PE #(32) PE_inst (
                .CLK(clk),
                .RSTn(rst_n),
                .A(PE_inputs[i][j]),
                .B(PE_inputs[(i+1)%4][j]),
                .Next_A(PE_outputs[i][j]),
                .Next_B(PE_outputs[(i+1)%4][j]),
                .PE_out(PE_inputs[i][j])
            );
        end
    end
endgenerate

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        busy <= 0;
    end else begin
        if (in_valid) begin
            // 设置数据写入全局缓冲区A和B的控制信号
            A_wr_en <= 1;
            A_index <= 0; // 设置A的索引
            A_data_in <= K; // 设置A的数据
            B_wr_en <= 1;
            B_index <= 0; // 设置B的索引
            B_data_in <= M; // 设置B的数据
            busy <= 1; // 设置忙状态
        end else if (busy) begin
            // 等待乘法操作完成
            A_wr_en <= 0;
            B_wr_en <= 0;
            C_wr_en <= 1;
            C_index <= 0; // 设置C的索引
            // 设置C的数据（来自Systolic Array的输出）
            C_data_in[127:0] <= {PE_outputs[0][0], PE_outputs[0][1], PE_outputs[0][2], PE_outputs[0][3],
                                 PE_outputs[1][0], PE_outputs[1][1], PE_outputs[1][2], PE_outputs[1][3],
                                 PE_outputs[2][0], PE_outputs[2][1], PE_outputs[2][2], PE_outputs[2][3],
                                 PE_outputs[3][0], PE_outputs[3][1], PE_outputs[3][2], PE_outputs[3][3]};
            busy <= 0; // 清除忙状态
        end else begin
            A_wr_en <= 0;
            B_wr_en <= 0;
            C_wr_en <= 0;
        end
    end
end

endmodule
