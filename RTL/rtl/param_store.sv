// =============================================================================
// param_store.sv
// SystemVerilog translation of parameter_handler.c
//
// 32-entry register file. Reset loads all library defaults (mirrors ph_init).
// Named address map (5-bit) covers every mbirjax runtime parameter.
// =============================================================================
`timescale 1ns/1ps
import mbirjax_pkg::*;

module param_store #(
  parameter int N_PARAMS = 32,
  parameter int ADDR_W   = 5
)(
  input  logic              clk, rst_n,
  // Write port (synchronous)
  input  logic              wr_en,
  input  logic [ADDR_W-1:0] wr_addr,
  input  logic [31:0]       wr_data,
  // Read port (combinational)
  input  logic [ADDR_W-1:0] rd_addr,
  output logic [31:0]       rd_data,
  // Named outputs for common parameters
  output logic [31:0] sigma_y_o, sigma_x_o,
  output logic [31:0] p_o, q_o, T_o,
  output logic [31:0] delta_det_ch_o, delta_det_row_o, delta_voxel_o,
  output logic [31:0] snr_db_o, sharpness_o,
  output logic        auto_reg_flag_o,
  output logic [15:0] sino_views_o, sino_rows_o, sino_chans_o,
  output logic [15:0] recon_rows_o, recon_cols_o, recon_slices_o
);

  // Address constants
  localparam [ADDR_W-1:0]
    A_SIGMA_Y   = 5'h00, A_SIGMA_X    = 5'h01, A_SIGMA_PROX = 5'h02,
    A_P         = 5'h03, A_Q          = 5'h04, A_T          = 5'h05,
    A_DDC       = 5'h06, A_DDR        = 5'h07, A_DRO        = 5'h08,
    A_DCO       = 5'h09, A_DV         = 5'h0A, A_SRC_ISO    = 5'h0B,
    A_SRC_DET   = 5'h0C, A_ALU        = 5'h0D,
    A_AUTO_REG  = 5'h10, A_POS        = 5'h11, A_SNR        = 5'h12,
    A_SHARP     = 5'h13, A_GRAN       = 5'h14, A_MAXOR      = 5'h15,
    A_VERB      = 5'h16, A_GEOM       = 5'h17,
    A_S_VIEWS   = 5'h18, A_S_ROWS     = 5'h19, A_S_CHANS    = 5'h1A,
    A_R_ROWS    = 5'h1B, A_R_COLS     = 5'h1C, A_R_SLICES   = 5'h1D;

  // fp32 constant bit-patterns
  localparam [31:0]
    FP_ZERO = 32'h0000_0000, FP_ONE  = 32'h3F80_0000,
    FP_1P2  = 32'h3F99_999A, FP_TWO  = 32'h4000_0000,
    FP_30   = 32'h41F0_0000, FP_1P5  = 32'h3FC0_0000;

  logic [31:0] regs [0:N_PARAMS-1];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      // Defaults — mirrors ph_init()
      for (int i = 0; i < N_PARAMS; i++) regs[i] <= 32'd0;
      regs[A_P]       <= FP_1P2;
      regs[A_Q]       <= FP_TWO;
      regs[A_T]       <= FP_ONE;
      regs[A_DDC]     <= FP_ONE;
      regs[A_DDR]     <= FP_ONE;
      regs[A_DV]      <= FP_ONE;
      regs[A_ALU]     <= FP_ONE;
      regs[A_SNR]     <= FP_30;
      regs[A_MAXOR]   <= FP_1P5;
      regs[A_AUTO_REG]<= 32'd1;
      regs[A_GRAN]    <= 32'd16;
      regs[A_VERB]    <= 32'd1;
    end else if (wr_en) begin
      regs[wr_addr] <= wr_data;
    end
  end

  assign rd_data         = regs[rd_addr];
  assign sigma_y_o       = regs[A_SIGMA_Y];
  assign sigma_x_o       = regs[A_SIGMA_X];
  assign p_o             = regs[A_P];
  assign q_o             = regs[A_Q];
  assign T_o             = regs[A_T];
  assign delta_det_ch_o  = regs[A_DDC];
  assign delta_det_row_o = regs[A_DDR];
  assign delta_voxel_o   = regs[A_DV];
  assign snr_db_o        = regs[A_SNR];
  assign sharpness_o     = regs[A_SHARP];
  assign auto_reg_flag_o = regs[A_AUTO_REG][0];
  assign sino_views_o    = regs[A_S_VIEWS][15:0];
  assign sino_rows_o     = regs[A_S_ROWS][15:0];
  assign sino_chans_o    = regs[A_S_CHANS][15:0];
  assign recon_rows_o    = regs[A_R_ROWS][15:0];
  assign recon_cols_o    = regs[A_R_COLS][15:0];
  assign recon_slices_o  = regs[A_R_SLICES][15:0];

endmodule : param_store
