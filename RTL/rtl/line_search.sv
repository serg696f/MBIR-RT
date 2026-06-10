// =============================================================================
// line_search.sv
// SystemVerilog translation of the alpha line-search step in vcd_utils.c
//
// Computes the optimal VCD step size α:
//   α = (fwd_lin − prior_lin) / (fwd_quad + prior_quad + ε)
//   α = clip(α, ε, 1.5)
//
// Inputs are pre-accumulated real scalars from the subset update loop.
// All arithmetic is real (synthesised as floating-point units on FPGA).
// =============================================================================
`timescale 1ns/1ps
import mbirjax_pkg::*;

module line_search (
  input  logic        clk, rst_n,
  input  logic        start_i,
  // Accumulated scalars (fp32)
  input  logic [31:0] fwd_lin_i,
  input  logic [31:0] prior_lin_i,
  input  logic [31:0] fwd_quad_i,
  input  logic [31:0] prior_quad_i,
  // Result
  output logic        done_o,
  output logic [31:0] alpha_o   // fp32, clipped to [eps, 1.5]
);

  localparam real ALPHA_MIN = 1.192093e-7;
  localparam real ALPHA_MAX = 1.5;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      done_o  <= 0;
      alpha_o <= 0;
    end else begin
      done_o <= 0;
      if (start_i) begin
        automatic real num, den, alpha;
        num   = fp32_to_real(fwd_lin_i) - fp32_to_real(prior_lin_i);
        den   = fp32_to_real(fwd_quad_i) + fp32_to_real(prior_quad_i) + MJ_EPS;
        alpha = mj_clipf(num / den, ALPHA_MIN, ALPHA_MAX);
        alpha_o <= real_to_fp32(alpha);
        done_o  <= 1;
      end
    end
  end

endmodule : line_search
