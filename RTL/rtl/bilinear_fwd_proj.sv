// =============================================================================
// bilinear_fwd_proj.sv
// SystemVerilog translation of pb_fwd_one_view() / cb_fwd_one_view()
//
// Bilinear forward projector: maps voxel cylinder (pixel batch, all slices)
// to one detector view using the angle θ (parallel) or full cone geometry.
//
// For parallel beam:
//   x   = (col - cx) * Δv
//   y   = (row - cy) * Δv
//   dch = (x·cosθ + y·sinθ) / Δd_ch  +  (NDC-1)/2  +  offset_ch
//   drow = slice  +  offset_row
//
// Architecture:
//   ┌────────────┐   ┌──────────┐   ┌────────────┐
//   │ Coordinate │   │ Bilinear │   │ Detector   │
//   │ Transform  │──►│  Splat   │──►│ Accumulate │──► det_out[]
//   └────────────┘   └──────────┘   └────────────┘
//
// One voxel (pi, s) is processed per clock when voxel_valid is asserted.
// =============================================================================
`timescale 1ns/1ps
import mbirjax_pkg::*;

module bilinear_fwd_proj #(
  parameter int MAX_DET_ROWS = 64,
  parameter int MAX_DET_COLS = 512,
  parameter int MAX_PIX      = 4096,
  parameter int MAX_SLICES   = 64
)(
  input  logic              clk, rst_n,
  // View parameters
  input  logic [31:0]       cos_theta_i,     // fp32 cos(θ)
  input  logic [31:0]       sin_theta_i,     // fp32 sin(θ)
  // Geometry
  input  logic [15:0]       recon_rows_i,    // NR
  input  logic [15:0]       recon_cols_i,    // NC
  input  logic [15:0]       recon_slices_i,  // NS
  input  logic [15:0]       det_rows_i,      // NDR
  input  logic [15:0]       det_chans_i,     // NDC
  input  logic [31:0]       delta_voxel_i,   // fp32
  input  logic [31:0]       delta_det_ch_i,  // fp32
  input  logic [31:0]       det_ch_off_i,    // fp32
  input  logic [31:0]       det_row_off_i,   // fp32
  // Streaming voxel input
  input  logic              voxel_valid_i,
  input  logic [31:0]       voxel_val_i,     // fp32 voxel_values[pi*NS+s]
  input  logic [15:0]       voxel_row_i,     // row index of current pixel
  input  logic [15:0]       voxel_col_i,     // col index
  input  logic [15:0]       voxel_slice_i,   // slice index s
  // Detector accumulation buffer write port
  output logic              det_wr_en_o,
  output logic [19:0]       det_wr_addr_o,   // ri*NDC + ci
  output logic [31:0]       det_wr_data_o,   // fp32 weighted contribution
  output logic              done_o
);

  // ── Coordinate computation (combinational, real arithmetic) ──────────────
  // x = (col - cx) * dv,  y = (row - cy) * dv
  // dch  = (x*cos + y*sin) / ddc + (NDC-1)/2 + off_ch
  // drow = slice + off_row

  logic [31:0] dch_fp, drow_fp;  // fp32 detector coordinates
  real         dch_r, drow_r;

  always_comb begin
    automatic real cx, cy, x, y, cos_t, sin_t, dv, ddc, off_ch, off_row;
    cx    = real'(recon_cols_i - 1) * 0.5;
    cy    = real'(recon_rows_i - 1) * 0.5;
    x     = (real'(voxel_col_i) - cx) * fp32_to_real(delta_voxel_i);
    y     = (real'(voxel_row_i) - cy) * fp32_to_real(delta_voxel_i);
    cos_t = fp32_to_real(cos_theta_i);
    sin_t = fp32_to_real(sin_theta_i);
    ddc   = fp32_to_real(delta_det_ch_i);
    off_ch  = fp32_to_real(det_ch_off_i);
    off_row = fp32_to_real(det_row_off_i);

    dch_r  = (x * cos_t + y * sin_t) / ddc
            + real'(det_chans_i - 1) * 0.5 + off_ch;
    drow_r = real'(voxel_slice_i) + off_row;
    dch_fp  = real_to_fp32(dch_r);
    drow_fp = real_to_fp32(drow_r);
  end

  // ── Bilinear splat ─────────────────────────────────────────────────────────
  // For each of the 4 bilinear neighbours:
  //   w  = (dr ? wr : 1-wr) * (dc ? wc : 1-wc)
  //   out[ri*NDC+ci] += w * val * dv

  logic [1:0]  splat_cnt;  // counts 0..3 for 4 bilinear neighbours
  logic        splat_active;
  real         wr, wc, val_r, dv_r;
  int          r0, c0;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      splat_cnt    <= 0;
      splat_active <= 0;
      det_wr_en_o  <= 0;
      done_o       <= 0;
    end else begin
      det_wr_en_o <= 0;
      done_o      <= 0;

      if (voxel_valid_i && !splat_active) begin
        // Latch intermediate values
        r0   = int'($floor(fp32_to_real(drow_fp)));
        c0   = int'($floor(fp32_to_real(dch_fp)));
        wr   = fp32_to_real(drow_fp) - real'(r0);
        wc   = fp32_to_real(dch_fp)  - real'(c0);
        val_r = fp32_to_real(voxel_val_i);
        dv_r  = fp32_to_real(delta_voxel_i);
        splat_active <= 1;
        splat_cnt    <= 0;
      end

      if (splat_active) begin
        automatic int  dr, dc, ri, ci;
        automatic real w;
        dr = int'(splat_cnt[1]);
        dc = int'(splat_cnt[0]);
        ri = r0 + dr;
        ci = c0 + dc;

        if (ri >= 0 && ri < int'(det_rows_i) &&
            ci >= 0 && ci < int'(det_chans_i)) begin
          w = (dr ? wr : 1.0 - wr) * (dc ? wc : 1.0 - wc);
          det_wr_en_o  <= 1;
          det_wr_addr_o<= 20'(ri * int'(det_chans_i) + ci);
          det_wr_data_o<= real_to_fp32(w * val_r * dv_r);
        end

        if (splat_cnt == 2'd3) begin
          splat_active <= 0;
          done_o       <= 1;
        end else begin
          splat_cnt <= splat_cnt + 1;
        end
      end
    end
  end

endmodule : bilinear_fwd_proj
