// =============================================================================
// bilinear_back_proj.sv
// SystemVerilog translation of pb_back_one_view() / cb_back_one_view()
//
// Bilinear back projector: maps one sinogram view to a pixel batch.
// coeff_power = 1 (normal), coeff_power = 2 (Hessian diagonal — squares coeff).
//
// For each pixel (row,col,slice):
//   Compute (dch, drow) detector coordinates.
//   Bilinear interpolation from 4 detector cells.
//   recon_out[pi*NS+s] += bilinear_sample
//
// Streaming: one pixel (pi, s) per clock.
// =============================================================================
`timescale 1ns/1ps
import mbirjax_pkg::*;

module bilinear_back_proj #(
  parameter int MAX_DET_ROWS = 64,
  parameter int MAX_DET_COLS = 512,
  parameter int MAX_PIX      = 4096,
  parameter int MAX_SLICES   = 64
)(
  input  logic        clk, rst_n,
  // View parameters
  input  logic [31:0] cos_theta_i, sin_theta_i,
  // Geometry (same ports as fwd proj)
  input  logic [15:0] recon_rows_i, recon_cols_i, recon_slices_i,
  input  logic [15:0] det_rows_i, det_chans_i,
  input  logic [31:0] delta_voxel_i, delta_det_ch_i,
  input  logic [31:0] det_ch_off_i, det_row_off_i,
  input  logic        coeff_power_2_i,  // 0=normal, 1=Hessian
  // Streaming pixel input
  input  logic        pixel_valid_i,
  input  logic [15:0] pixel_row_i, pixel_col_i, pixel_slice_i,
  // Sinogram read port (combinational)
  output logic [19:0] sino_rd_addr_o,  // ri*NDC+ci
  input  logic [31:0] sino_rd_data_i,  // fp32 sinogram sample
  // Output: accumulated recon contribution
  output logic        recon_wr_en_o,
  output logic [31:0] recon_wr_data_o,  // fp32 to add to recon[pi*NS+s]
  output logic        done_o
);

  logic [1:0]  interp_cnt;
  logic        interp_active;
  real         wr, wc, accum;
  int          r0, c0;
  real         dch_r, drow_r, dv_r, ddc_r;

  always_comb begin
    automatic real cx, cy, x, y;
    cx    = real'(recon_cols_i - 1) * 0.5;
    cy    = real'(recon_rows_i - 1) * 0.5;
    x     = (real'(pixel_col_i) - cx) * fp32_to_real(delta_voxel_i);
    y     = (real'(pixel_row_i) - cy) * fp32_to_real(delta_voxel_i);
    ddc_r = fp32_to_real(delta_det_ch_i);
    dch_r  = (x * fp32_to_real(cos_theta_i) + y * fp32_to_real(sin_theta_i))
             / ddc_r + real'(det_chans_i - 1) * 0.5 + fp32_to_real(det_ch_off_i);
    drow_r = real'(pixel_slice_i) + fp32_to_real(det_row_off_i);
    dv_r   = fp32_to_real(delta_voxel_i);
  end

  // Use sino_rd_addr_o combinationally for first tap while interp_cnt drives
  assign sino_rd_addr_o = 20'(
    (int'($floor(drow_r)) + int'(interp_cnt[1])) * int'(det_chans_i) +
    (int'($floor(dch_r))  + int'(interp_cnt[0]))
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      interp_cnt    <= 0;
      interp_active <= 0;
      recon_wr_en_o <= 0;
      done_o        <= 0;
      accum         <= 0.0;
    end else begin
      recon_wr_en_o <= 0;
      done_o        <= 0;

      if (pixel_valid_i && !interp_active) begin
        r0   = int'($floor(drow_r));
        c0   = int'($floor(dch_r));
        wr   = drow_r - real'(r0);
        wc   = dch_r  - real'(c0);
        accum         <= 0.0;
        interp_active <= 1;
        interp_cnt    <= 0;
      end

      if (interp_active) begin
        automatic int  dr, dc, ri, ci;
        automatic real w, coeff, sv;
        dr = int'(interp_cnt[1]);
        dc = int'(interp_cnt[0]);
        ri = r0 + dr;
        ci = c0 + dc;

        if (ri >= 0 && ri < int'(det_rows_i) &&
            ci >= 0 && ci < int'(det_chans_i)) begin
          w     = (dr ? wr : 1.0 - wr) * (dc ? wc : 1.0 - wc);
          coeff = w * dv_r;
          sv    = fp32_to_real(sino_rd_data_i);
          accum <= accum + (coeff_power_2_i ? coeff*coeff*sv : coeff*sv);
        end

        if (interp_cnt == 2'd3) begin
          recon_wr_en_o  <= 1;
          recon_wr_data_o<= real_to_fp32(accum);
          interp_active  <= 0;
          done_o         <= 1;
        end else begin
          interp_cnt <= interp_cnt + 1;
        end
      end
    end
  end

endmodule : bilinear_back_proj
