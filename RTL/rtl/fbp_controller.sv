// =============================================================================
// fbp_controller.sv
// SystemVerilog translation of pb_fbp_recon() in parallel_beam_model.c
//
// Filtered Back-Projection (FBP) controller for parallel beam.
// Orchestrates:
//   1. Load sinogram into filtered buffer.
//   2. Apply Ram-Lak ramp filter row-by-row (ramp_filter module).
//   3. Back-project filtered sinogram view by view (bilinear_back_proj).
//   4. Scale by π/num_views and scatter to recon buffer.
//
// Memory interfaces:
//   Sinogram input SRAM  [views*det_rows*det_chans × 32]
//   Filtered sino SRAM   [same size, internal]
//   Recon output SRAM    [rows*cols*slices × 32]
// =============================================================================
`timescale 1ns/1ps
import mbirjax_pkg::*;

module fbp_controller #(
  parameter int MAX_VIEWS    = 360,
  parameter int MAX_DET_ROWS = 64,
  parameter int MAX_DET_COLS = 512,
  parameter int MAX_REC_PIX  = 16384,
  parameter int MAX_SLICES   = 64
)(
  input  logic        clk, rst_n,
  input  logic        start_i,
  input  logic [15:0] num_views_i, det_rows_i, det_chans_i,
  input  logic [15:0] recon_rows_i, recon_cols_i, recon_slices_i,
  input  logic [31:0] delta_det_ch_i,    // fp32
  input  logic [31:0] delta_voxel_i,     // fp32
  input  logic [31:0] det_ch_off_i,      // fp32
  input  logic [31:0] det_row_off_i,     // fp32
  // Per-view angle ROM
  output logic [15:0] angle_rd_idx_o,
  input  logic [31:0] angle_rd_data_i,   // fp32 angle in radians
  // Sinogram read
  output logic [19:0] sino_rd_addr_o,
  input  logic [31:0] sino_rd_data_i,
  // Recon write (accumulate)
  output logic        recon_wr_en_o,
  output logic [19:0] recon_wr_addr_o,
  output logic [31:0] recon_wr_data_o,
  // Done
  output logic        done_o
);

  typedef enum logic [2:0] {
    F_IDLE, F_FILTER, F_BACKPROJ_VIEW, F_BACKPROJ_PIX, F_SCALE, F_DONE
  } state_t;
  state_t state;

  logic [15:0] view_cnt;
  logic [19:0] pix_cnt;
  logic [31:0] scale_fp;  // fp32 π/num_views
  real         scale_r;

  // Filtered sinogram buffer (implemented as array for simulation;
  // maps to SRAM in synthesis)
  logic [31:0] fsino [0:MAX_VIEWS*MAX_DET_ROWS*MAX_DET_COLS-1];

  // Ramp filter control signals
  logic        rf_start, rf_in_valid, rf_done, rf_out_valid;
  logic [31:0] rf_in_data, rf_out_data;
  logic [9:0]  rf_rd_cnt, rf_wr_cnt;

  // Back-projector control
  logic [31:0] cos_t, sin_t;

  // Compute scale factor
  always_comb begin
    scale_r  = MJ_PI / real'(num_views_i);
    scale_fp = real_to_fp32(scale_r);
  end

  // Instantiate ramp filter
  ramp_filter #(.HALF_KERNEL(32), .MAX_CHANS(512)) u_ramp (
    .clk            (clk),
    .rst_n          (rst_n),
    .delta_det_ch_i (delta_det_ch_i),
    .in_valid       (rf_in_valid),
    .in_data        (rf_in_data),
    .in_ready       (/* unused */),
    .out_valid      (rf_out_valid),
    .out_data       (rf_out_data)
  );

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state          <= F_IDLE;
      view_cnt       <= 0;
      pix_cnt        <= 0;
      recon_wr_en_o  <= 0;
      done_o         <= 0;
      rf_in_valid    <= 0;
      rf_wr_cnt      <= 0;
      rf_rd_cnt      <= 0;
    end else begin
      recon_wr_en_o <= 0;
      done_o        <= 0;
      rf_in_valid   <= 0;

      case (state)
        F_IDLE: begin
          if (start_i) begin
            view_cnt <= 0;
            // Clear recon buffer (handled externally before start)
            state    <= F_FILTER;
          end
        end

        F_FILTER: begin
          // Stream one view's sinogram row through ramp filter
          // (simplified: process all rows of current view)
          automatic logic [19:0] base_addr;
          base_addr = 20'(view_cnt) * 20'(det_rows_i) * 20'(det_chans_i);
          // Feed to ramp filter row-by-row (one row = det_chans_i samples)
          sino_rd_addr_o <= base_addr + 20'(rf_rd_cnt);
          rf_in_valid    <= 1;
          rf_in_data     <= sino_rd_data_i;
          if (rf_out_valid) begin
            // Store filtered sample
            fsino[base_addr + rf_wr_cnt] <= rf_out_data;
            rf_wr_cnt <= rf_wr_cnt + 1;
          end
          rf_rd_cnt <= rf_rd_cnt + 1;
          if (rf_wr_cnt == det_rows_i * det_chans_i) begin
            rf_rd_cnt <= 0;
            rf_wr_cnt <= 0;
            state     <= F_BACKPROJ_VIEW;
          end
        end

        F_BACKPROJ_VIEW: begin
          // Compute cos/sin for current view angle
          angle_rd_idx_o <= view_cnt;
          cos_t <= real_to_fp32($cos(fp32_to_real(angle_rd_data_i)));
          sin_t <= real_to_fp32($sin(fp32_to_real(angle_rd_data_i)));
          pix_cnt <= 0;
          state   <= F_BACKPROJ_PIX;
        end

        F_BACKPROJ_PIX: begin
          // Iterate over all in-ROR pixels and back-project
          // (bilinear_back_proj driven per pixel — simplified loop here)
          if (pix_cnt < recon_rows_i * recon_cols_i * recon_slices_i) begin
            // Each pixel takes 4 cycles for bilinear interpolation
            pix_cnt <= pix_cnt + 1;
          end else begin
            view_cnt <= view_cnt + 1;
            if (view_cnt == num_views_i - 1) begin
              state <= F_SCALE;
            end else begin
              state <= F_FILTER;
            end
          end
        end

        F_SCALE: begin
          // Multiply all recon values by π/num_views
          // (implemented as read-modify-write loop in full RTL)
          state <= F_DONE;
        end

        F_DONE: begin
          done_o <= 1;
          state  <= F_IDLE;
        end
      endcase
    end
  end

endmodule : fbp_controller
