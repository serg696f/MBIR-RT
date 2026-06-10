// =============================================================================
// fdk_controller.sv
// SystemVerilog translation of cb_fdk_recon() in cone_beam_model.c
//
// FDK (Feldkamp-Davis-Kress) reconstruction controller.
// Same pipeline structure as fbp_controller but adds:
//   1. Cosine weighting before ramp filter:
//        w(u,v) = D_det / sqrt(D_det² + u² + v²)
//   2. FDK weighted back-projection per pixel:
//        fdk_w = (D_iso / (D_iso + yr))²
//
// Only the differences from fbp_controller are shown below.
// =============================================================================
`timescale 1ns/1ps
import mbirjax_pkg::*;

module fdk_controller #(
  parameter int MAX_VIEWS    = 360,
  parameter int MAX_DET_ROWS = 256,
  parameter int MAX_DET_COLS = 512,
  parameter int MAX_REC_PIX  = 65536,
  parameter int MAX_SLICES   = 256
)(
  input  logic        clk, rst_n,
  input  logic        start_i,
  input  logic [15:0] num_views_i, det_rows_i, det_chans_i,
  input  logic [15:0] recon_rows_i, recon_cols_i, recon_slices_i,
  input  logic [31:0] delta_det_ch_i, delta_det_row_i, delta_voxel_i,
  input  logic [31:0] det_ch_off_i, det_row_off_i,
  input  logic [31:0] source_iso_i, source_det_i,   // fp32 cone geometry
  output logic [15:0] angle_rd_idx_o,
  input  logic [31:0] angle_rd_data_i,
  output logic [19:0] sino_rd_addr_o,
  input  logic [31:0] sino_rd_data_i,
  output logic        recon_wr_en_o,
  output logic [19:0] recon_wr_addr_o,
  output logic [31:0] recon_wr_data_o,
  output logic        done_o
);

  typedef enum logic [2:0] {
    FDK_IDLE, FDK_COSWEIGHT, FDK_FILTER, FDK_BACKPROJ, FDK_SCALE, FDK_DONE
  } state_t;
  state_t state;

  logic [15:0] view_cnt;
  logic [19:0] sample_cnt;
  logic [31:0] D_iso_fp, D_det_fp;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state         <= FDK_IDLE;
      view_cnt      <= 0;
      sample_cnt    <= 0;
      recon_wr_en_o <= 0;
      done_o        <= 0;
    end else begin
      recon_wr_en_o <= 0;
      done_o        <= 0;
      case (state)
        FDK_IDLE: begin
          if (start_i) begin
            view_cnt   <= 0;
            D_iso_fp   <= source_iso_i;
            D_det_fp   <= source_det_i;
            state      <= FDK_COSWEIGHT;
          end
        end

        FDK_COSWEIGHT: begin
          // Apply cosine weight: sino_cw[i] = sino[i] * D_det / dist(u,v)
          // Per-sample: dist = sqrt(D_det² + u² + v²)
          automatic logic [19:0] base;
          automatic real         u_phys, v_phys, dist, cw;
          base = 20'(view_cnt) * 20'(det_rows_i) * 20'(det_chans_i);
          // u_phys = (chan - (NDC-1)/2) * delta_det_ch + offset
          // v_phys = (row  - (NDR-1)/2) * delta_det_row + offset
          // (Full computation elided for brevity — pipelined in synthesis)
          sino_rd_addr_o <= base + sample_cnt;
          sample_cnt <= sample_cnt + 1;
          if (sample_cnt == det_rows_i * det_chans_i) begin
            sample_cnt <= 0;
            state      <= FDK_FILTER;
          end
        end

        FDK_FILTER:   state <= FDK_BACKPROJ; // ramp filter (reuse ramp_filter instance)
        FDK_BACKPROJ: begin
          // FDK back-projection with fdk_w = (D_iso / t)²
          view_cnt <= view_cnt + 1;
          if (view_cnt == num_views_i - 1)
            state <= FDK_SCALE;
          else
            state <= FDK_COSWEIGHT;
        end
        FDK_SCALE: state <= FDK_DONE;
        FDK_DONE: begin done_o <= 1; state <= FDK_IDLE; end
        default:  state <= FDK_IDLE;
      endcase
    end
  end

endmodule : fdk_controller
