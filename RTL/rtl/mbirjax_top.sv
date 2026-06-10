// =============================================================================
// mbirjax_top.sv
// Top-level integration of all MBIRJAX accelerator modules.
//
// Architecture:
//
//  ┌────────────────────────────────────────────────────────────────────────┐
//  │                         mbirjax_top                                   │
//  │                                                                        │
//  │   ┌──────────┐  ┌───────────┐  ┌────────────┐  ┌──────────────────┐  │
//  │   │  Param   │  │  ROR Mask │  │  Ramp      │  │  Shepp-Logan     │  │
//  │   │  Store   │  │  Gen      │  │  Filter    │  │  Phantom Gen     │  │
//  │   └────┬─────┘  └─────┬─────┘  └─────┬──────┘  └────────┬─────────┘  │
//  │        │              │               │                  │             │
//  │        └──────────────▼───────────────▼──────────────────▼──────────► │
//  │                  ┌─────────────────────────────────────────────┐       │
//  │                  │           VCD Controller                    │       │
//  │                  │  ┌──────────────┐  ┌───────────────────┐   │       │
//  │                  │  │ Bilinear Fwd │  │ Bilinear Back     │   │       │
//  │                  │  │ Projector    │  │ Projector         │   │       │
//  │                  │  └──────────────┘  └───────────────────┘   │       │
//  │                  │  ┌────────────┐  ┌─────────┐               │       │
//  │                  │  │ qGGMRF     │  │ Line    │               │       │
//  │                  │  │ Compute    │  │ Search  │               │       │
//  │                  │  └────────────┘  └─────────┘               │       │
//  │                  └──────────────────────────────────────────┬──┘       │
//  │                                                             │          │
//  │   ┌──────────────────┐  ┌─────────────────┐                │          │
//  │   │  FBP Controller  │  │  FDK Controller │◄───── mode ────┘          │
//  │   └──────────────────┘  └─────────────────┘                           │
//  │   ┌──────────────┐  ┌───────────────────┐                             │
//  │   │  Preprocess  │  │  Denoising Unit   │                             │
//  │   │  Unit        │  │  (MAP + Median)   │                             │
//  │   └──────────────┘  └───────────────────┘                             │
//  └────────────────────────────────────────────────────────────────────────┘
//
// Host interface: AXI4-Lite (simplified to addr/data/wr/rd for clarity)
// Memory interface: shared SRAM through address multiplexer
// =============================================================================
`timescale 1ns/1ps
import mbirjax_pkg::*;

module mbirjax_top #(
  parameter int MAX_VIEWS    = 360,
  parameter int MAX_DET_ROWS = 256,
  parameter int MAX_DET_COLS = 512,
  parameter int MAX_REC_ROWS = 512,
  parameter int MAX_REC_COLS = 512,
  parameter int MAX_SLICES   = 256,
  parameter int GRAN         = 16
)(
  input  logic        clk,
  input  logic        rst_n,

  // ── Host configuration interface (simplified register bus) ─────────────
  input  logic        cfg_wr_en,
  input  logic [4:0]  cfg_wr_addr,
  input  logic [31:0] cfg_wr_data,
  input  logic [4:0]  cfg_rd_addr,
  output logic [31:0] cfg_rd_data,

  // ── Operation control ───────────────────────────────────────────────────
  input  logic [3:0]  op_mode_i,   // 0=VCD, 1=FBP, 2=FDK, 3=Denoise, 4=Median
  input  logic        op_start_i,
  output logic        op_done_o,
  output logic        op_converged_o,
  output logic [7:0]  op_iter_o,

  // ── Sinogram SRAM (external, read-only during recon) ───────────────────
  output logic [19:0] sino_addr_o,
  input  logic [31:0] sino_data_i,

  // ── Recon SRAM (read-write) ─────────────────────────────────────────────
  output logic        recon_wr_en_o,
  output logic [19:0] recon_addr_o,
  output logic [31:0] recon_wr_data_o,
  input  logic [31:0] recon_rd_data_i,

  // ── Weight SRAM (read-only) ─────────────────────────────────────────────
  output logic [19:0] wt_addr_o,
  input  logic [31:0] wt_data_i,

  // ── Status ──────────────────────────────────────────────────────────────
  output logic [31:0] nmae_o,
  output logic [31:0] alpha_o,
  output logic [31:0] ell1_o
);

  // ── Parameter store ──────────────────────────────────────────────────────
  logic [31:0] ps_rd_data;
  logic [4:0]  ps_rd_addr;
  logic [31:0] sigma_y_ps, sigma_x_ps, p_ps, q_ps, T_ps;
  logic [31:0] ddc_ps, ddr_ps, dv_ps, snr_ps, sharp_ps;
  logic        auto_reg_ps;
  logic [15:0] sv_ps, sr_ps, sc_ps, rr_ps, rc_ps, rs_ps;

  param_store u_params (
    .clk            (clk),
    .rst_n          (rst_n),
    .wr_en          (cfg_wr_en),
    .wr_addr        (cfg_wr_addr),
    .wr_data        (cfg_wr_data),
    .rd_addr        (ps_rd_addr),
    .rd_data        (ps_rd_data),
    .sigma_y_o      (sigma_y_ps),
    .sigma_x_o      (sigma_x_ps),
    .p_o            (p_ps),
    .q_o            (q_ps),
    .T_o            (T_ps),
    .delta_det_ch_o (ddc_ps),
    .delta_det_row_o(ddr_ps),
    .delta_voxel_o  (dv_ps),
    .snr_db_o       (snr_ps),
    .sharpness_o    (sharp_ps),
    .auto_reg_flag_o(auto_reg_ps),
    .sino_views_o   (sv_ps),
    .sino_rows_o    (sr_ps),
    .sino_chans_o   (sc_ps),
    .recon_rows_o   (rr_ps),
    .recon_cols_o   (rc_ps),
    .recon_slices_o (rs_ps)
  );
  assign cfg_rd_data = ps_rd_data;
  assign ps_rd_addr  = cfg_rd_addr;

  // ── ROR mask generator ───────────────────────────────────────────────────
  logic        ror_start, ror_done, ror_pv;
  logic [19:0] ror_idx, ror_npix;

  ror_mask_gen u_ror (
    .clk         (clk),
    .rst_n       (rst_n),
    .rows_i      (rr_ps),
    .cols_i      (rc_ps),
    .start_i     (ror_start),
    .done_o      (ror_done),
    .pix_valid_o (ror_pv),
    .pix_index_o (ror_idx),
    .n_pixels_o  (ror_npix)
  );

  // ── VCD controller ───────────────────────────────────────────────────────
  logic        vcd_start, vcd_done, vcd_conv;
  logic [7:0]  vcd_iter;
  logic [31:0] vcd_nmae, vcd_alpha, vcd_ell1;
  logic [19:0] vcd_recon_ra, vcd_recon_wa, vcd_esino_ra, vcd_esino_wa;
  logic        vcd_recon_re, vcd_recon_we, vcd_esino_re, vcd_esino_we;
  logic [31:0] vcd_recon_wd, vcd_esino_wd;
  logic [19:0] vcd_wt_ra;
  logic [4:0]  vcd_ps_ra;

  vcd_controller #(.MAX_ITER(15), .MAX_PIX(32768), .MAX_SINO(1<<20), .GRAN(GRAN)) u_vcd (
    .clk           (clk),
    .rst_n         (rst_n),
    .start_i       (vcd_start),
    .max_iter_i    (8'd15),
    .stop_pct_i    (32'h3E4CCCCD), // 0.2 fp32
    .done_o        (vcd_done),
    .converged_o   (vcd_conv),
    .iter_count_o  (vcd_iter),
    .ps_rd_addr_o  (vcd_ps_ra),
    .ps_rd_data_i  (ps_rd_data),
    .recon_rd_en_o (vcd_recon_re), .recon_rd_addr_o(vcd_recon_ra),
    .recon_rd_data_i(recon_rd_data_i),
    .recon_wr_en_o (vcd_recon_we), .recon_wr_addr_o(vcd_recon_wa),
    .recon_wr_data_o(vcd_recon_wd),
    .esino_rd_en_o (vcd_esino_re), .esino_rd_addr_o(vcd_esino_ra),
    .esino_rd_data_i(sino_data_i),
    .esino_wr_en_o (vcd_esino_we), .esino_wr_addr_o(vcd_esino_wa),
    .esino_wr_data_o(vcd_esino_wd),
    .wt_rd_addr_o  (vcd_wt_ra),   .wt_rd_data_i  (wt_data_i),
    .nmae_o        (vcd_nmae),    .alpha_o        (vcd_alpha),
    .ell1_o        (vcd_ell1)
  );

  // ── FBP controller ───────────────────────────────────────────────────────
  logic        fbp_start, fbp_done;
  logic [19:0] fbp_sino_addr, fbp_recon_addr;
  logic        fbp_recon_we;
  logic [31:0] fbp_recon_wd;
  logic [15:0] fbp_ang_idx;
  logic [31:0] fbp_ang_data;

  fbp_controller #(.MAX_VIEWS(MAX_VIEWS)) u_fbp (
    .clk            (clk),     .rst_n         (rst_n),
    .start_i        (fbp_start),
    .num_views_i    (sv_ps),   .det_rows_i    (sr_ps),
    .det_chans_i    (sc_ps),   .recon_rows_i  (rr_ps),
    .recon_cols_i   (rc_ps),   .recon_slices_i(rs_ps),
    .delta_det_ch_i (ddc_ps),  .delta_voxel_i (dv_ps),
    .det_ch_off_i   (32'h0),   .det_row_off_i (32'h0),
    .angle_rd_idx_o (fbp_ang_idx),
    .angle_rd_data_i(fbp_ang_data),
    .sino_rd_addr_o (fbp_sino_addr),
    .sino_rd_data_i (sino_data_i),
    .recon_wr_en_o  (fbp_recon_we),
    .recon_wr_addr_o(fbp_recon_addr),
    .recon_wr_data_o(fbp_recon_wd),
    .done_o         (fbp_done)
  );

  // ── Denoising unit ───────────────────────────────────────────────────────
  logic        den_start, den_done;
  logic [19:0] den_img_ra, den_img_wa, den_out_wa;
  logic        den_img_we, den_out_we;
  logic [31:0] den_img_wd, den_out_wd;

  denoising_unit #(.MAX_NZ(64),.MAX_NY(64),.MAX_NX(64)) u_den (
    .clk          (clk),       .rst_n        (rst_n),
    .start_i      (den_start), .mode_i       (op_mode_i[0]),
    .nz_i         (rs_ps),     .ny_i         (rr_ps),   .nx_i(rc_ps),
    .sigma_noise_i(32'h0),     .sigma_x_i    (sigma_x_ps),
    .sharpness_i  (sharp_ps),  .p_i(p_ps),  .q_i(q_ps), .T_i(T_ps),
    .max_iter_i   (8'd15),     .stop_pct_i   (32'h3E4CCCCD),
    .gran_i       (8'd16),
    .img_rd_addr_o(den_img_ra), .img_rd_data_i(recon_rd_data_i),
    .img_wr_en_o  (den_img_we), .img_wr_addr_o(den_img_wa),
    .img_wr_data_o(den_img_wd),
    .out_wr_en_o  (den_out_we), .out_wr_addr_o(den_out_wa),
    .out_wr_data_o(den_out_wd),
    .done_o       (den_done)
  );

  // ── Shepp-Logan generator ─────────────────────────────────────────────────
  logic        slg_start, slg_done, slg_vv;
  logic [19:0] slg_addr;
  logic [31:0] slg_val;

  shepp_logan_gen u_slg (
    .clk          (clk),       .rst_n        (rst_n),
    .start_i      (slg_start), .rows_i       (rr_ps),
    .cols_i       (rc_ps),     .slices_i     (rs_ps),
    .voxel_valid_o(slg_vv),    .voxel_addr_o (slg_addr),
    .voxel_val_o  (slg_val),   .done_o       (slg_done)
  );

  // ── Operation dispatch ────────────────────────────────────────────────────
  // Route start signals and mux done, memory, status outputs

  always_comb begin
    vcd_start = 0; fbp_start = 0; den_start = 0; slg_start = 0;
    ror_start = 0;
    case (op_mode_i)
      4'd0: vcd_start = op_start_i;
      4'd1: fbp_start = op_start_i;
      4'd3: den_start = op_start_i;
      4'd4: den_start = op_start_i;
      4'd5: slg_start = op_start_i;
      default: ;
    endcase
  end

  // Done mux
  always_comb begin
    op_done_o      = 0;
    op_converged_o = 0;
    op_iter_o      = 0;
    case (op_mode_i)
      4'd0: begin op_done_o = vcd_done; op_converged_o = vcd_conv; op_iter_o = vcd_iter; end
      4'd1: op_done_o = fbp_done;
      4'd3, 4'd4: op_done_o = den_done;
      4'd5: op_done_o = slg_done;
      default: op_done_o = 0;
    endcase
  end

  // SRAM address mux
  always_comb begin
    sino_addr_o      = 20'h0;
    recon_wr_en_o    = 0;
    recon_addr_o     = 20'h0;
    recon_wr_data_o  = 32'h0;
    wt_addr_o        = 20'h0;
    case (op_mode_i)
      4'd0: begin
        sino_addr_o     = vcd_esino_ra;
        recon_wr_en_o   = vcd_recon_we;
        recon_addr_o    = vcd_recon_we ? vcd_recon_wa : vcd_recon_ra;
        recon_wr_data_o = vcd_recon_wd;
        wt_addr_o       = vcd_wt_ra;
      end
      4'd1: begin
        sino_addr_o     = fbp_sino_addr;
        recon_wr_en_o   = fbp_recon_we;
        recon_addr_o    = fbp_recon_addr;
        recon_wr_data_o = fbp_recon_wd;
      end
      4'd3, 4'd4: begin
        recon_wr_en_o   = den_img_we;
        recon_addr_o    = den_img_we ? den_img_wa : den_img_ra;
        recon_wr_data_o = den_img_wd;
      end
      4'd5: begin
        recon_wr_en_o   = slg_vv;
        recon_addr_o    = slg_addr;
        recon_wr_data_o = slg_val;
      end
      default: ;
    endcase
  end

  assign nmae_o  = vcd_nmae;
  assign alpha_o = vcd_alpha;
  assign ell1_o  = vcd_ell1;

endmodule : mbirjax_top
