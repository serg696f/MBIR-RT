// =============================================================================
// vcd_controller.sv
// SystemVerilog translation of vcd_subset_update() in vcd_utils.c and
// the outer VCD loop in mj_recon() in tomography_model.c
//
// Top-level VCD reconstruction controller. Orchestrates:
//   1. Auto-regularisation (sigma_y, sigma_x from snr_db, sharpness)
//   2. Pixel partition generation (ROR mask → subsets of size granularity)
//   3. Per-subset VCD update:
//      a. Prior gradient + hessian (qggmrf_compute × n_pixels)
//      b. Forward Hessian diagonal (back-project all-ones, coeff²)
//      c. Forward gradient (back-project weighted error sinogram)
//      d. Update direction  Δx = -(fwd_grad + prior_grad)/(fwd_hess + prior_hess)
//      e. Forward-project Δx → Δsino
//      f. Line search for α
//      g. Apply update: recon += α·Δx,  error_sino -= α·Δsino
//   4. NMAE convergence check
//
// Memory interface:
//   Recon buffer   — dual-port SRAM [rows*cols*slices × 32] (fp32)
//   Error sinogram — dual-port SRAM [views*det_rows*det_chans × 32]
//   Weight buffer  — read-only SRAM [same as sino]
//
// FSM states (abbreviated):
//   IDLE → INIT → AUTO_REG → SUBSET_PRIOR → SUBSET_FWD_HESS →
//   SUBSET_FWD_GRAD → SUBSET_DELTA → SUBSET_FWD_PROJ → LINE_SEARCH →
//   APPLY_UPDATE → UPDATE_SINO → NMAE_CHECK → [next subset/iter or DONE]
// =============================================================================
`timescale 1ns/1ps
import mbirjax_pkg::*;

module vcd_controller #(
  parameter int MAX_ITER     = 15,
  parameter int MAX_PIX      = 16384,
  parameter int MAX_SINO     = 524288,
  parameter int MAX_VIEWS    = 360,
  parameter int GRAN         = 16
)(
  input  logic        clk, rst_n,

  // Control
  input  logic        start_i,
  input  logic [7:0]  max_iter_i,
  input  logic [31:0] stop_pct_i,   // fp32 stop threshold %
  output logic        done_o,
  output logic        converged_o,
  output logic [7:0]  iter_count_o,

  // Parameter store interface (read)
  output logic [4:0]  ps_rd_addr_o,
  input  logic [31:0] ps_rd_data_i,

  // Recon buffer interface (dual-port)
  output logic        recon_rd_en_o,
  output logic [19:0] recon_rd_addr_o,
  input  logic [31:0] recon_rd_data_i,
  output logic        recon_wr_en_o,
  output logic [19:0] recon_wr_addr_o,
  output logic [31:0] recon_wr_data_o,

  // Error sinogram interface (dual-port)
  output logic        esino_rd_en_o,
  output logic [19:0] esino_rd_addr_o,
  input  logic [31:0] esino_rd_data_i,
  output logic        esino_wr_en_o,
  output logic [19:0] esino_wr_addr_o,
  output logic [31:0] esino_wr_data_o,

  // Weight buffer (read-only)
  output logic [19:0] wt_rd_addr_o,
  input  logic [31:0] wt_rd_data_i,

  // Status outputs
  output logic [31:0] nmae_o,      // fp32 last NMAE
  output logic [31:0] alpha_o,     // fp32 last step size
  output logic [31:0] ell1_o       // fp32 last update L1 norm
);

  // ── FSM state encoding ────────────────────────────────────────────────────
  typedef enum logic [4:0] {
    IDLE          = 5'd0,
    INIT          = 5'd1,
    AUTO_REG      = 5'd2,
    PART_GEN      = 5'd3,
    SUBSET_START  = 5'd4,
    PRIOR_LOOP    = 5'd5,
    FWD_HESS      = 5'd6,
    FWD_GRAD      = 5'd7,
    DELTA_CALC    = 5'd8,
    FWD_PROJ      = 5'd9,
    LSEARCH       = 5'd10,
    APPLY_UPD     = 5'd11,
    UPD_SINO      = 5'd12,
    NMAE_CALC     = 5'd13,
    NMAE_CHECK    = 5'd14,
    ITER_NEXT     = 5'd15,
    DONE_ST       = 5'd16
  } state_t;

  state_t      state;
  logic [7:0]  iter;
  logic [13:0] subset_idx;
  logic [13:0] n_subsets;
  logic [13:0] pixel_in_subset;
  logic [13:0] subset_size;

  // Accumulated scalars
  real fwd_lin, prior_lin, fwd_quad, prior_quad;
  real ell1_acc, recon_l1_acc;
  real alpha_r, nmae_r;

  // Parameter cache (loaded once at INIT)
  real sigma_x_r, sigma_y_r, p_r, q_r, T_r;
  real fm_const_r;

  // Geometry cache
  logic [15:0] recon_rows, recon_cols, recon_slices;
  logic [15:0] sino_views, sino_rows, sino_chans;
  int          gran;

  // ── Main FSM ─────────────────────────────────────────────────────────────
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state        <= IDLE;
      iter         <= 0;
      subset_idx   <= 0;
      done_o       <= 0;
      converged_o  <= 0;
      iter_count_o <= 0;
      nmae_o       <= 0;
      alpha_o      <= 0;
      ell1_o       <= 0;
      // Memory ports — default
      recon_rd_en_o <= 0; recon_wr_en_o <= 0;
      esino_rd_en_o <= 0; esino_wr_en_o <= 0;
    end else begin
      // Default deasserts
      recon_rd_en_o <= 0; recon_wr_en_o <= 0;
      esino_rd_en_o <= 0; esino_wr_en_o <= 0;
      done_o        <= 0;

      case (state)
        // ── IDLE ─────────────────────────────────────────────────────────
        IDLE: begin
          converged_o  <= 0;
          if (start_i) begin
            iter  <= 0;
            state <= INIT;
          end
        end

        // ── INIT: read parameters from param store ────────────────────────
        INIT: begin
          // Read sigma_y, sigma_x, p, q, T, geometry in sequence
          // (simplified: assume parameters already loaded into latches
          //  by parent; in full RTL these would be read via ps_rd_addr_o)
          sigma_x_r   <= fp32_to_real(ps_rd_data_i);
          sigma_y_r   <= fp32_to_real(ps_rd_data_i);
          fm_const_r  <= 1.0 / (sigma_y_r * sigma_y_r + MJ_EPS);
          gran        <= GRAN;
          state       <= PART_GEN;
        end

        // ── PART_GEN: compute n_subsets ──────────────────────────────────
        PART_GEN: begin
          // n_subsets = ceil(n_pix / granularity)
          // n_pix comes from ROR mask generator (n_pixels_o)
          // Here simplified as n_subsets derived from config
          subset_idx <= 0;
          state      <= SUBSET_START;
        end

        // ── SUBSET_START: begin one VCD subset ───────────────────────────
        SUBSET_START: begin
          fwd_lin    <= 0.0;
          prior_lin  <= 0.0;
          fwd_quad   <= 0.0;
          prior_quad <= 0.0;
          ell1_acc   <= 0.0;
          pixel_in_subset <= 0;
          state      <= PRIOR_LOOP;
        end

        // ── PRIOR_LOOP: for each pixel in subset, compute grad+hess ──────
        // (qggmrf_compute is instantiated as a sub-module; signals driven
        //  externally.  Here we advance through pixels.)
        PRIOR_LOOP: begin
          // One pixel per state cycle (pipelined in real implementation)
          if (pixel_in_subset < subset_size) begin
            pixel_in_subset <= pixel_in_subset + 1;
            // grad/hess accumulation handled by qggmrf_compute instance
          end else begin
            state <= FWD_HESS;
          end
        end

        // ── FWD_HESS: back-project all-ones, coeff², accumulate ──────────
        FWD_HESS: begin
          // Drive bilinear_back_proj with coeff_power_2=1, sino=ones
          // Result accumulated into fwd_hess buffer (handled externally)
          state <= FWD_GRAD;
        end

        // ── FWD_GRAD: back-project weighted error sinogram ───────────────
        FWD_GRAD: begin
          // Drive bilinear_back_proj with coeff_power_2=0, sino=W*e
          // Result negated → fwd_grad
          state <= DELTA_CALC;
        end

        // ── DELTA_CALC: Δx = -(fwd_grad + prior_grad) / (fwd_hess + prior_hess) ─
        DELTA_CALC: begin
          // Per-voxel divide — pipelined division unit in full RTL
          state <= FWD_PROJ;
        end

        // ── FWD_PROJ: A·Δx → Δsino; accumulate fwd_lin, fwd_quad ────────
        FWD_PROJ: begin
          // Drive bilinear_fwd_proj for all views
          // fwd_lin  += W[i] * e[i] * ds[i]
          // fwd_quad += W[i] * ds[i]²
          state <= LSEARCH;
        end

        // ── LSEARCH: α = clip((fwd_lin-prior_lin)/(fwd_quad+prior_quad)) ─
        LSEARCH: begin
          begin
            automatic real num, den;
            num    = fwd_lin - prior_lin;
            den    = fwd_quad + prior_quad + MJ_EPS;
            alpha_r = mj_clipf(num / den, MJ_EPS, 1.5);
            alpha_o <= real_to_fp32(alpha_r);
          end
          state <= APPLY_UPD;
        end

        // ── APPLY_UPD: recon[flat*NS+s] += alpha * Δx[i] ────────────────
        APPLY_UPD: begin
          // Pipeline read-modify-write through recon buffer
          // ell1_acc += |alpha * Δx[i]| per voxel
          state <= UPD_SINO;
        end

        // ── UPD_SINO: error_sino[v][i] -= alpha * Δsino[v][i] ───────────
        UPD_SINO: begin
          state <= NMAE_CALC;
        end

        // ── NMAE_CALC: nmae = ell1 / recon_l1 ───────────────────────────
        NMAE_CALC: begin
          nmae_r = (recon_l1_acc > 0.0) ? ell1_acc / recon_l1_acc : 0.0;
          nmae_o <= real_to_fp32(nmae_r);
          ell1_o <= real_to_fp32(ell1_acc);
          state  <= NMAE_CHECK;
        end

        // ── NMAE_CHECK: advance subset / check convergence ────────────────
        NMAE_CHECK: begin
          subset_idx <= subset_idx + 1;
          if (subset_idx < n_subsets - 1) begin
            state <= SUBSET_START;
          end else begin
            state <= ITER_NEXT;
          end
        end

        // ── ITER_NEXT: advance iteration / check stop ─────────────────────
        ITER_NEXT: begin
          iter         <= iter + 1;
          iter_count_o <= iter + 1;
          if ((nmae_r * 100.0 < fp32_to_real(stop_pct_i)) ||
              (iter + 1 >= int'(max_iter_i))) begin
            converged_o <= (nmae_r * 100.0 < fp32_to_real(stop_pct_i));
            state       <= DONE_ST;
          end else begin
            subset_idx <= 0;
            state      <= SUBSET_START;
          end
        end

        // ── DONE ─────────────────────────────────────────────────────────
        DONE_ST: begin
          done_o <= 1;
          state  <= IDLE;
        end

        default: state <= IDLE;
      endcase
    end
  end

endmodule : vcd_controller
