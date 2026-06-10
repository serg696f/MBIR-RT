// =============================================================================
// qggmrf_compute.sv
// SystemVerilog translation of qggmrf_grad_hess_subset() in vcd_utils.c
//
// Computes the qGGMRF prior gradient and Hessian diagonal for one voxel
// at (row, col, slice) by summing over all 26 neighbours.
//
// For each neighbour k:
//   delta  = x_center − x_nbr_k                 (neighbour clamped to edges)
//   grad  += b[k] · ρ'(delta)
//   hess  += b[k] · ρ''(delta)
//
// ρ'  and ρ'' computed as real arithmetic in automatic functions from pkg.
//
// Ports:
//   start_i      — begin computation for this voxel
//   center_val_i — fp32 voxel value at (row,col,slice)
//   nbr_val_i    — fp32 neighbour value (fetched externally, one per cycle)
//   nbr_ready_i  — neighbour value is valid this cycle
//   nbr_req_o    — request next neighbour {row,col,slice}
//   grad_o, hess_o — fp32 accumulated results (valid when done_o)
// =============================================================================
`timescale 1ns/1ps
import mbirjax_pkg::*;

module qggmrf_compute (
  input  logic        clk, rst_n,
  // qGGMRF parameters
  input  logic [31:0] sigma_x_i,  // fp32
  input  logic [31:0] p_i, q_i, T_i,  // fp32
  // Voxel being processed
  input  logic [15:0] voxel_row_i, voxel_col_i, voxel_slice_i,
  input  logic [15:0] recon_rows_i, recon_cols_i, recon_slices_i,
  // Control
  input  logic        start_i,
  // Neighbour interface (external memory controller)
  output logic        nbr_req_o,
  output logic [15:0] nbr_row_o, nbr_col_o, nbr_slice_o,
  input  logic [31:0] nbr_val_i,   // fp32
  input  logic        nbr_valid_i,
  // Center value
  input  logic [31:0] center_val_i,  // fp32
  // Results
  output logic        done_o,
  output logic [31:0] grad_o,   // fp32
  output logic [31:0] hess_o    // fp32
);

  typedef enum logic [1:0] { GH_IDLE, GH_REQ, GH_WAIT, GH_DONE } state_t;
  state_t gh_state;

  logic [4:0]  nbr_cnt;   // 0..25
  real         grad_acc, hess_acc;
  offset3_t    offs [0:25];
  real         b    [0:25];  // neighbour weights

  // Pre-compute offset table and weights
  initial begin
    get_nbr_offsets(offs);
    for (int k = 0; k < 26; k++)
      b[k] = nbr_weight(int'(offs[k].dz), int'(offs[k].dy), int'(offs[k].dx));
  end

  logic [15:0] cur_row, cur_col, cur_slc;
  real         ctr_val;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      gh_state  <= GH_IDLE;
      nbr_cnt   <= 0;
      nbr_req_o <= 0;
      done_o    <= 0;
      grad_acc  <= 0.0;
      hess_acc  <= 0.0;
      grad_o    <= 0;
      hess_o    <= 0;
    end else begin
      nbr_req_o <= 0;
      done_o    <= 0;
      case (gh_state)
        GH_IDLE: begin
          if (start_i) begin
            grad_acc  <= 0.0;
            hess_acc  <= 0.0;
            nbr_cnt   <= 0;
            ctr_val   <= fp32_to_real(center_val_i);
            gh_state  <= GH_REQ;
          end
        end
        GH_REQ: begin
          if (nbr_cnt == 26) begin
            gh_state <= GH_DONE;
          end else begin
            // Request neighbour k
            begin
              automatic int nr, nc, ns;
              nr = mj_clamp(int'(voxel_row_i)  + int'(offs[nbr_cnt].dy),
                            0, int'(recon_rows_i)  - 1);
              nc = mj_clamp(int'(voxel_col_i)  + int'(offs[nbr_cnt].dx),
                            0, int'(recon_cols_i)  - 1);
              ns = mj_clamp(int'(voxel_slice_i)+ int'(offs[nbr_cnt].dz),
                            0, int'(recon_slices_i)- 1);
              nbr_row_o   <= 16'(nr);
              nbr_col_o   <= 16'(nc);
              nbr_slice_o <= 16'(ns);
              nbr_req_o   <= 1;
            end
            gh_state <= GH_WAIT;
          end
        end
        GH_WAIT: begin
          if (nbr_valid_i) begin
            // Accumulate gradient and hessian
            automatic real delta, rho_p, rho_pp, bk;
            delta  = ctr_val - fp32_to_real(nbr_val_i);
            bk     = b[nbr_cnt];
            rho_p  = qggmrf_rho_prime (delta, fp32_to_real(sigma_x_i),
                                        fp32_to_real(p_i), fp32_to_real(q_i),
                                        fp32_to_real(T_i));
            rho_pp = qggmrf_rho_dprime(delta, fp32_to_real(sigma_x_i),
                                        fp32_to_real(p_i), fp32_to_real(q_i),
                                        fp32_to_real(T_i));
            grad_acc <= grad_acc + bk * rho_p;
            hess_acc <= hess_acc + bk * rho_pp;
            nbr_cnt  <= nbr_cnt + 1;
            gh_state <= GH_REQ;
          end
        end
        GH_DONE: begin
          grad_o   <= real_to_fp32(grad_acc);
          hess_o   <= real_to_fp32(hess_acc);
          done_o   <= 1;
          gh_state <= GH_IDLE;
        end
      endcase
    end
  end

endmodule : qggmrf_compute
