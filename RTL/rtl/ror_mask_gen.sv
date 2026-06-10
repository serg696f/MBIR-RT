// =============================================================================
// ror_mask_gen.sv
// SystemVerilog translation of mj_gen_ror_mask() in tomography_model.c
//
// Region-of-Reconstruction mask: marks pixels inside the largest inscribed
// circle of the rows×cols reconstruction grid.
//
// For pixel (row, col):
//   dx = col - (cols-1)/2
//   dy = row - (rows-1)/2
//   in_ror = (dx*dx + dy*dy) <= r²  where r = min(rows,cols)/2
//
// Output: one-hot mask signal for each (row,col) pair.
// Parameterised for synthesis by MAX_ROWS × MAX_COLS grid.
// =============================================================================
`timescale 1ns/1ps
import mbirjax_pkg::*;

module ror_mask_gen #(
  parameter int MAX_ROWS = 64,
  parameter int MAX_COLS = 64
)(
  input  logic clk, rst_n,
  input  logic [15:0] rows_i,   // actual reconstruction rows
  input  logic [15:0] cols_i,   // actual reconstruction cols
  input  logic        start_i,  // pulse to begin mask generation
  output logic        done_o,
  // Output FIFO-style: valid + pixel flat index
  output logic        pix_valid_o,
  output logic [19:0] pix_index_o,    // row*cols + col
  output logic [19:0] n_pixels_o      // total pixels in mask
);

  typedef enum logic [1:0] { IDLE, SCAN, DONE_ST } state_t;
  state_t state;

  logic [15:0] row_cnt, col_cnt;
  logic [31:0] cx2, cy2;   // 2*(centre) to avoid fractions
  logic [31:0] r2_4;       // 4*r² — scale everything by 4
  logic [31:0] dx2, dy2;
  logic [19:0] n_pix;

  // Compute scaled parameters from input dimensions
  // r² = (min(rows,cols)/2)²  → 4r² = min(rows,cols)²
  logic [15:0] min_dim;
  assign min_dim = (rows_i < cols_i) ? rows_i : cols_i;
  assign r2_4    = {16'd0, min_dim} * {16'd0, min_dim}; // min² = 4r²

  // Centre coordinates (in units of 1 pixel = multiplied by 2 for fractions)
  // cx = (cols-1)/2 → 2cx = cols-1, dy2 = (2*row - (rows-1))²
  // Use integer arithmetic: 4*(dx²+dy²) ≤ min² same as (2dx)²+(2dy)² ≤ min²

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state       <= IDLE;
      row_cnt     <= 0;
      col_cnt     <= 0;
      n_pix       <= 0;
      done_o      <= 0;
      pix_valid_o <= 0;
      pix_index_o <= 0;
      n_pixels_o  <= 0;
    end else begin
      pix_valid_o <= 0;
      done_o      <= 0;
      case (state)
        IDLE: begin
          if (start_i) begin
            row_cnt <= 0;
            col_cnt <= 0;
            n_pix   <= 0;
            state   <= SCAN;
          end
        end
        SCAN: begin
          // Compute 4*(dx² + dy²) in fixed point
          // 2*dx = 2*col - (cols-1)  (integer, exact)
          // 2*dy = 2*row - (rows-1)
          begin
            automatic logic signed [31:0] dx2r, dy2r;
            dx2r = $signed({16'd0, col_cnt} << 1) - $signed({16'd0, cols_i - 1});
            dy2r = $signed({16'd0, row_cnt} << 1) - $signed({16'd0, rows_i - 1});
            dx2  = dx2r * dx2r;
            dy2  = dy2r * dy2r;
          end

          if (dx2 + dy2 <= r2_4) begin
            pix_valid_o <= 1;
            pix_index_o <= {4'd0, row_cnt} * {4'd0, cols_i} + {4'd0, col_cnt};
            n_pix       <= n_pix + 1;
          end

          // Advance raster scan
          if (col_cnt == cols_i - 1) begin
            col_cnt <= 0;
            if (row_cnt == rows_i - 1) begin
              state      <= DONE_ST;
              n_pixels_o <= n_pix + ((dx2 + dy2 <= r2_4) ? 20'd1 : 20'd0);
            end else begin
              row_cnt <= row_cnt + 1;
            end
          end else begin
            col_cnt <= col_cnt + 1;
          end
        end
        DONE_ST: begin
          done_o <= 1;
          state  <= IDLE;
        end
      endcase
    end
  end

endmodule : ror_mask_gen
