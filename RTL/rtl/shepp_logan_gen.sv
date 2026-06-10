// =============================================================================
// shepp_logan_gen.sv
// SystemVerilog translation of phantom_shepp_logan() in phantom.c
//
// Generates the 3-D modified Shepp-Logan phantom into the recon buffer.
// 10 ellipsoids with pre-stored parameters (cx,cy,cz,ax,ay,az,phi,value).
//
// For each voxel (row,col,slice):
//   x = 2*(col+0.5)/cols - 1
//   y = 2*(row+0.5)/rows - 1
//   z = 2*(slice+0.5)/slices - 1
//   v = 0
//   for each ellipsoid e:
//     (dxr,dyr) = rotate (x-cx,y-cy) by phi
//     if (dxr/ax)²+(dyr/ay)²+(z-cz/az)² ≤ 1: v += value_e
//   out[...] = clip(v, 0, 1)
//
// Streaming output: one voxel per clock.
// =============================================================================
`timescale 1ns/1ps
import mbirjax_pkg::*;

module shepp_logan_gen #(
  parameter int MAX_ROWS   = 64,
  parameter int MAX_COLS   = 64,
  parameter int MAX_SLICES = 64
)(
  input  logic        clk, rst_n,
  input  logic        start_i,
  input  logic [15:0] rows_i, cols_i, slices_i,
  // Output: streaming voxel
  output logic        voxel_valid_o,
  output logic [19:0] voxel_addr_o,   // (row*cols+col)*slices+slice
  output logic [31:0] voxel_val_o,    // fp32
  output logic        done_o
);

  // Ellipsoid ROM (10 entries × 8 fields)
  localparam int N_ELLIPSOIDS = 10;

  // Stored as real arrays (synthesised as constant ROM)
  real sl_cx[0:9], sl_cy[0:9], sl_cz[0:9];
  real sl_ax[0:9], sl_ay[0:9], sl_az[0:9];
  real sl_phi[0:9], sl_val[0:9];

  initial begin
    // Modified Shepp-Logan table (low-dynamic-range)
    sl_cx  = '{ 0.0,  0.0,  0.22, -0.22, 0.0,    0.0,   0.0,  -0.08,  0.0,   0.06};
    sl_cy  = '{ 0.0, -0.0184, 0.0, 0.0,  0.35,   0.10, -0.10, -0.605,-0.606,-0.605};
    sl_cz  = '{ 0.0,  0.0,    0.0, 0.0,  0.0,    0.0,   0.0,   0.0,   0.0,   0.0 };
    sl_ax  = '{ 0.69, 0.6624, 0.11, 0.16, 0.21,  0.046, 0.046, 0.046, 0.023, 0.023};
    sl_ay  = '{ 0.92, 0.874,  0.31, 0.41, 0.25,  0.046, 0.046, 0.023, 0.023, 0.046};
    sl_az  = '{ 0.90, 0.880,  0.22, 0.28, 0.41,  0.05,  0.05,  0.05,  0.02,  0.02 };
    sl_phi = '{ 0.0,  0.0,  -18.0*MJ_PI/180.0, 18.0*MJ_PI/180.0,
                0.0,  0.0,   0.0,  0.0,  0.0,  0.0};
    sl_val = '{ 1.0, -0.8, -0.2, -0.2,  0.1,   0.1,   0.1,   0.1,   0.1,   0.1};
  end

  // Scan counters
  logic [15:0] row_c, col_c, slc_c;
  logic        active;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      row_c       <= 0; col_c  <= 0; slc_c  <= 0;
      active      <= 0;
      voxel_valid_o <= 0; done_o <= 0;
    end else begin
      voxel_valid_o <= 0;
      done_o        <= 0;
      if (start_i) begin
        row_c <= 0; col_c <= 0; slc_c <= 0;
        active <= 1;
      end
      if (active) begin
        // Compute voxel coordinates in [-1,1]
        automatic real x, y, z, v;
        automatic real dxr, dyr, cp, sp;
        x = 2.0 * (real'(col_c) + 0.5) / real'(cols_i) - 1.0;
        y = 2.0 * (real'(row_c) + 0.5) / real'(rows_i) - 1.0;
        z = 2.0 * (real'(slc_c) + 0.5) / real'(slices_i) - 1.0;
        v = 0.0;
        for (int e = 0; e < N_ELLIPSOIDS; e++) begin
          cp  = $cos(sl_phi[e]);
          sp  = $sin(sl_phi[e]);
          dxr = (x - sl_cx[e]) * cp + (y - sl_cy[e]) * sp;
          dyr =-(x - sl_cx[e]) * sp + (y - sl_cy[e]) * cp;
          if ((dxr/sl_ax[e])**2 + (dyr/sl_ay[e])**2 +
              ((z-sl_cz[e])/sl_az[e])**2 <= 1.0)
            v += sl_val[e];
        end
        v = mj_clipf(v, 0.0, 1.0);

        voxel_addr_o  <= 20'((row_c * cols_i + col_c) * slices_i + slc_c);
        voxel_val_o   <= real_to_fp32(v);
        voxel_valid_o <= 1;

        // Advance raster
        if (slc_c == slices_i - 1) begin
          slc_c <= 0;
          if (col_c == cols_i - 1) begin
            col_c <= 0;
            if (row_c == rows_i - 1) begin
              active <= 0;
              done_o <= 1;
            end else row_c <= row_c + 1;
          end else col_c <= col_c + 1;
        end else slc_c <= slc_c + 1;
      end
    end
  end

endmodule : shepp_logan_gen
