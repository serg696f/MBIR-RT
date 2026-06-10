// =============================================================================
// ramp_filter.sv
// SystemVerilog translation of apply_ramp_1d() in parallel_beam_model.c
//
// 1-D Ram-Lak ramp filter applied along the detector channel axis.
// Direct-domain convolution kernel (FBP ramp):
//   h[0]        = 1/4
//   h[k odd]    = -1 / (k² π²)
//   h[k even≠0] = 0
//
// Architecture: streaming pipelined MAC.
//   - One sinogram row arrives sample-by-sample (in_valid / in_data).
//   - Filtered row emerges sample-by-sample (out_valid / out_data).
//   - Internal line buffer (shift register depth = HALF_KERNEL).
//   - Odd-kernel coefficients pre-computed at elaboration time.
//
// Parameterised:
//   HALF_KERNEL — number of non-zero kernel taps on each side (odd taps 1,3,..)
//   DATA_W      — fp32 data width (32)
// =============================================================================
`timescale 1ns/1ps
import mbirjax_pkg::*;

module ramp_filter #(
  parameter int HALF_KERNEL = 64,   // max odd tap index = 2*HALF_KERNEL - 1
  parameter int DATA_W      = 32,
  parameter int MAX_CHANS   = 512
)(
  input  logic              clk, rst_n,
  // Detector spacing (fp32) needed to scale output
  input  logic [31:0]       delta_det_ch_i,   // fp32
  // Streaming input
  input  logic              in_valid,
  input  logic [DATA_W-1:0] in_data,           // fp32 sinogram sample
  output logic              in_ready,
  // Streaming output
  output logic              out_valid,
  output logic [DATA_W-1:0] out_data            // fp32 filtered sample
);

  // ── Pre-compute integer kernel tap indices ─────────────────────────────────
  // Odd taps at k = 1, 3, 5, ..., (2*HALF_KERNEL - 1)
  // Coefficient h[k] = -1 / (k² π²)  stored as real

  // ── Line buffer (ring buffer of MAX_CHANS depth) ───────────────────────────
  logic [DATA_W-1:0] line_buf [0:MAX_CHANS-1];
  logic [9:0]        wr_ptr, rd_ptr;
  logic [9:0]        buf_len;

  // ── Accumulator registers ─────────────────────────────────────────────────
  // We implement a simplified version: for synthesis targets with no
  // transcendental functions, the filter coefficient ROM is pre-loaded.
  // The accumulation runs over HALF_KERNEL odd taps per output sample.

  typedef enum logic [1:0] { F_IDLE, F_CENTER, F_TAPS, F_EMIT } fstate_t;
  fstate_t fstate;

  logic [9:0]  tap_cnt;
  logic [9:0]  out_cnt;
  real         accum;
  logic [9:0]  center_ptr;

  // Coefficient ROM (pre-computed at elaboration)
  logic [31:0] coeff_rom [0:HALF_KERNEL-1];  // fp32 h[2k+1] for k=0..HK-1

  // Build coefficient ROM during initial block
  initial begin
    for (int k = 0; k < HALF_KERNEL; k++) begin
      // h[2k+1] = -1 / ((2k+1)^2 * pi^2)
      real hval;
      int  odd_k;
      odd_k = 2*k + 1;
      hval  = -1.0 / (real'(odd_k * odd_k) * MJ_PI * MJ_PI);
      coeff_rom[k] = real_to_fp32(hval);
    end
  end

  // ── Streaming FSM ─────────────────────────────────────────────────────────
  // Simplified: buffer full row then compute output row.
  // For real-time pipelining, a systolic array structure is preferable.

  assign in_ready = (fstate == F_IDLE || fstate == F_CENTER);

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      fstate   <= F_IDLE;
      wr_ptr   <= 0;
      rd_ptr   <= 0;
      buf_len  <= 0;
      tap_cnt  <= 0;
      out_cnt  <= 0;
      accum    <= 0.0;
      out_valid<= 0;
      out_data <= 0;
      center_ptr <= 0;
    end else begin
      out_valid <= 0;
      case (fstate)
        F_IDLE: begin
          wr_ptr  <= 0;
          buf_len <= 0;
          if (in_valid) begin
            line_buf[0] <= in_data;
            wr_ptr      <= 1;
            buf_len     <= 1;
            fstate      <= F_CENTER;
          end
        end
        F_CENTER: begin
          // Accumulate input samples into line buffer
          if (in_valid) begin
            line_buf[wr_ptr] <= in_data;
            wr_ptr           <= wr_ptr + 1;
            buf_len          <= buf_len + 1;
          end else begin
            // Input row done — switch to output mode
            out_cnt    <= 0;
            center_ptr <= 0;
            fstate     <= F_TAPS;
          end
        end
        F_TAPS: begin
          // Compute one output sample:
          // accum = line_buf[center] * 0.25 + sum over odd taps
          if (tap_cnt == 0) begin
            // Start accumulation with centre tap
            accum   <= fp32_to_real(line_buf[center_ptr]) * 0.25;
            tap_cnt <= 1;
          end else if (tap_cnt <= HALF_KERNEL) begin
            // Add odd tap at index (2*tap_cnt - 1) on both sides
            begin
              automatic int    off;
              automatic real   hval;
              automatic int    li, ri;
              off  = 2 * int'(tap_cnt) - 1;
              hval = fp32_to_real(coeff_rom[tap_cnt - 1]);
              li   = int'(center_ptr) - off;
              ri   = int'(center_ptr) + off;
              if (li >= 0)            accum <= accum + hval * fp32_to_real(line_buf[li]);
              if (ri < int'(buf_len)) accum <= accum + hval * fp32_to_real(line_buf[ri]);
            end
            tap_cnt <= tap_cnt + 1;
          end else begin
            // Emit output sample (divide by delta_det_ch)
            begin
              automatic real scaled;
              scaled  = accum / fp32_to_real(delta_det_ch_i);
              out_data  <= real_to_fp32(scaled);
              out_valid <= 1;
            end
            tap_cnt    <= 0;
            out_cnt    <= out_cnt + 1;
            center_ptr <= center_ptr + 1;
            if (center_ptr == buf_len - 1) fstate <= F_EMIT;
          end
        end
        F_EMIT: begin
          // All output samples emitted
          fstate <= F_IDLE;
        end
      endcase
    end
  end

endmodule : ramp_filter
