// =============================================================================
// preprocess_unit.sv
// SystemVerilog translation of preprocess.c
//
// Streaming sinogram pre-processing:
//   MODE 0: transmission → attenuation   out = -log(max(in/air, 1e-6))
//   MODE 1: compute Poisson weights       out = exp(-|in|)
//   MODE 2: bad pixel correction (median of ±2 channel window)
//
// Streaming: one sample per clock in MODE 0/1.
// Mode 2 buffers a full row (5-element shift register).
// =============================================================================
`timescale 1ns/1ps
import mbirjax_pkg::*;

module preprocess_unit (
  input  logic        clk, rst_n,
  input  logic [1:0]  mode_i,       // 0=atten, 1=weights, 2=bad_pix
  // Air scan value (for mode 0; fixed 1.0 if air_scan disabled)
  input  logic [31:0] air_scan_i,   // fp32; use 0x3F800000 for unity
  // Streaming input
  input  logic        in_valid_i,
  input  logic [31:0] in_data_i,    // fp32
  output logic        in_ready_o,
  // Streaming output
  output logic        out_valid_o,
  output logic [31:0] out_data_o    // fp32
);

  localparam real LOG_MIN = 1e-6;

  // 5-sample shift register for bad-pixel correction (mode 2)
  logic [31:0] shift_reg [0:4];
  logic [2:0]  shift_cnt;
  logic        shift_full;

  assign in_ready_o = 1;  // always ready

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      out_valid_o <= 0;
      out_data_o  <= 0;
      shift_cnt   <= 0;
      shift_full  <= 0;
    end else begin
      out_valid_o <= 0;

      if (in_valid_i) begin
        case (mode_i)
          2'b00: begin
            // Transmission → attenuation
            automatic real s, a, ratio;
            s = fp32_to_real(in_data_i);
            a = fp32_to_real(air_scan_i);
            if (a <= 0.0) a = 1.0;
            ratio = s / a;
            if (ratio < LOG_MIN) ratio = LOG_MIN;
            out_data_o  <= real_to_fp32(-$ln(ratio));
            out_valid_o <= 1;
          end
          2'b01: begin
            // Poisson weights
            automatic real v;
            v = fp32_to_real(in_data_i);
            out_data_o  <= real_to_fp32($exp(-mj_absf(v)));
            out_valid_o <= 1;
          end
          2'b10: begin
            // Bad pixel correction: shift register median
            for (int i = 4; i > 0; i--) shift_reg[i] <= shift_reg[i-1];
            shift_reg[0] <= in_data_i;
            if (shift_cnt < 4) begin
              shift_cnt  <= shift_cnt + 1;
            end else begin
              shift_full <= 1;
            end
            if (shift_full) begin
              // Check centre (shift_reg[2]) for bad value
              automatic real center, candidates[4], sorted[4];
              automatic int  n_valid;
              center  = fp32_to_real(shift_reg[2]);
              n_valid = 0;
              // Collect non-center, non-bad values
              for (int k = 0; k < 5; k++) begin
                if (k != 2) begin
                  automatic real v2;
                  v2 = fp32_to_real(shift_reg[k]);
                  if (v2 == v2 && v2 >= 0.0) begin  // not NaN, not negative
                    candidates[n_valid] = v2;
                    n_valid++;
                  end
                end
              end
              // Output: if center is bad, use median of candidates; else pass through
              automatic logic center_bad;
              center_bad = (center != center) || (center < 0.0);  // NaN or negative
              if (center_bad && n_valid > 0) begin
                // Simple 4-element sort (bubble)
                for (int i = 0; i < n_valid-1; i++)
                  for (int j = i+1; j < n_valid; j++)
                    if (candidates[j] < candidates[i]) begin
                      automatic real tmp;
                      tmp = candidates[i]; candidates[i] = candidates[j]; candidates[j] = tmp;
                    end
                out_data_o <= real_to_fp32(candidates[n_valid/2]);
              end else begin
                out_data_o <= center_bad ? 32'h0 : shift_reg[2];
              end
              out_valid_o <= 1;
            end
          end
          default: begin
            out_data_o  <= in_data_i;
            out_valid_o <= 1;
          end
        endcase
      end
    end
  end

endmodule : preprocess_unit
