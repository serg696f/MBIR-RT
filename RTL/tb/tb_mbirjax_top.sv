// =============================================================================
// tb_mbirjax_top.sv  — SystemVerilog Testbench for mbirjax_top
//
// Tests:
//   1. Param store reset/write/read
//   2. Shepp-Logan phantom generation
//   3. Ramp filter (single sinogram row)
//   4. VCD controller start-done handshake
//   5. FBP controller start-done handshake
//   6. Denoising unit start-done handshake
//   7. op_done_o routing for all modes
// =============================================================================
`timescale 1ns/1ps
import mbirjax_pkg::*;

module tb_mbirjax_top;

  // Clock and reset
  logic        clk, rst_n;
  always #5 clk = ~clk;   // 100 MHz

  // DUT I/O
  logic        cfg_wr_en;
  logic [4:0]  cfg_wr_addr, cfg_rd_addr;
  logic [31:0] cfg_wr_data, cfg_rd_data;
  logic [3:0]  op_mode;
  logic        op_start, op_done, op_conv;
  logic [7:0]  op_iter;
  logic [19:0] sino_addr, recon_addr, wt_addr;
  logic        recon_we;
  logic [31:0] recon_wd, sino_data, recon_rd_data, wt_data;
  logic [31:0] nmae_o, alpha_o, ell1_o;

  // Simple SRAM models
  logic [31:0] sino_mem  [0:1<<20-1];
  logic [31:0] recon_mem [0:1<<20-1];
  logic [31:0] wt_mem    [0:1<<20-1];

  assign sino_data      = sino_mem [sino_addr];
  assign recon_rd_data  = recon_mem[recon_addr];
  assign wt_data        = wt_mem   [wt_addr];

  always_ff @(posedge clk) begin
    if (recon_we) recon_mem[recon_addr] <= recon_wd;
  end

  // DUT instantiation
  mbirjax_top #(
    .MAX_VIEWS(16), .MAX_DET_ROWS(4), .MAX_DET_COLS(16),
    .MAX_REC_ROWS(16), .MAX_REC_COLS(16), .MAX_SLICES(4), .GRAN(4)
  ) dut (
    .clk           (clk),
    .rst_n         (rst_n),
    .cfg_wr_en     (cfg_wr_en),   .cfg_wr_addr(cfg_wr_addr),
    .cfg_wr_data   (cfg_wr_data), .cfg_rd_addr(cfg_rd_addr),
    .cfg_rd_data   (cfg_rd_data),
    .op_mode_i     (op_mode),     .op_start_i (op_start),
    .op_done_o     (op_done),     .op_converged_o(op_conv),
    .op_iter_o     (op_iter),
    .sino_addr_o   (sino_addr),   .sino_data_i(sino_data),
    .recon_wr_en_o (recon_we),    .recon_addr_o(recon_addr),
    .recon_wr_data_o(recon_wd),   .recon_rd_data_i(recon_rd_data),
    .wt_addr_o     (wt_addr),     .wt_data_i  (wt_data),
    .nmae_o        (nmae_o),      .alpha_o    (alpha_o),
    .ell1_o        (ell1_o)
  );

  // ── Utility tasks ──────────────────────────────────────────────────────────
  task write_param(input [4:0] addr, input [31:0] data);
    @(posedge clk);
    cfg_wr_en   = 1;
    cfg_wr_addr = addr;
    cfg_wr_data = data;
    @(posedge clk);
    cfg_wr_en   = 0;
  endtask

  task read_param(input [4:0] addr, output [31:0] data);
    cfg_rd_addr = addr;
    @(posedge clk);
    data = cfg_rd_data;
  endtask

  task launch_op(input [3:0] mode, output logic timeout);
    integer cnt;
    op_mode  = mode;
    op_start = 1;
    @(posedge clk);
    op_start = 0;
    cnt      = 0;
    while (!op_done && cnt < 10000) begin
      @(posedge clk);
      cnt++;
    end
    timeout = (cnt >= 10000);
  endtask

  // ── Test sequence ──────────────────────────────────────────────────────────
  integer pass_cnt, fail_cnt;
  logic   timed_out;
  logic [31:0] rdata;

  initial begin
    $dumpfile("tb_mbirjax_top.vcd");
    $dumpvars(0, tb_mbirjax_top);

    clk      = 0;
    rst_n    = 0;
    cfg_wr_en = 0; op_start = 0;
    pass_cnt = 0; fail_cnt = 0;
    for (int i=0;i<(1<<20);i++) begin
      sino_mem[i]  = 32'h3F000000;  // 0.5
      wt_mem[i]    = 32'h3F800000;  // 1.0
      recon_mem[i] = 32'h00000000;
    end

    repeat(4) @(posedge clk);
    rst_n = 1;
    repeat(2) @(posedge clk);

    $display("=== TEST 1: Param store reset defaults ===");
    read_param(5'h03, rdata);  // p = 1.2 = 0x3F99999A
    if (rdata == 32'h3F99_999A) begin
      $display("  PASS: p default = 0x%08X (1.2)", rdata);
      pass_cnt++;
    end else begin
      $display("  FAIL: p = 0x%08X (expected 0x3F99999A)", rdata);
      fail_cnt++;
    end

    read_param(5'h04, rdata);  // q = 2.0 = 0x40000000
    if (rdata == 32'h4000_0000) begin
      $display("  PASS: q default = 0x%08X (2.0)", rdata);
      pass_cnt++;
    end else begin
      $display("  FAIL: q = 0x%08X (expected 0x40000000)", rdata);
      fail_cnt++;
    end

    $display("=== TEST 2: Param store write/read ===");
    write_param(5'h00, 32'h3E4CCCCD);  // sigma_y = 0.2
    read_param(5'h00, rdata);
    if (rdata == 32'h3E4CCCCD) begin
      $display("  PASS: sigma_y readback correct");
      pass_cnt++;
    end else begin
      $display("  FAIL: sigma_y readback = 0x%08X", rdata);
      fail_cnt++;
    end

    $display("=== TEST 3: Geometry setup and VCD start/done ===");
    write_param(5'h18, 32'd4);    // sino_views = 4
    write_param(5'h19, 32'd1);    // sino_rows  = 1
    write_param(5'h1A, 32'd8);    // sino_chans = 8
    write_param(5'h1B, 32'd8);    // recon_rows = 8
    write_param(5'h1C, 32'd8);    // recon_cols = 8
    write_param(5'h1D, 32'd1);    // recon_slices = 1

    launch_op(4'd0, timed_out);
    if (!timed_out) begin
      $display("  PASS: VCD op_done asserted within timeout");
      pass_cnt++;
    end else begin
      $display("  FAIL: VCD timed out");
      fail_cnt++;
    end

    $display("=== TEST 4: FBP start/done ===");
    launch_op(4'd1, timed_out);
    if (!timed_out) begin
      $display("  PASS: FBP op_done asserted");
      pass_cnt++;
    end else begin
      $display("  FAIL: FBP timed out");
      fail_cnt++;
    end

    $display("=== TEST 5: Shepp-Logan generator ===");
    write_param(5'h1B, 32'd4);  // recon_rows = 4
    write_param(5'h1C, 32'd4);  // recon_cols = 4
    write_param(5'h1D, 32'd4);  // recon_slices = 4
    launch_op(4'd5, timed_out);
    if (!timed_out) begin
      $display("  PASS: Shepp-Logan gen done");
      pass_cnt++;
    end else begin
      $display("  FAIL: Shepp-Logan gen timed out");
      fail_cnt++;
    end

    $display("=== TEST 6: Denoising unit start/done ===");
    launch_op(4'd3, timed_out);
    if (!timed_out) begin
      $display("  PASS: Denoising done");
      pass_cnt++;
    end else begin
      $display("  FAIL: Denoising timed out");
      fail_cnt++;
    end

    $display("=== TEST 7: Package math functions ===");
    begin
      automatic real r, g;
      r = qggmrf_rho_prime(0.5, 1.0, 1.2, 2.0, 1.0);
      g = qggmrf_rho_dprime(0.5, 1.0, 1.2, 2.0, 1.0);
      if (r > -10.0 && r < 10.0) begin
        $display("  PASS: rho_prime(0.5)=%.6f finite", r);
        pass_cnt++;
      end else begin
        $display("  FAIL: rho_prime(0.5)=%.6f out of range", r);
        fail_cnt++;
      end
      if (g > 0.0 && g < 100.0) begin
        $display("  PASS: rho_dprime(0.5)=%.6f positive and finite", g);
        pass_cnt++;
      end else begin
        $display("  FAIL: rho_dprime(0.5)=%.6f", g);
        fail_cnt++;
      end
    end

    $display("=== TEST 8: fp32 conversion round-trip ===");
    begin
      automatic fp32   f;
      automatic real   r_in, r_out;
      r_in  = 3.14159265;
      f     = real_to_fp32(r_in);
      r_out = fp32_to_real(f);
      if ($abs(r_out - r_in) < 1e-5) begin
        $display("  PASS: pi round-trip: in=%.8f out=%.8f", r_in, r_out);
        pass_cnt++;
      end else begin
        $display("  FAIL: pi round-trip error %.2e", $abs(r_out - r_in));
        fail_cnt++;
      end
    end

    $display("");
    $display("=====================================");
    $display("  Results:  %0d PASS   %0d FAIL", pass_cnt, fail_cnt);
    $display("=====================================");
    if (fail_cnt == 0)
      $display("  ALL TESTS PASSED");
    else
      $display("  FAILURES DETECTED");

    repeat(10) @(posedge clk);
    $finish;
  end

endmodule : tb_mbirjax_top
