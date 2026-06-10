# mbirjax SystemVerilog — simulation Makefile
# Usage: make xsim | make vcs | make iverilog
TOP   ?= tb_mbirjax_top
FLIST := filelist.f
.PHONY: xsim vcs iverilog clean
xsim:
	xvlog --sv -f $(FLIST) --work work && \
	xelab -debug all $(TOP) -s snap && xsim snap --runall
vcs:
	vcs -sverilog -f $(FLIST) -o simv -debug_pp && ./simv
iverilog:
	iverilog -g2012 -f $(FLIST) -o sim && vvp sim
clean:
	rm -rf work xsim.dir *.log *.jou *.vcd simv sim csrc
