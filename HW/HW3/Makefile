#==============================================================================#
# AIC2021 Project1 - TPU Design                                                #
# file: Makefile                                                               #
# description: Makefile for TPU testbench                                      #
# authors: kaikai (deekai9139@gmail.com)                                       #
#          suhan  (jjs93126@gmail.com)                                         #
#==============================================================================#

#------------------------------------------------------------------------------#
# Change your own verilog compiler.                                            #
#------------------------------------------------------------------------------#
# VERILOG=xrun
#VERILOG=ncverilog
VERILOG=iverilog

#------------------------------------------------------------------------------#
# Directories Declarations                                                     #
#------------------------------------------------------------------------------#
CUR_DIR=$(PWD)
TESTBENCH=TESTBENCH
RTL_DIR=RTL


verif1: clean
	python3 data_generator.py --mode 0 --target_dir verif1 --ncases 100
	cp verif1/input.txt verif1/input.bk
	mv verif1/input.txt $(TESTBENCH)/
	$(VERILOG) -o verif $(TESTBENCH)/TESTBENCH.v
	vvp verif

verif2: clean
	python3 data_generator.py --mode 1 --target_dir verif2 --ncases 100
	cp verif2/input.txt verif2/input.bk
	mv verif2/input.txt $(TESTBENCH)/
	$(VERILOG) -o verif $(TESTBENCH)/TESTBENCH.v
	vvp verif

verif3: clean
	python3 data_generator.py --mode 2 --target_dir verif3 --ncases 100
	cp verif3/input.txt verif3/input.bk
	mv verif3/input.txt $(TESTBENCH)/
	$(VERILOG) -o verif $(TESTBENCH)/TESTBENCH.v
	vvp verif

real: clean
	python3 data_generator.py --mode 3 --target_dir verif_real --ncases 100
	cp verif_real/input.txt verif_real/input.bk
	mv verif_real/input.txt $(TESTBENCH)/
	$(VERILOG) -o verif $(TESTBENCH)/TESTBENCH.v
	vvp verif

clean:
	rm -rf verif* TESTBENCH/input.txt verif
