"""run selected seeds of one experiment: python run_one.py EXP seed [seed ...]"""
import sys
from run_growth_exp import main
main(sys.argv[1], seeds=[int(s) for s in sys.argv[2:]])
