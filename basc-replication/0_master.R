### Bayesian Alternative to Synthetic Control
## Pang, Liu, Xu

# A simulated example

rm(list=ls())
# set path, e.g.
# setwd("~/Dropbox/ProjectZ/pblasso/replication")


## ---------------------------------------------------------------------------------------------------------


## please install the following packages
# 1. install from files provided by the authors: pblasso, panelView, gsynth
# 2. install from CRAN
# install.packages("haven")
# install.packages("ggplot2")
# install.packages("Synth")
# install.packages("doParallel")
# install.packages("foreach")
# install.packages("abind")
# install.packages("xtable")


## ---------------------------------------------------------------------------------------------------------

# These 3 files replicate the three empirical examples
source("1_ex_sim0.R")
source("2_ex_adh2015.R")
source("3_ex_xu2017.R")

# The following 7 files conduct Monte Carlo exercises
source("4_sim_baseline.R")
source("5_sim_gamma.R")
source("6_sim_normalF.R")
source("7_sim_shortpanel.R")
source("8_sim_single_r3.R")
source("9_sim_single_r8.R")
source("10_sim_single_X.R")

# The following 3 files summarize results from Monte Carlo exercises
source("11_summ_baseline.R")
source("12_summ_single.R")
source("12_summ_single.R")

