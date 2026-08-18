### Bayesian Alternative to Synthetic Control
## Pang, Liu & Xu

#### Plot simulation results for 
# Single Treated Unit
# 1) 3 factors with covariates
# 2) 3 factors without covariates
# 3) 8 factors without covariates

rm(list=ls())
library(pblasso)
library(xtable)

## plot
load("./tempdata/sim_single_r3.RData")
round(storage,3)
# RMSE
methods <- c(2,4,5)
mycol <- c(1,"gray50",2)
mypch = c(17,15,16)
mylty = c(1,2,1)
pdf("./graphs/sim_single_rmse.pdf", height = 6, width = 6)
par(mar = c(3,3,1,1))
plot(1, type = "n", xlim = c(0.5, 3.5), ylim = c(0,6), 
  axes = FALSE, xlab = "", ylab = ""); box()
mtext("#Pre-treatment Periods",1,line = 2)
mtext("RMSE",2,line = 2)
axis(2); axis(1, at = c(1:3), labels = c(20,40,80))
for (i in 1:3) {
  j <- methods[i]
  lines(storage[j,3,1:3], type = "b", pch = mypch[i], col = mycol[i], lty = mylty[i])
  lines(storage[j,3,4:6], type = "b", pch = mypch[i], col = mycol[i], lty = mylty[i])
  if (i == 3) {
    text(1,storage[j,3,1]-0.1, "Nco = 30", cex = 0.6, pos = 2, col = mycol[i])
    text(1,storage[j,3,4], "Nco = 50", cex = 0.6, pos = 2, col = mycol[i])
  } else if (i == 2){
    text(1,storage[j,3,1], "Nco = 30", cex = 0.6, pos = 2)
    text(1,storage[j,3,4]+0.1, "Nco = 50", cex = 0.6, pos = 2)
  } else {
    text(1,storage[j,3,1], "Nco = 30", cex = 0.6, pos = 2)
    text(1,storage[j,3,4], "Nco = 50", cex = 0.6, pos = 2)
  }
}
legend("bottomright",legend = c("SCM","Gsynth","DM-LFM  "), bty = "n",
  pch = mypch, col = mycol, lty = mylty)
graphics.off()

# Coverage
methods <- c(4,5)
mycol <- c("gray50",2)
mypch = c(15,16)
mylty = c(2,1)
pdf("./graphs/sim_single_cover.pdf", height = 6, width = 6)
par(mar = c(3,3,1,1))
plot(1, type = "n", xlim = c(0.5, 3.5), ylim = c(0,1), 
  axes = FALSE, xlab = "", ylab = ""); box()
mtext("#Pre-treatment Periods",1,line = 2)
mtext("Coverage",2,line = 2)
axis(2); axis(1, at = c(1:3), labels = c(20,40,80))
abline(h = 0.95, lwd = 3, col = "#AAAAAA50")
for (i in 1:2) {
  j <- methods[i]
  lines(storage[j,4,1:3], type = "b", pch = mypch[i], col = mycol[i], lty = mylty[i])
  lines(storage[j,4,4:6], type = "b", pch = mypch[i], col = mycol[i], lty = mylty[i])
  text(1,storage[j,4,1], "Nco = 30", cex = 0.6, pos = 2)
  text(1,storage[j,4,4], "Nco = 50", cex = 0.6, pos = 2)
}
legend("bottomright",legend = c("Gsynth","DM-LFM  "), bty = "n",
  pch = mypch, col = mycol, lty = mylty)
graphics.off()

##################
## Tables
##################

## Table A5
load("./tempdata/sim_single_r3.RData")
round(storage,3)
NN <- rep(c(30, 50), each =3)
T0 <- rep(c(20, 40, 60), 2)
# Bias and RMSE
out <- matrix(NA, 6, 12) # 6 cases
colnames(out) <- rep(c("","synth1","synth2","gsynth1","gsynth2","bayes"),2)
out[,2:6] <- t(storage[,1,]) # Bias
out[,8:12] <- t(storage[,3,]) # RMSE
out <- cbind(NN, T0, out)
out1 <- out
# Coverage and Time
out <- matrix(NA, 6, 12) # 6 cases
colnames(out) <- rep(c("","synth1","synth2","gsynth1","gsynth2","bayes"),2)
out[,2:6] <- t(storage[,4,]) # Coverage
out[,8:12] <- t(storage[,6,]) # Time
out <- cbind(NN, T0, out)
out2 <- out
out <- rbind(out1, out2)
print(xtable(out, digits = c(0, 0, 0, rep(2,12))), 
  include.rownames = FALSE)

round(out1, 2) # upper panel
round(out2, 2) # lower panel

## Table A6
# r = 8; weak factors
load("./tempdata/sim_single_r8.RData")
round(storage,3)
NN <- rep(c(30, 50), each =3)
T0 <- rep(c(20, 40, 60), 2)
# Bias and RMSE
out <- matrix(NA, 6, 12) # 6 cases
colnames(out) <- rep(c("","synth1","synth2","gsynth1","gsynth2","bayes"),2)
out[,2:6] <- t(storage[,1,]) # Bias
out[,8:12] <- t(storage[,3,]) # RMSE
out <- cbind(NN, T0, out)
out1 <- out
# Coverage and Time
out <- matrix(NA, 6, 12) # 6 cases
colnames(out) <- rep(c("","synth1","synth2","gsynth1","gsynth2","bayes"),2)
out[,2:6] <- t(storage[,4,]) # Coverage
out[,8:12] <- t(storage[,6,]) # Time
out <- cbind(NN, T0, out)
out2 <- out
out <- rbind(out1, out2)
print(xtable(out, digits = c(0, 0, 0, rep(2,12))), 
  include.rownames = FALSE)

round(out1, 2) # upper panel
round(out2, 2) # lower panel

## Table A7
# r = 3; strong factors and covariates
load("./tempdata/sim_single_X.RData")
round(storage,3)
NN <- rep(c(30, 50), each =3)
T0 <- rep(c(20, 40, 60), 2)
# Bias and RMSE
out <- matrix(NA, 6, 12) # 6 cases
colnames(out) <- rep(c("","synth1","synth2","gsynth1","gsynth2","bayes"),2)
out[,2:6] <- t(storage[,1,]) # Bias
out[,8:12] <- t(storage[,3,]) # RMSE
out <- cbind(NN, T0, out)
out1 <- out
# Coverage and Time
out <- matrix(NA, 6, 12) # 6 cases
colnames(out) <- rep(c("","synth1","synth2","gsynth1","gsynth2","bayes"),2)
out[,2:6] <- t(storage[,4,]) # Coverage
out[,8:12] <- t(storage[,6,]) # Time
out <- cbind(NN, T0, out)
out2 <- out
out <- rbind(out1, out2)
print(xtable(out, digits = c(0, 0, 0, rep(2,12))), 
  include.rownames = FALSE)

round(out1, 2) # upper panel
round(out2, 2) # lower panel

