### Bayesian Alternative to Synthetic Control
## Pang, Liu, Xu

#### Present simulation results for 
# sim_baseline
# sim_gamma
# sim_ar1
# sim_shortpanel

# please set path

rm(list=ls())
library(pblasso)
#library(panelView)

###################################
## Baseline
###################################

load("./tempdata/sim_baseline.RData")

## Table A1
print(storage)
out <- matrix(NA, 6, 20) # 6 cases
storage <- abs(storage)
colnames(out) <- rep(c("","Bias","SD","RMSE","Coverage"),4)
for (i in 1:6) {
  out[i, 2:5] <- storage[1,,i]
  out[i, 7:10] <- storage[2,,i]
  out[i, 12:15] <- storage[3,,i]
  out[i, 17:20] <- storage[4,,i]
}
NN <- rep(c(40, 80), each =3)
T0 <- rep(c(20, 40, 80), 3)
out <- cbind(NN, T0, out)
library(xtable)
print(xtable(out, digits = c(0, 0, 0, rep(2,20))), include.rownames = FALSE)
round(out,2)


#### Figure 4 ####
# RMSE
methods <- c(1,2,3,4)
mycol <- c("red",1,"red",1)
mypch = c(17,15,16,16)
mylty = c(2,1,2,1)
cex <- c(1,1,1,1.2)
pdf("./graphs/sim_baseline_rmse.pdf", height = 6, width = 6)
par(mar = c(3,3,1,1))
plot(1, type = "n", xlim = c(0.7, 3.3), ylim = c(0,4), 
  axes = FALSE, xlab = "", ylab = ""); box()
mtext("#Pre-treatment Periods",1,line = 2)
mtext("RMSE",2,line = 2)
axis(2); axis(1, at = c(1:3), labels = c(20,40,80))
for (i in 1:4) {
  j <- methods[i]
  lines(storage[j,3,1:3], type = "b", pch = mypch[i], col = mycol[i], lty = mylty[i], cex = cex[i])
}
legend("topright",legend = c("w/o factors, w/o covar   ",
  "w/ factors, w/o covar",
  "w/ factors & covar (fixed coef)",
  "w/ factors & covar (varying coef)"), bty = "n", lwd = 1.5,
  pch = mypch, col = mycol, lty = mylty)
graphics.off()

# Coverage
methods <- c(1,2,3,4)
mycol <- c("red",1,"red",1)
mypch = c(17,15,16,16)
mylty = c(2,1,2,1)
cex <- c(1,1,1,1.2)
pdf("./graphs/sim_baseline_cover.pdf", height = 6, width = 6)
par(mar = c(3,3,1,1))
plot(1, type = "n", xlim = c(0.7, 3.3), ylim = c(0,1), 
  axes = FALSE, xlab = "", ylab = ""); box()
abline(h = 0.95, col = "#AAAAAA50", lty = 1, lwd = 3)
mtext("#Pre-treatment Periods",1,line = 2)
mtext("Coverage",2,line = 2)
axis(2); axis(1, at = c(1:3), labels = c(20,40,80))
for (i in 1:4) {
  j <- methods[i]
  lines(storage[j,4,1:3], type = "b", pch = mypch[i], col = mycol[i], lty = mylty[i], cex = cex[i])
}
legend("bottomright",legend = c("w/o factors, w/o covar   ",
  "w/ factors, w/o covar",
  "w/ factors & covar (fixed coef)",
  "w/ factors & covar (varying coef)"), bty = "n", lwd = 1.5,
  pch = mypch, col = mycol, lty = mylty)
graphics.off()


###################################
## Gamma Error (Table A2)
###################################


load("./tempdata/sim_gamma.RData")
print(storage)

out <- matrix(NA, 6, 5) # 6 cases
storage <- abs(storage)
colnames(out) <- c("","Bias","SD","RMSE","Coverage")
for (i in 1:6) {
  out[i, 2:5] <- storage[1,,i]
}
NN <- rep(c(40, 80), each =3)
T0 <- rep(c(20, 40, 80), 3)
out <- cbind(NN, T0, out)
library(xtable)
print(xtable(out, digits = c(0, 0, 0, rep(2,5))), include.rownames = FALSE)
round(out,2)


###################################
## Normal vs AR1 Factors (Table A3)
###################################

load("./tempdata/sim_normalF.RData")
print(storage)

out <- matrix(NA, 6, 15) # 6 cases
storage <- abs(storage)
colnames(out) <- rep(c("","Bias","SD","RMSE","Coverage"), 3)
for (i in 1:6) {
  out[i, 2:5] <- storage[1,,i]
  out[i, 7:10] <- storage[2,,i]
  out[i, 12:15] <- storage[3,,i]
}
NN <- rep(c(50, 100), each =3)
T0 <- rep(c(20, 40, 80), 2)
out <- cbind(NN, T0, out)
library(xtable)
print(xtable(out, digits = c(0, 0, 0, rep(2,15))), include.rownames = FALSE)
round(out,2)


###################################
## Short panel (Table A4)
###################################


load("./tempdata/sim_shortpanel.RData")
print(storage)

out <- matrix(NA, 12, 5) # 6 cases
storage <- abs(storage)
colnames(out) <- c("","Bias","SD","RMSE","Coverage")
for (i in 1:12) {
  out[i, 2:5] <- storage[1,,i]
}
NN <- rep(50, 12)
mm <- rep(c(0.2, 0.4, 0.6), each =4)
T0 <- rep(c(5, 10, 15, 20), 3)
out <- cbind(mm, T0, out)
library(xtable)
print(xtable(out, digits = c(0, 2, 0, rep(2,5))), include.rownames = FALSE)
round(out,2)


