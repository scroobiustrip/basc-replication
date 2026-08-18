### Bayesian Alternative to Synthetic Control
## Pang, Liu, Xu

## Example: Xu (2017)

rm(list=ls())
library(panelView)
library(pblasso)
library(ggplot2)

source("code/simulateSA.R")
source("code/summary_function.R")
source("code/plot_function.R")

library(gsynth)
data(gsynth)

p.treat <- panelView(turnout ~ policy_edr, index = c("abb","year"), data = turnout, background = "white",
  xlab = "Time", ylab = "Unit", by.timing = TRUE, axis.lab = "time", main = "")
ggsave(p.treat, file = "./graphs/xu_treat.pdf", height = 5)

p.outcome <- panelView(turnout ~ policy_edr, index = c("abb","year"), data = turnout, type = "outcome", 
  theme.bw = TRUE, xlab = "Time", ylab = "Outcome", main = "")
ggsave(p.outcome, file = "./graphs/xu_outcome.pdf", height = 5)

set.seed(1234)

index <- c("abb", "year")
data <- turnout 
Yname <- "turnout" 
Dname <- "policy_edr" 
Xname <- c("policy_mail_in", "policy_motor")
Zname <- c("policy_mail_in", "policy_motor")
Aname <- c("policy_mail_in", "policy_motor")
r <- 10 
niter <- 25000
re <- "both"

N <- length(unique(data[, index[1]]))
TT <- length(unique(data[, index[2]]))

## sort data by treatment 
data <- data[order(data[, index[1]], data[, index[2]]), ]

n.treat <- tapply(data[, Dname], data[, index[1]], sum)
data$n.treat <- rep(n.treat, each = TT)

data <- data[order(-data[, "n.treat"], data[, index[1]], data[, index[2]]),]
data$id <- rep(1:N, each = TT)

index[1] <- "id"

xlasso <- 0
zlasso <- 1
alasso <- 1
flasso <- 1

## treated id 
s.n.treat <- sort(n.treat, decreasing = TRUE)
tr.name <- names(s.n.treat)[which(s.n.treat > 0)]

## with lasso prior 
out1 <- pblasso(data = data, index = index, 
                Yname = Yname, Dname = Dname, 
                Xname = Xname, Zname = Zname, Aname = Aname, 
                re = re, r = r, niter = niter, burn = 5000, 
                xlasso = xlasso, zlasso = zlasso,
                alasso = alasso, flasso = flasso)


dim(out1$gamma)
wg <- out1$wg
pdf("./graphs/xu_wg.pdf", height = 6, width = 14)
par(mfrow = c(2, 5), mar = c(3,3,2,1))
k <- 1
for (i in c(2,3,4,1,5,6,7,9,8,10)) {
  if (k == 1) {
    plot(density(wg[i,]), main = paste0("Factor ",k), xlim = c(-25,25))
  } else {
    plot(density(wg[i,]), main = paste0("Factor ",k), xlim = c(-10,10))
  } 
   k <- k + 1 
}
graphics.off()


## compare with gsynth
out2 <- gsynth(turnout~policy_edr+policy_mail_in+policy_motor, data = data, 
               index = c("id","year"), force = 3, r = 2, CV = 0, se = 1, 
               inference = "parametric", nboots = 1000)


## save result
save(data, out1, out2, file = "./tempdata/xu2017.RData")

###################################

load("./tempdata/xu2017.RData")

## summary coef 
sout1 <- coefSummary(out1, burn = 0)
eout1 <- effSummary(out1, usr.id = NULL, burn = 0, cumu = FALSE, rela.period = TRUE)

## estimated effects

## usr.id: if left blank, overall att will be calculated,
## cumu: whether to estimate cumulative effect or att at each time
## rela.period: report timing relative to treatment or true time (useful for one treated unit) 

## plot results

## plotlength : for random effects, columns of plots
## rm1 whether to plot intercept for beta, or random time or unit effect 
p1.beta <- coefPlot(sout1, type = "beta", plotlength = 5, labelname = c("Intercept", Xname), main = NULL,            
                    legend = TRUE, xlim = NULL, ylim = NULL, rm1 = 0)
p1.beta

p1.alpha <- coefPlot(sout1, type = "alpha", plotlength = 3, labelname = c("Intercept", Zname), main = NULL,            
                     legend = TRUE, xlim = c(-15, 15), ylim = NULL, rm1 = 0)
p1.alpha

p1.xi <- coefPlot(sout1, type = "xi", plotlength = 3, labelname = c("Intercept", Aname), main = NULL,            
                  legend = TRUE, ylim = c(-10, 10), xlim = NULL, rm1 = 0)
p1.xi


## plot att
p1.att <- effPlot(eout1, type = "eff", x1.pos = 0, y.pos = 0, 
                  ylim = c(-5,15), xlim = c(-(TT - 11),6),
                  main = "", xlab = "Term Relative to EDR Reform", 
                  ylab = "Effect on Turnout %", cex.xlab = 15, cex.ylab = 15, 
                  cex.xaxis = 12, cex.yaxis = 12)
ggsave(p1.att, file = "./graphs/xu_att1.pdf", height = 5, width = 6)

## gsynth result
est.eff <- as.data.frame(out2$est.att)
names(est.eff)[c(1, 3, 4)] <- c("estimated_ATT", "estimated_ATT_ci_l", 
                                "estimated_ATT_ci_u")
est.eff$time <- -(TT - 11):10
out2$est.eff <- est.eff
p2.att <- effPlot(out2, type = "eff", x1.pos = 0, y.pos = 0, 
  xlim = c(-(TT - 11),6), ylim = c(-5,15), 
  main = "", xlab = "Term Relative to EDR Reform", 
  ylab = "Effect on Turnout %", cex.xlab = 15, cex.ylab = 15, 
  cex.xaxis = 12, cex.yaxis = 12) 
ggsave(p2.att, file = "./graphs/xu_att2.pdf", height = 5, width = 6)




## -------------------------------------------------- ##
### Table A11 
## individual effects
for (i in 1:6) {
  # DM-LFM (right column)
  eout.ind <- effSummary(out1, usr.id = i, burn = 0, cumu = FALSE, rela.period = TRUE)
  p1.att <- effPlot(eout.ind, type = "eff", x1.pos = 0, y.pos = 0, ylim = c(-15,25), 
    main = "", xlab = "Term relative to reform", 
    ylab = "Turnout %") 
  ggsave(paste0("./graphs/xu_iatt",i,".pdf"), p1.att, width = 7, height = 4)

  # Gsynth (left column)
  est.eff <- as.data.frame(out2$est.ind[,,i])
  names(est.eff)[c(1, 3, 4)] <- c("estimated_ATT", "estimated_ATT_ci_l", 
    "estimated_ATT_ci_u")
  est.eff$time <- eout.ind$est.eff$time
  out2$est.eff <- est.eff
  p1.att <- effPlot(out2, type = "eff", x1.pos = 0, y.pos = 0, ylim = c(-15,25), 
    main = "", xlab = "Term relative to reform", 
    ylab = "Turnout %") 
  ggsave(paste0("./graphs/xu_iattg",i,".pdf"), p1.att, width = 7, height = 4)
}

#######################

## factors * gamma
getFL <- function(out, j, xlim, ylim, xlab, ylab, Tlabel){
  nsims <- dim(out$f)[3]
  TT <- dim(out$f)[1]
  n <- dim(out$ga)[1]
  FG <- array(NA, dim = c(TT, n, nsims))  # T* N* nsims
  for (i in 1:nsims) {
    FG[,,i] <- out$f[,j,i]%*%t(out$gamma[,j,i])*out$wg[j,i]
  }
  FG.mean <- apply(FG, c(1,2), mean)
  plot(1, type = "n", xlim = xlim, 
    xlab = xlab, ylab = ylab, ylim = ylim)  
  for (i in 1:ncol(FG.mean)) {
    lines(Tlabel, FG.mean[,i], col = "#AAAAAA70")
  }
  for (k in out$tr.unit.pos) {
    lines(Tlabel, FG.mean[,k], col = "blue")
  }
}



pdf("./graphs/xu_f1.pdf", height = 4.5, width = 6.5)
par(mar = c(4,4,0.5,3))
getFL(out1, j = 2, ylim = c(-50, 50), xlim = c(1920,2012),
  xlab = "Year", ylab = "Turnout %", Tlabel = out2$time)
legend("topright", c("EDR States","No EDR States"), col = c("blue","gray"), 
  cex = 1.2, lty = 1, lwd = 2, bty = "n") 
graphics.off()





## plot trace for att 
att.trace <- function(x) {    
    pos <- which(x$rela.time.tr >= 0) ## post treatment period
    yo <- x$yo_t[pos] ## observed
    yct <- x$yct[pos,]
    niter <- dim(yct)[2]
    att <- sapply(1:niter, function(i) {return(mean(yo - yct[,i ]))})
    return(att)
}


## att trace
att <- att.trace(out1)


library(ggplot2)
cex.title <- 20
cex.legend <- 15
cex.label <- 15
cex.axis <- 12

plot.trace <- function(x, xlab, ylab, ylim) {
  data <- cbind.data.frame(x, 1:length(x), as.factor(rep(1, length(x))))
  names(data) <- c("trace", "time", "chain")
  p <- ggplot(data) + xlab(xlab) +  ylab(ylab)
  p <- p + geom_line(aes(time, trace,
   colour = chain,
   group = chain))
  p <- p + theme_bw() + ylim(ylim)
  p <- p + theme(panel.grid.major = element_blank(),
   panel.grid.minor = element_blank(),
               #axis.line = element_blank(),
               #axis.ticks = element_blank(),
   axis.title=element_text(size=cex.label),
   axis.title.x = element_text(margin = margin(t = 8, r = 0, b = 0, l = 0)),
   axis.title.y = element_text(margin = margin(t = 0, r = 8, b = 0, l = 0)),
   axis.text = element_text(color="black", size=cex.axis),
   axis.text.x = element_text(size = cex.axis),
   axis.text.y = element_text(size = cex.axis),
               #plot.background = element_rect(fill = "gray90"),
               #legend.background = element_rect(fill = "gray90"),
   legend.position = "none",
   legend.margin = margin(c(0, 5, 5, 0)),
   legend.text = element_text(margin = margin(r = 10, unit = "pt"), size = cex.legend),
   legend.title=element_blank(),
   plot.title = element_text(size=cex.title, hjust = 0.5,face="bold",margin = margin(8, 0, 8, 0)))
  return(p)
}

trace.att <- plot.trace(att, "Iterations", "ATT", c(2.5,7.5))
ggsave(trace.att, file = "./graphs/xu_trace1.pdf", height = 3, width = 10)


