### Bayesian Alternative to Synthetic Control
## Pang, Liu, Xu

# A simulated example


# set path, e.g.
# setwd("~/Dropbox/ProjectZ/pblasso/replication")


## ---------------------------------------------------------------------------------------------------------
rm(list=ls())
library(pblasso)
library(panelView)

## ---------------------------------------------------------------------------------------------------------
## gen simulated data
source("code/simulateSA.R")
source("code/summary_function.R")
source("code/plot_function.R")


## ---------------------------------------------------------------------------------------------------------
# beta
set.seed(1234)
N <- 50; TT = 30; p = 9
beta <- c(6,4,2,rep(0, 6))
# alpha
alpha <- matrix(0, N, p)
for (i in c(1:3)) {
    alpha[,i] <- rnorm(N, 0,beta[i]*0.5)
}
# xi (drift)
ar.coef <- 0.7
xi <- matrix(0, TT, p)
for (i in c(1:3)) {
    ts <- arima.sim(list(order = c(1,0,0), ar = ar.coef), TT)
    xi[,i] <- ts/sd(ts) * beta[i] * 0.5
}
niter <- 25000
burn <- 5000


## ---------------------------------------------------------------------------------------------------------
data <- simulateSA(N = N, TT = TT, r = 2, p = p,
    tr.threshold = 0.8, # e.g. tr.star>=0.5, D = 1, length(tr.threshold) = ntr - 1
    tr.start = 21, #length(tr.start) = ntr-1
    beta = beta, alpha = alpha, xi = xi, 
    mu=3, # grand mean
    force = 3, # both unit and time fixed effects 
    Rtype = "n", ## uniform (u)or normal (n)
    Ftype = c("ar1","ar1"), ## factor type: white noise, ar1, drift
    Fsize = c(2,2), ## influence of the factor
    eff.size = 1,
    eff.noise = 0.5,
    tr.noise = 0.5,
    ar.coef = ar.coef,
    time.invariant = FALSE,
    seed=1237)
names(data)
table(data$treat)/TT/N*100

table(data$time, data$treat)

plot(data$F1[1:TT], type = "l", ylim = c(-10,30))
lines(data$F2[1:TT], col = 2)


p.treat <- panelView(Y ~ D, index = c("id","time"), data = data, background = "white",
  xlab = "Time", ylab = "Unit", by.timing = TRUE, axis.lab = "time", main = "")

p.outcome <- panelView(Y ~ D, index = c("id","time"), data = data, type = "outcome", 
  theme.bw = TRUE, xlab = "Time", ylab = "Outcome", main = "", ylim = c(-50,50))


## ---------------------------------------------------------------------------------------------------------
index <- c("id", "time")
data <- data 
Yname <- "Y" 
Dname <- "D" 
Xname <- Zname <- Aname <- paste0("X",1:p)
r <- 10 
re <- "both"
y1 <- apply(matrix(data[which(data$treat==1),"eff"], 30, 5), 1, mean)
att <- cbind.data.frame(x1 = c(-19:10), y1 = y1)


## without factors
out0 <- pblasso(data = data, index = index, 
                Yname = Yname, Dname = Dname, 
                Xname = Xname, Zname = NULL, Aname = NULL, # no unit effect
                re = re, r = 0, niter = niter, burn = burn,
                xlasso = 1, zlasso = 0, alasso = 0, flasso = 0,
                a1 = 0.001, a2 = 0.001, ## hyper prior shrink on beta
                b1 = 0.001, b2 = 0.001, ## hyper prior shrink on alpha_i
                c1 = 0.001, c2 = 0.001, ## hyper prior shrink on xi_t
                p1 = 0.001, p2 = 0.001) ## hyper prior shrink on factor terms

## with factors
out1 <- pblasso(data = data, index = index, 
                Yname = Yname, Dname = Dname, 
                Xname = Xname, Zname = Xname, Aname = Xname, # no unit effect
                re = re, r = r, niter = niter, burn = burn,
                xlasso = 1, zlasso = 1, alasso = 1, flasso = 1,
                a1 = 0.001, a2 = 0.001, ## hyper prior shrink on beta
                b1 = 0.001, b2 = 0.001, ## hyper prior shrink on alpha_i
                c1 = 0.001, c2 = 0.001, ## hyper prior shrink on xi_t
                p1 = 0.001, p2 = 0.001) ## hyper prior shrink on factor terms

save(out0, out1, burn, data, file =  "./tempdata/sim0.RData")


## ---------------------------------------------------------------------------------------------------------


load("./tempdata/sim0.RData")
x1 <- 1:30
y1 <- apply(matrix(data[which(data$treat==1),"eff"], 30, 5), 1, mean)
att <- cbind.data.frame(x1 = c(-19:10), y1 = y1)
Xname <- Zname <- Aname <- paste0("X",1:9)

## ---------------------------------------------------------------------------------------------------------
## look at data
p.treat <- panelView(Y ~ D, index = c("id","time"), data = data, background = "white",
  xlab = "Time", ylab = "Unit", by.timing = TRUE, axis.lab = "time", main = "")
ggsave(p.treat, file = "./graphs/sim0_treat.pdf", height = 5, width = 7)

p.outcome <- panelView(Y ~ D, index = c("id","time"), data = data, type = "outcome", 
  theme.bw = TRUE, xlab = "Time", ylab = "Outcome", main = "", ylim = c(-40,50))
ggsave(p.outcome, file = "./graphs/sim0_outcome.pdf", height = 5, width = 7)



sout0 <- coefSummary(out0, burn = 0)
eout0 <- effSummary(out0, usr.id = NULL, burn = 0, cumu = FALSE, rela.period = TRUE)
sout1 <- coefSummary(out1, burn = 0)
eout1 <- effSummary(out1, usr.id = NULL, burn = 0, cumu = FALSE, rela.period = TRUE)

## ATT
p1.att <- effPlot(eout1, type = "eff", x1.pos = 0, y.pos = 0, 
                  ylim = c(-15,20), xlim = c(-19,10), 
                  xlab = "Time Relative to Treatment", ylab = "ATT",
                  cex.xlab = 15, cex.ylab = 15, 
                  cex.xaxis = 12, cex.yaxis = 12,
                  main = "")
p1.att <- p1.att + geom_line(aes(x1, y1), colour = "red", att, linetype = "dashed")
p1.att
ggsave(p1.att, file = "./graphs/sim0_att.pdf", height = 5, width = 6)

p0.att <- effPlot(eout0, type = "eff", x1.pos = 0, y.pos = 0, 
    ylim = c(-15,20), xlim = c(-19,10), 
    xlab = "Time Relative to Treatment", ylab = "ATT",
    cex.xlab = 15, cex.ylab = 15, 
    cex.xaxis = 12, cex.yaxis = 12,
    main = "")
p0.att <- p0.att + geom_line(aes(x1, y1), colour = "red", att, linetype = "dashed")
p0.att
ggsave(p0.att, file = "./graphs/sim0_att0.pdf", height = 5, width = 6)


p1.beta <- coefPlot(sout1, type = "beta", main = "",
                    labelname = c("Intercept", Xname), 
                    xlim  = c(-1, 8), 
                    xlab = "Coefficients", ylab = "",
                    cex.xlab = 15, cex.ylab = 15, 
                    cex.xaxis = 12, cex.yaxis = 12)
p1.beta
ggsave(p1.beta, file = "./graphs/sim0_beta.pdf", height = 5.2, width = 4)


ylim.v <- c(rep(c(-10,10),4),rep(c(-5,5),6))
ylim.list <- split(ylim.v, ceiling(seq_along(ylim.v)/2))
p1.xi <- coefPlot(sout1, type = "xi", plotlength = 2, 
                  labelname = c("Intercept", Aname), ## control title
                  main = NULL,            
                  legend = FALSE, ylim = ylim.list, xlim = NULL, rm1 = 0,
                  xlab = "Period", ylab = "Coefficients",
                  cex.xlab = 15, cex.ylab = 15, 
                  cex.xaxis = 12, cex.yaxis = 12)
p1.xi
ggsave(p1.xi, file = "./graphs/sim0_xi.pdf", height = 7, width = 15)


p1.alpha <- coefPlot(sout1, type = "alpha", plotlength = 10, 
  xlab = "Coefficient", ylab = "",
  labelname = c("Intercept", Zname), main = NULL, xlim = c(-10,10),            
  legend = 0, rm1 = 0, cex.xlab = 10, cex.xaxis = 10)
ggsave(p1.alpha, file = "./graphs/sim0_alpha.pdf", height = 5, width = 15)


## distribution of gamma^2
pdf("./graphs/sim0_wg.pdf", height = 4.5, width = 10)
dim(out1$gamma)
wg <- out1$wg
par(mfrow = c(2, 5), mar = c(3,3,2,1))
for (i in 1:10) {
    if (i %in% c(1,2)) {xlim = c(-10,10)} else {xlim = c(-4,4)}
    plot(density(wg[i,]), xlim = xlim, main = paste0("Factor ",i))
}
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

itt.trace <- function(x, unit, time) {    
    pos <- which(x$rela.time.tr == 5)[unit] ## post treatment period
    yo <- x$yo_t[pos] ## observed
    yct <- x$yct[pos,]
    niter <- dim(yct)[2]
    tt <- yo - yct
    return(tt)
}


## att trace
att <- att.trace(out1)
itt <- itt.trace(out1, 1, 5)  # unit 1, period 25



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

trace.att <- plot.trace(att, "Iterations", "ATT", c(3.8,6.2))
trace.itt <- plot.trace(itt, "Iterations", "TE(1, 25)", c(0,11))
ggsave(trace.att, file = "./graphs/sim0_trace1.pdf", height = 3, width = 10)
ggsave(trace.itt, file = "./graphs/sim0_trace2.pdf", height = 3, width = 10)

