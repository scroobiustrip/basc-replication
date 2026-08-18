### Bayesian Alternative to Synthetic Control
## Pang, Liu, Xu

## Example: ADH (2015)

rm(list=ls())
library(haven)
library(panelView)
library(pblasso)
library(ggplot2)

source("code/simulateSA.R")
source("code/summary_function.R")
source("code/plot_function.R")


set.seed(1234)
data <- as.data.frame(read_dta("data/repgermany.dta"))
names(data)
data <- data[order(data$index, data$year), ]

#[1] "index"     "country"   "year"      "gdp"       "infrate"  
#[6] "trade"     "schooling" "invest60"  "invest70"  "invest80" 
#[11] "industry"

##################################

library(Synth)
dataprep.out <- dataprep(
  foo = data,
  predictors    = c("gdp","trade","infrate"),
  dependent     = "gdp",
  unit.variable = 1,
  time.variable = 3,
  special.predictors = list(
    list("industry", 1971:1980, c("mean")),
    list("schooling",c(1970,1975), c("mean")),
    list("invest70" ,1980, c("mean"))
  ),
  treatment.identifier = 7,
  controls.identifier = setdiff(unique(data$index),7),
  time.predictors.prior = 1971:1980,
  time.optimize.ssr = 1981:1990,
  unit.names.variable = 2,
  time.plot = 1960:2003
)

# fit training model
synth.out <- synth(data.prep.obj=dataprep.out, Margin.ipop=.005, Sigf.ipop=7, Bound.ipop=6)

path.plot(synth.res = synth.out,
 dataprep.res = dataprep.out,
 Ylab = c("Per Capita GDP (PPP, 2002 USD)"),
 Xlab = c("Year"), 
 Ylim = c(0,32000), 
 Legend = c("West Germany","Synthetic West Germany"))

gaps.plot(synth.res = synth.out,
 dataprep.res = dataprep.out, 
 Ylab = c("Gap in Per Capita GDP (PPP, 2002 USD)"),
 Xlab = c("Year"), 
 Ylim = c(-5000,5000))

#  control outcome matrix
N <- length(unique(data$country))
years <- unique(data$year)
TT <- length(years)
Y.tr <- data$gdp[which(data$country == "West Germany")]
data.co <- data[which(data$country != "West Germany"),]
Y.co.mat <- matrix(data.co$gdp, (N - 1), TT, byrow = TRUE)
w.co <- synth.out$solution.w
Y.ct <- t(Y.co.mat)%*%w.co
gap <- Y.tr - Y.ct
plot(years, Y.ct, type = "l", lty = 2)
lines(years, Y.tr)

out <- list()
est.eff <- as.data.frame(cbind(gap, gap, gap, gap))
names(est.eff)[c(1, 3, 4)] <- c("estimated_ATT", "estimated_ATT_ci_l", 
                                "estimated_ATT_ci_u")
est.eff$time <- years
out$est.eff <- est.eff

p0.att <- effPlot(out, type = "eff", xlab = "Year", 
  ylab = "Gap in Per Capita GDP (PPP, 2002 USD)",
  x1.pos = 1989, y.pos = 0, ylim = c(-5000, 5000),
  main = "", cex.xlab = 13, cex.ylab = 13, 
  cex.xaxis = 12, cex.yaxis = 12)
ggsave(p0.att, file = "./graphs/adh_att0.pdf", height = 5, width = 5.5)

out <- list()
est.eff <- as.data.frame(cbind(Y.tr, Y.ct))
names(est.eff)[c(1, 2)] <- c("observed","estimated_counterfactual")
est.eff$time <- years
out$est.eff <- est.eff

p1.ct <- effPlot(out, type = "outcome", main = "", 
  x1.pos = 1989, y.pos = NULL, ylim = c(0, 32000),
  xlab = "Year", ylab = "Per Capita GDP (PPP, 2002 USD)", 
  legend.pos = c(0.2,0.85), CI = FALSE, 
  legend.labs = c("West Germany","Estimated Y(0)"))
ggsave(p1.ct, file = "./graphs/adh_ct0.pdf", height = 5, width = 6)


##################################

## sort data 
data <- data[order(data[, "index"], data[, "year"]),]

## generate treatment indicator
D <- rep(0, dim(data)[1])
D[which(data$index == 7 & data$year >= 1990)] <- 1
data$D <- D

## save old data 
data.old <- data

## generate unit-invariant covariates
data$pgdp <- data$trade <- data$inflation <- data$industry <- data$schooling <- data$invest <- NA
label <- unique(data$index)

for (i in 1:length(label)) {
	pos <- which(data$index == label[i])
	subdata <- data.old[pos,]	
	data[pos, "pgdp"] <- mean(subdata[, "gdp"], na.rm = TRUE)
	data[pos, "trade"] <- mean(subdata[, "trade"], na.rm = TRUE)
	data[pos, "inflation"] <- mean(subdata[, "infrate"], na.rm = TRUE)
	data[pos, "industry"] <- mean(subdata[, "industry"], na.rm = TRUE)
	data[pos, "schooling"] <- mean(subdata[, "schooling"], na.rm = TRUE)
	data[pos, "invest"] <- mean(unlist(c(subdata[, c("invest60", "invest70", "invest80")])), na.rm = TRUE)
} 


index <- c("index", "year")
Yname <- "gdp" 
Dname <- "D" 
Aname <- Xname <- c("pgdp", "trade", "inflation", "industry", "schooling", "invest")
Zname <- NULL
r <- 10 
niter <- 25000
re <- "time"

p.outcome <- panelView(gdp ~ D, index = index, data = data, type = "outcome", 
  theme.bw = TRUE, xlab = "Time", ylab = "Outcome", main = "", ylim = c(0, 40000),
  legend.labs = c("Other Countries", "West Germany (Pre)", "West Germany (Post)"), )
ggsave(p.outcome, file = "./graphs/adh_outcome.pdf", height = 5)

p.treat <- panelView(gdp ~ D, index = c("country","year"), data = data, background = "white",
  xlab = "Time", ylab = "", by.timing = TRUE, main = "", axis.lab.gap = c(1, 0))
# ggsave(p.treat, file = "./graphs/adh_treat.pdf", height = 5)

## with flat prior 
out1 <- pblasso(data = data, index = index, 
                Yname = Yname, Dname = Dname, 
                Xname = Xname, Zname = Zname, Aname = Aname, 
                re = re, r = r, niter = niter, burn = 5000,
                xlasso = 0, zlasso = 0,
                alasso = 0, flasso = 1)


## placebo test
data2 <- data
D[which(data$index == 7 & data$year >= 1987)] <- 1
data2$D <- D
out2 <- pblasso(data = data2, index = index, 
                Yname = Yname, Dname = Dname, 
                Xname = Xname, Zname = Zname, Aname = Aname, 
                re = re, r = r, niter = niter, burn = 5000,
                xlasso = 0, zlasso = 0,
                alasso = 0, flasso = 1)

save(data, data2, out1, out2, file = "./tempdata/adh2015.RData")

### plotting
load("./tempdata/adh2015.RData")

## summary
sout1 <- coefSummary(out1, burn = 0)
sout2 <- coefSummary(out2, burn = 0)

## estimated effects
##overall 
eout1 <- effSummary(out1, usr.id = NULL, burn = 0, cumu = FALSE, rela.period = TRUE)
eout2 <- effSummary(out2, usr.id = NULL, burn = 0, cumu = FALSE, rela.period = TRUE)

eout1$est.eff$time <- 1960:2003
eout2$est.eff$time <- 1960:2003

## plot effects
p1.att <- effPlot(eout1, type = "eff", xlab = "Year", ylab = "Gap in Per Capita GDP (PPP, 2002 USD)",
                  x1.pos = 1989, y.pos = 0, ylim = c(-8000, 8000),
                  main = "", cex.xlab = 13, cex.ylab = 13, 
                  cex.xaxis = 12, cex.yaxis = 12)
p1.att

ggsave(p1.att, file = "./graphs/adh_att.pdf", height = 5, width = 5.5)

p2.att <- effPlot(eout2, type = "eff", xlab = "Year", ylab = "Gap in Per Capita GDP (PPP, 2002 USD)",
  x1.pos = 1989, y.pos = 0, ylim = c(-8000, 8000),
  main = "", cex.xlab = 13, cex.ylab = 13, 
  cex.xaxis = 12, cex.yaxis = 12) +
  geom_vline(xintercept=1986, linetype="dashed", colour = "gray50", size=0.5)
ggsave(p2.att, file = "./graphs/adh_att_p.pdf", height = 5, width = 5.5)


## counterfactual plot
source("code/plot_function.R")
p1.ct <- effPlot(eout1, type = "outcome", main = "", 
                  x1.pos = 1989, y.pos = NULL, ylim = c(0, 35000),
                  xlab = "Year", ylab = "Per Capita GDP (PPP, 2002 USD)", 
                  legend.pos = c(0.2, 0.85), 
                  legend.labs = c("West Germany","Estimated Y(0)", "95% CI"))
ggsave(p1.ct, file = "./graphs/adh_ct.pdf", height = 5, width = 6)



## plot results
p1.beta <- coefPlot(sout1, type = "beta", plotlength = 3, labelname = c("Intercept", Xname),             
                    legend = TRUE, xlim = NULL, ylim = NULL, rm1 = 0)
#ggsave(p1.beta, file = "./graphs/adh_beta.pdf", height = 7, width = 4)


ylim.v <- c(c(-20000, 20000),rep(c(-100, 100), 6))
ylim.list <- split(ylim.v, ceiling(seq_along(ylim.v)/2))
p1.xi <- coefPlot(sout1, type = "xi", plotlength = 2, 
                  labelname = c("Intercept", Aname), ## control title
                  main = NULL,            
                  legend = FALSE, ylim = ylim.list, xlim = NULL, rm1 = 0,
                  xlab = "Period", ylab = "Coefficients",
                  cex.xlab = 15, cex.ylab = 15, 
                  cex.xaxis = 12, cex.yaxis = 12)
ggsave(p1.xi, file = "./graphs/adh_xi.pdf", height = 8, width = 15)


## distribution of gamma^2
dim(out1$gamma)
wg <- out1$wg
pdf("./graphs/adh_wg.pdf", height = 6, width = 14)
par(mfrow = c(2, 5), mar = c(3,3,2,1))
k <- 1
for (i in 1:10) {
  if (k == 1) {
    plot(density(wg[i,]), main = paste0("Factor ", k), xlim = c(-7000, 7000))
  } else {
    plot(density(wg[i,]), main = paste0("Factor ", k), xlim = c(-3000, 3000))
  } 
   k <- k + 1 
}
graphics.off()


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



pdf("./graphs/adh_f1.pdf", height = 5, width = 7)
par(mar = c(4,3,0.5,3))
getFL(out1, j = 1, ylim = c(-10000,15000), xlim = c(1960,2005),
  xlab = "Year", ylab = "Per Capita GDP (PPP, 2002 USD)", Tlabel = 1960:2003)
legend("topleft", c("West Germany","Other Countries"), col = c("blue","gray"), 
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

trace.att <- plot.trace(att, "Iterations", "ATT", c(-2500,0))
ggsave(trace.att, file = "./graphs/adh_trace1.pdf", height = 3, width = 10)
