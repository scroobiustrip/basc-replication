### Bayesian Alternative to Synthetic Control
## Pang, Liu, Xu

#### Comparing Different Methods 
## for cases with a single treated unit
## 3 factors

rm(list=ls())
library(pblasso)
library(Synth)
library(gsynth)
library(panelView)

## parallel computing
library(doParallel)
library(foreach)
library(abind)

## gen simulated data
source("code/simulateCalib.R")
source("code/summary_function.R")
source("code/plot_function.R")

## simulation parameters
nsims <- 500

# Change data dimension
N.list <- c(30, 50) + 1 
T.list <- c(20, 40, 60) + 10 
#N.list <- 31
#T.list <- 35
cases <- cbind(rep(N.list, each = length(T.list)),
  rep(T.list, length(N.list)))
colnames(cases) <- c("N","TT")
cases
ncases <- nrow(cases)
ar.coef <- 0.6

## storage results
models <- c("synth1","synth2","gsynth1","gsynth2","bayes")
storage <- array(NA, dim = c(length(models), 6, ncases))
dimnames(storage) <- list(models,c("bias","sd","rmse","cover","fail","time"),paste("case",1:ncases))

## fixed parameters
p <- 0
beta <- NULL

## draw fixed factors and xi (time series)
set.seed(12345)
T.max <- max(T.list)

## blasso setup
index <- c("id", "time")
covar <- paste0("X",1:p)
r <- 3 # number of factors
r.max <- 10 
niter <- 10000 ## for MCMC
lambda.sd <- rep(1,r) * 4
lambda.mean <- rep(0,r)

## register multiple cores
cores<-detectCores()
cl<-makeCluster(cores)  # the cluster name should be reserved
registerDoParallel(cl)
cat("Nodes: ",cores,"\n",sep="")
f <- function(){function(...) abind(...,along=3)}



begin.time<-Sys.time()

for (k in 1:ncases) { 

  N <- cases[k,1]
  TT <- cases[k,2]
  T0 <- TT - 10
  
  this.case <- paste0("Nco = ", N-1, "; T0 = ", T0)
  cat(paste0("Case ",k,": ",this.case,"\n"))

  # time.effects  
  time.eff <- getTS(type = "drift", TT, ar.coef) * 8
  
  ## result is of dimension: 3 * 2 * nsims
  result <- foreach(i=1:nsims, .combine=f(),
    .inorder=FALSE,.packages = c("Synth","gsynth","pblasso")) %dopar% {

    ## simulate data
    data <- simulateCalib(N = N, 
      TT = TT, 
      tr.threshold = (N-1)/N, # e.g. tr.star>=0.5, D = 1, length(tr.threshold) = ntr - 1
      tr.start = (TT-10+1), #length(tr.start) = ntr-1
      tr.coef = c(0.1, 0.1, 0.1), 
      tr.noise = 1,
      p = 0, # number of covariates
      mu= 0, # grand mean
      force = 2, # 0 = none; 1 = unit, 2 = time, 3 = unit and time
      time.eff = time.eff, # time fixed effects 
      lambda.mean = lambda.mean,
      lambda.sd = lambda.sd,
      factors = NULL, # given fixed factor
      error.sd = 5,
      ar.coef = 0.6,
      seed=NULL)
    #panelView(Y ~ D, data = data, index = index, type = "outcome")
    #panelView(Y ~ D, data = data, index = index, type = "treat")

    tr.id <- unique(data$id[which(data$D==1)])
    co.id <- setdiff(unique(data$id),tr.id)
    
    #  control outcome matrix
    Y.tr <- data$Y[which(data$id == tr.id)]
    Y.co.mat <- matrix(data$Y[which(data$id != tr.id)], (N-1), TT, byrow = TRUE)
 
    ## true ATT 
    true.eff <- mean(data$eff[which(data$D == 1)])

    out1 <- out2 <- out3 <- out4 <- out5 <- rep(NA, 6) # c("bias","sd","rmse","cover","fail","time")
    output <- matrix(NA, nrow = length(models), ncol = 3)

    T0 <- TT - 10

    ## synth 
    est <- bias <- cover <- time <- NA
    tryCatch({

      T0.match <- floor(seq(1, T0, length.out = r))
      Y.list <- vector(mode = "list", length = length(T0.match))
      for (m in 1:length(T0.match)) {
        Y.list[[m]] <- list("Y",T0.match[m],"mean")
      }
      btime<-Sys.time()      
      dataprep.out <- dataprep(foo = data
        , predictors= NULL
        , special.predictors = Y.list
        , predictors.op = c("mean")
        , dependent     = c("Y")
        , unit.variable = c("id")
        , time.variable = c("time")
        , treatment.identifier = tr.id
        , controls.identifier  = co.id
        , time.predictors.prior = 1:T0
        , time.optimize.ssr    = setdiff(1:T0, T0.match)
        , time.plot  = 1:TT 
        )
      synth.out <- synth(data.prep.obj = dataprep.out)
      #path.plot(synth.res = synth.out, dataprep.res = dataprep.out)
      #gaps.plot(synth.res = synth.out, dataprep.res = dataprep.out)
      w.co <- synth.out$solution.w
      gap <- Y.tr - t(Y.co.mat)%*%w.co
      est <- mean(gap[(T0+1):TT])
      bias <- est - true.eff
      cover <- NA
      time <- difftime(Sys.time(), btime, unit = "secs") 
    }, 
    error = function(err) {
    }) # end of tryCatch  
    out1 <- c(est, bias, cover, time)

    ## synth 
    time <- NA
    tryCatch({

      T0.match <- floor(seq(1, T0, length.out = (T0-r)))
      Y.list <- vector(mode = "list", length = length(T0.match))
      for (m in 1:length(T0.match)) {
        Y.list[[m]] <- list("Y",T0.match[m],"mean")
      }
      btime<-Sys.time()      
      dataprep.out <- dataprep(foo = data
        , predictors= NULL
        , special.predictors = Y.list
        , predictors.op = c("mean")
        , dependent     = c("Y")
        , unit.variable = c("id")
        , time.variable = c("time")
        , treatment.identifier = tr.id
        , controls.identifier  = co.id
        , time.predictors.prior = 1:T0
        , time.optimize.ssr    = setdiff(1:T0, T0.match)
        , time.plot  = 1:TT 
        )
      synth.out <- synth(data.prep.obj = dataprep.out)
      #path.plot(synth.res = synth.out, dataprep.res = dataprep.out)
      #gaps.plot(synth.res = synth.out, dataprep.res = dataprep.out)
      w.co <- synth.out$solution.w
      gap <- Y.tr - t(Y.co.mat)%*%w.co
      est <- mean(gap[(T0+1):TT])
      bias <- est - true.eff
      cover <- NA
      time <- difftime(Sys.time(), btime, unit = "secs")
    }, 
    error = function(err) {
    }) # end of tryCatch  
    out2 <- c(est, bias, cover, time)

    ## gsynth (r known)
    est <- bias <- cover <- time <- NA
    btime<-Sys.time()      
    gsynth.out <-  gsynth(Y ~ D, data = data, index = index, force = "time", 
        CV = 0, r = r, se = TRUE, parallel = FALSE, inference = "parametric")
    est <- gsynth.out$att.avg
    bias <- est - true.eff
    ci <- gsynth.out$est.avg[3:4]
    cover <- ifelse(ci[1]<=true.eff & ci[2]>=true.eff, 1, 0)
    time <- difftime(Sys.time(), btime, unit = "secs")
    out3 <- c(est, bias, cover, time)
    
    ## gsynth (r unknown)
    time <- NA      
    btime<-Sys.time()      
    gsynth.out <-  gsynth(Y ~ D, data = data, index = index, force = "time", 
        CV = 1, r = c(0,10), se = TRUE, parallel = FALSE, inference = "parametric")
    est <- gsynth.out$att.avg
    bias <- est - true.eff
    ci <- gsynth.out$est.avg[3:4]
    cover <- ifelse(ci[1]<=true.eff & ci[2]>=true.eff, 1, 0)
    time <- difftime(Sys.time(), btime, unit = "secs")
    out4 <- c(est, bias, cover, time)
    
    ## pblasso: w/ factors, flexible parameters
    est <- bias <- cover <- time <- NA
    tryCatch({
      btime<-Sys.time()
      bout <-  pblasso(data = data, index = index, 
        Yname = "Y", Dname = "D", 
        Xname = NULL, Zname = NULL, Aname = NULL, 
        re = "time", r = r.max, niter = niter, burn = 2000,
        xlasso = 0, zlasso = 0, alasso = 0, flasso = 1) ## hyper prior shrink on factor terms
      eout <- effSummary(bout, usr.id = NULL, burn = 0, cumu = FALSE, rela.period = TRUE)
      est <- eout$est.avg[1]
      bias <- est - true.eff
      ci <- eout$est.avg[2:3]
      cover <- ifelse(ci[1]<=true.eff & ci[2]>=true.eff, 1, 0)
      time <- difftime(Sys.time(), btime, unit = "secs")
    }, 
    error = function(err) {
    }) # end of tryCatch 
    out5 <- c(est, bias, cover, time)        
     
    # save
    output <- matrix(c(out1, out2, out3, out4, out5), 
       nrow = length(models), ncol = 4, byrow = TRUE) #  #models * (bias, coverage, time)
    #return(output)     

  } # end of foreach

  sd <- apply(result[,1,],1,sd,na.rm = TRUE)
  bias <- apply(result[,2,],1,mean,na.rm = TRUE)
  rmse <- sqrt(apply(result[,2,]^2, 1, mean,na.rm = TRUE))  
  fail <- apply(is.na(result[,1,]),1,mean)
  cover <- apply(result[,3,],1,mean,na.rm = TRUE)
  time <- apply(result[,4,],1,mean,na.rm = TRUE)
  out <- cbind(bias, sd, rmse, cover, fail, time)
  rownames(out) <- models
  print(round(out,3))
  storage[,,k] <- out

  save(storage, file = "./tempdata/sim_single_r3.RData")

  print(Sys.time()-begin.time) 

}

stopCluster(cl) # stop parallel computing
print(storage)
print(Sys.time()-begin.time)

