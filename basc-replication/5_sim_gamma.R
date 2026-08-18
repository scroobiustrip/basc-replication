### Bayesian Alternative to Synthetic Control
## Pang, Liu, Xu

#### Non-normal error

rm(list=ls())
library(pblasso)
#library(panelView)

## parallel computing
library(doParallel)
library(foreach)
library(abind)

## gen simulated data
source("code/simulateSA.R")
source("code/summary_function.R")
source("code/plot_function.R")

## simulation parameters
nsims <- 500

# Change data dimension
N.list <- c(50, 100)
T.list <- c(30, 50, 90) # T0: 20, 40, 80
#N.list <- c(50)
#T.list <- c(50)
cases <- cbind(rep(N.list, each = length(T.list)),
  rep(T.list, length(N.list)))
colnames(cases) <- c("N","TT")
ncases <- nrow(cases)

## storage results
models <- c("XF2")
storage <- array(NA, dim = c(length(models), 4, ncases))
dimnames(storage) <- list(models,
  c("bias","sd","rmse","coverage"),
  paste("case",1:ncases))

## fixed parameters
p <- 9 
beta <- c(6,4,2,rep(0, 6))

## draw fixed factors and xi (time series)
set.seed(12345)
T.max <- max(T.list)
ar.coef <- 0.7 

## blasso setup
index <- c("id", "time")
covar <- paste0("X",1:p)
r <- 2 # number of factors
r.max <- 10 
niter <- 10000 ## for MCMC

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
  
  this.case <- paste0("N = ", N, "; T0 = ", T0)
  cat(paste0("Case ",k,": ",this.case,"\n"))

  # alpha
  alpha <- matrix(0, N, p)
  for (i in c(1:3)) {
    alpha[,i] <- rnorm(N, 0,beta[i]*0.5)
  }

  # xi
  xi <- matrix(0,TT,p)
    for (j in c(1:3)) {
      ts <- arima.sim(list(order = c(1,0,0), ar = ar.coef), n = TT)
      xi[,j] <- ts/sd(ts) * beta[i] * 0.5
  }
  
  
  ## result is of dimension: 3 * 2 * nsims
  result <- foreach(i=1:nsims, .combine=f(),
    .inorder=FALSE,.packages = "pblasso") %dopar% {
     

    ## simulate data
    data <- simulateSA(N = N, TT = TT, r = r, p = p,
      tr.threshold = 0.8, # e.g. tr.star>=0.5,a D = 1, length(tr.threshold) = ntr - 1
      tr.start = (T0 + 1), #length(tr.start) = ntr-1
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
      error.type = "g",
      time.invariant = FALSE)    

    # ts <- 0.5*seq(from=0, length.out = TT, by=1) + 
    #    arima.sim(list(order = c(1,0,0), ar = ar.coef), n = TT)
    # #plot(ts)    

    ## true ATT
    true.eff <- mean(data$eff[which(data$D == 1)])

    # panelView(Y ~ D, index = c("id","time"), data = data, by.timing = TRUE)
    # panelView(Y ~ D, index = c("id","time"), data = data, type = "outcome", theme.bw = TRUE)

    tryCatch({

      ## Model 4: w/ factors and covariates; varying beta
      out <-  pblasso(data = data, index = index, 
        Yname = "Y", Dname = "D", 
        Xname = covar, Zname = covar, Aname = covar, 
        re = "both", r = r.max, niter = niter, ar1 = 1, burn = 2000,
        xlasso = 1, zlasso = 1, alasso = 1, flasso = 1)

      eout <- effSummary(out, usr.id = NULL, burn = 0, cumu = FALSE, rela.period = TRUE)
      est <- eout$est.avg[1]
      bias <- est - true.eff 
      cover <- ifelse(eout$est.avg[2] <= true.eff & eout$est.avg[3] >= true.eff, 1, 0) # coverage
      res <- c(est, bias, cover)

      output <- matrix(c(res), nrow = length(models), ncol = 3, byrow = TRUE) # 4 models * (bias, coverage, coverage)
      return(output)
    }, 
    error = function(err) {
       output <- matrix(NA, nrow = length(models), ncol = 3)
    }) # end of tryCatch

  } # end of foreach

  sd <- apply(result[,1,,drop = FALSE],1,sd,na.rm = TRUE)
  bias <- apply(result[,2,,drop = FALSE],1,mean,na.rm = TRUE)
  rmse <- sqrt(apply(result[,2,,drop = FALSE]^2, 1, mean,na.rm = TRUE))  
  coverage <- apply(result[,3,,drop = FALSE],1,mean,na.rm = TRUE)
  out <- cbind(bias, sd, rmse, coverage)
  print(out)
  storage[,,k] <- out

  save(storage, file = "./tempdata/sim_gamma.RData")

  print(Sys.time()-begin.time) 

}

stopCluster(cl) # stop parallel computing
print(storage)
print(Sys.time()-begin.time)

