################################################################################################
# Program: simulateSingle.R
# Author:	
# Aim: Generate simulated sample for single treated unit
# Revised: 2020/11/23
################################################################################################

getTS <- function(type = "ar1", TT, ar.coef) {
      if (type %in% c("ar1","drift")) {
        ts <- arima.sim(list(order = c(1,0,0), ar = ar.coef), n = TT)               
        } else {
        ts <- rnorm(TT)
        }
        if (type == "drift") {
           ts <- ts + 0.5*seq(from=0, length.out = TT, by=1)
        }
        return(ts)
}

simulateCalib<-function(
    N, 
    TT, 
    tr.threshold = 0.5, # e.g. tr.star>=0.5, D = 1, length(tr.threshold) = ntr - 1
    tr.start = 10, #length(tr.start) = ntr-1
    tr.coef = NULL, # c(-0.6, -0.4)
    p = 10, # number of covariates
    beta = c(4,4,2,2,rep(0, 6)),  
    time.invariant = FALSE,  
    alpha = NULL, 
    xi = NULL,    
    mu=0, # grand mean
    force, # 0 = none; 1 = unit, 2 = time, 3 = unit and time
    time.eff = NULL, # time fixed effects 
    alpha.sd = 1, # sd for unit fixed effects
    lambda.mean = NULL,
    lambda.sd = NULL,
    factors = NULL, # given fixed factor
    Rtype = "n", ## uniform (u)or normal (n)
    error.type = "n", # "n" for normal, "g" for gamma
    error.sd = 1,
    tr.noise = 0.5,
    ar.coef = NULL,
    seed=NULL
    ) {    

    # panel reshape: T*N -> TN * 1

    if (is.null(seed)==FALSE) {set.seed(seed)}

    # number of treatment status
    ntr <- length(tr.threshold) + 1

    # factors and loadings
    r <- length(lambda.sd)
    lambda <- matrix(NA, N, r)
    for (i in 1:r) {
        lambda[,i] <- rnorm(N, lambda.mean[i], lambda.sd[i])
    } 
    if (is.null(factors)==TRUE) {
        factors <- matrix(NA, TT, r)
        for (i in 1:r) {
            factors[,i] <- getTS(type = "ar1", TT, ar.coef)
        }
    }  


    ## covariates
    if (p>0) {

        if (time.invariant == FALSE) {
            X <- array(rnorm(N*TT*p), dim = c(TT, N, p))
        } else {
            W <- matrix(rnorm(N*p), nrow = N, ncol = p)
            X <- array(NA, dim = c(TT, N, p))
            for (i in 1:TT) {
                X[i,,] <- W
            }
        }
        ## beta fit 
        Xfit <- matrix(0, TT, N)
        for (i in 1:p) {
            Xfit <- Xfit + X[,,i] * beta[i]
        }
        ## alpha fit
        Zfit <- matrix(0, TT, N)
        if (is.null(alpha)==FALSE) {            
            for (i in 1:p) {
                Zfit <- Zfit + X[,,i] * matrix(rep(alpha[,i], each = TT), TT, N)
            }
        } else {
            alpha <- matrix(0, N, p)
        }
        ## xi fit
        if (is.null(xi)==TRUE) {  
            xi <- matrix(0, TT, p)
            for (i in c(1:p)) {
                xi[,i] <- getTS(type = "ar1", TT, ar.coef)
            }
        }
        Afit <- matrix(0, TT, N)
        for (i in 1:p) {
            Afit <- Afit + X[,,i] * matrix(rep(xi[,i], N), TT, N)
        }         
    } 

    # fixed effects
    if (force==1|force==3) { ## unit fixed effect
        unit.fe <- rnorm(N,sd=alpha.sd)
    } else {
        unit.fe <- rep(0, N) 
    }
    unitFE <- matrix(rep(unit.fe,each=TT),TT,N)
    if (force==2|force==3) { ## time fixed effect
        if (is.null(time.eff)==TRUE) {
            ts <- arima.sim(list(order = c(1,0,0), ar = ar.coef), TT) 
        } else {
            ts <- time.eff 
        }
        timeFE <- matrix(rep(ts,N),TT,N)
    }

    ## treatment assignment
    if (r == 1) {
        ps.raw <- tr.coef[1]*lambda[,1] + tr.coef[2]* unit.fe + rnorm(N, 0, tr.noise)
    } else {
        ps.raw <- tr.coef[1]*lambda[,1] + tr.coef[2]*lambda[,2] + tr.coef[3]* unit.fe + rnorm(N, 0, tr.noise)
    }
    if (p > 0 & time.invariant == TRUE) {
        ps.raw <- ps.raw + W[,1] * tr.coef[4] + W[,2]* tr.coef[5]
    }
    tr.star <- (ps.raw - min(ps.raw))/(max(ps.raw)-min(ps.raw)) # propensity score
    tr.rank <- rank(tr.star)
    tr.rank <- (tr.rank-1)/(max(tr.rank)-1) # bounded between from 0 to 1    
    D <- matrix(0, TT, N)
    treat <- rep(0, N) 
    T0 <- rep(TT, N)
    for (i in 1:(ntr-1)) {
        tr.id <- which(tr.rank*10000>=tr.threshold[i]*10000+1)
        treat[tr.id] <- i             
        T0[tr.id] <- tr.start[i]-1
        for (j in tr.id) {
            D[tr.start[i]:TT,j] <- 1  
        }
    }     
    
    ## disturbances
    e <- rnorm(TT*N,sd=error.sd)
    e <- matrix(e, TT, N)
    
    
    ## outcome variable
    Y <- matrix(mu, TT, N)
    
    if (r>0) {
        Y <- Y + factors%*%t(lambda)
    }
    if (p>0) {
        Y <- Y + Xfit + Zfit + Afit
    }
    
    # fixed effects
    if (force==1|force==3) { ## unit fixed effect
        Y <- Y + 1*unitFE
    }
    if (force==2|force==3) { ## time fixed effect
        Y <- Y + 1*timeFE
    }  
     

    ## treatment effect
    eff <- matrix(0,TT,N)
    # for (i in 1:N){
    #     eff[, i] <- rnorm(TT, mean = cumsum(D[, i]) * eff.size, sd = eff.noise)
    # }
    # eff <- eff*D
    Y <- Y + eff

    ## disturbance
    Y <- Y + e


    ## panel structure
    panel<-as.data.frame(cbind(
        rep(101:(100+N),each=TT),
        rep(1:TT,N),
        rep(1:TT,N),
        c(Y),
        c(e),
        c(eff),
        rep(T0, each = TT)))
    cname<-c("id","time","t","Y","error","eff","T0")

    ## treatment indicator
    treat <- rep(D[nrow(D),], each = TT)
    panel <- cbind(panel,c(D), treat)
    cname <- c(cname,"D","treat")

    ## covar
    if (p>0) {
        for (i in 1:p) {
            panel<-cbind(panel,c(X[,,i]))
            cname<-c(cname,paste("X",i,sep=""))
        }
    }    

    ## additive fixed effects
    if (force==1|force==3) {
        panel <- cbind(panel,c(unitFE))
        cname <- c(cname,"unitFE")
    }
    if (force==2|force==3) {

        panel <- cbind(panel,c(timeFE))
        cname <- c(cname,"timeFE")
    }
    
    if (r>0) {
        for (i in 1:r) {
            panel<-cbind(panel,rep(factors[,i],N))
            cname<-c(cname,paste("F",i,sep=""))
        }
        for (i in 1:r) {
            panel<-cbind(panel,rep(lambda[,i],each=TT))
            cname<-c(cname,paste("L",i,sep=""))
        }
    }
    colnames(panel)<-cname

    if (r>0) for (i in 1:r) {
        panel[,paste("FL",i,sep="")]<-panel[,paste("F",i,sep="")]*panel[,paste("L",i,sep="")]
    }

    ## return(list(panel=panel,lambda=lambda,lambda=lambda,factor=factor))
    return(panel)
}


