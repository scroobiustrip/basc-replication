## Bayesian Alternative to Synthetic Control

We are aiming to replicate the findings from the paper [A Bayesian Alternative to Synthetic Control for Comparative Case Studies](https://www.cambridge.org/core/services/aop-cambridge-core/content/view/C23BD67E4BBBB8C88ADAEAE169696A45/S104719872100022Xa.pdf/a-bayesian-alternative-to-synthetic-control-for-comparative-case-studies.pdf).

Given the following implementation in Python and NumPyro:

### Data Generating Process

```python
# data generating process
N = 50
T = 30
K = 10

t = np.arange(T)
T_treat = 21
P_treat = 0.1

Γ = rng.normal(size=(N, 2))
tr = Γ.dot([0.7, 0.3]) + rng.normal(0., 0.5, size=N)
treat_crit = np.percentile(tr, 100 * (1. - P_treat))
treated = tr > treat_crit
n_treated = treated.sum().item()

X = np.empty((N, K))
X[:, 0] = 1.
X[:, 1:] = rng.normal(size=(N, K - 1))
K_star = 4

B = np.zeros(K)
B[:K_star] = 3., 6., 4., 2.

A = np.zeros((N, K))
A[:, :K_star] = rng.normal(0., 0.5 * B[:K_star], size=(N, K_star))

import scipy as sp

def ar1(k, innov):
    t = np.arange(innov.shape[-1])
    expon = sp.linalg.toeplitz(t)
    return np.dot(innov, np.triu(np.power(k, expon)))

innov_Ξ = rng.normal(size=(K_star, T))

Ξ = np.zeros((K, T))

Ξ[:K_star] = ar1(0.6, innov_Ξ)

innov_F = rng.normal(size=(2, T))

F = ar1(0.7, innov_F)

Δ = np.zeros((N, T))
Δ[:, T_treat:] = (
    t[T_treat:] - T_treat + rng.normal(0., 0.5, size=(N, T - T_treat))
)

w = np.zeros((N, T))
w[treated, T_treat:] = 1
control_mask = ~np.asarray(w, dtype=bool)

Θ = (B + A)[..., np.newaxis] + Ξ[np.newaxis]
XΘ = (X[..., np.newaxis] * Θ).sum(axis=1)
y = Δ * w + XΘ + Γ.dot(F) + rng.normal(size=(N, T))
N_factor = 2

>> {'X': (50, 10), 'T': 30, 'y': (50, 30), 'control_mask': (50, 30), 'n_factor': 2}
```

### Model

```python
import numpyro
import numpyro.distributions as dist
from numpyro import deterministic, sample, plate, handlers
from numpyro.infer import MCMC, NUTS, Predictive, init_to_median


HALFNORMAL_SCALE = 1. / np.sqrt(1. - 2. / np.pi)

def noncentered_normal(name: str, shape: tuple[int, ...], μ: float = 0.):
    if μ is None:
        μ = sample(f"{name}_μ", dist.Normal(0, 5))
    Δ = sample(
        f"{name}_Δ", 
        dist.Normal(0, 1).expand(shape).to_event(len(shape))
    )
    σ = sample(f"{name}_σ", dist.HalfNormal(5 * HALFNORMAL_SCALE))
    return deterministic(f"{name}", μ + Δ * σ)


# model
def basc_model(
    T: int,
    X: Float[Array, "n_donor n_covariate"],
    Y: Float[Array, "time n_donor"] = None,
    control_mask: Bool[Array, "time n_donor"] = None,
    treated_idx: Int[Array, "n_donor time"] = None,
    n_factor: int = 10,
    sign_row: int | None = None,
    target_sign=None,
):
    """
    Bayesian alternative to synthetic control.
    `sign_row` / `target_sign` are None for the reference (unidentified-sign) fit
    and set for the final fit, exactly as in the PyMC two-stage procedure.
    """
    N, K = X.shape
    n_block = T - (n_factor + 1)

    # ---- covariate part -------------------------------------------------- #
    beta = sample("beta", dist.Normal(0, 5).expand([K]).to_event(1))

    alpha = noncentered_normal("alpha", shape=(N, K), μ=0.0)              # unit heterogeneity
    xi = noncentered_normal("xi", shape=(K, T), μ=0.0)                    # time dynamics

    theta = beta[None, :, None] + alpha[..., None] + xi[None, ...]        # (N, K, T)

    # ---- latent factor part ---------------------------------------------- #
    f_pos_row = sample(
        "f_pos_row",
        dist.HalfNormal(HALFNORMAL_SCALE).expand([n_factor]).to_event(1)
    )

    f_block_unid = sample(
        "f_block_unid",
        dist.Normal(0, 1).expand([n_block, n_factor]).to_event(2)
    )

    gamma_unid = sample(
        "gamma_unid",
        dist.Normal(0, 1).expand([N, n_factor]).to_event(2)
    )

    if sign_row is None:
        f_block, gamma = f_block_unid, gamma_unid
    else:
        # hard sign anchor: pins the orientation of each factor column so that
        # chains cannot settle in mirror-image modes.
        s = jnp.asarray(target_sign) * jnp.sign(f_block_unid[sign_row])
        f_block = deterministic("f_block", s * f_block_unid)
        gamma = deterministic("gamma", s * gamma_unid)

    f = jnp.concatenate(
        [
            jnp.eye(n_factor), 
            f_pos_row[None, :], 
            f_block
        ],
        axis=0
    )

    # ---- likelihood ------------------------------------------------------
    mu = deterministic(
        "mu", 
        (X[..., jnp.newaxis] * theta).sum(axis=1) 
        + jnp.dot(gamma, f.T)
    )

    sigma = sample("sigma", dist.HalfNormal(HALFNORMAL_SCALE))

    with handlers.mask(mask=control_mask):
        sample("likelihood", dist.Normal(mu, sigma), obs=Y)

    deterministic("counterfactual", mu[treated_idx, :])
```

Implement the following example from the original paper's R code, using the West Germany Reunification dataset:

**Note:** the original Stata file can be found at: https://github.com/jgreathouse9/mlsynth/blob/main/basedata/repgermany.dta


# simulateSA.R

```r
simulateSA<-function(
    N, 
    TT, 
    r, # number of factors
    tr.threshold = 0.5, # e.g. tr.star>=0.5, D = 1, length(tr.threshold) = ntr - 1
    tr.start = 10, #length(tr.start) = ntr-1
    p = 10, # number of covariates
    beta = c(4,4,2,2,rep(0, 6)),    
    alpha = NULL, 
    xi = NULL,    
    mu=0, # grand mean
    force, 
    factor = NULL, # given fixed factor
    Rtype = "n", ## uniform (u)or normal (n)
    Ftype = "drift", ## factor type: white noise, ar1, drift
    Fsize = 1, ## influence of the factor
    eff.size = 1,
    eff.noise = 1,
    tr.noise = 0.3,
    time.invariant = TRUE, # X_{it} does not change over time
    ar.coef = 0.7,
    error.type = "n", # "n" for normal, "g" for gamma
    seed=NULL
    ) {

    # panel reshape: T*N -> TN * 1

    if (is.null(seed)==FALSE) {set.seed(seed)}

    #############################
    ## Data generating process
    #############################

    getTS <- function(type = "ar1", TT, ar.coef) {
        if (type %in% c("ar1","drift")) {
            ts <- arima.sim(list(order = c(1,0,0), ar = ar.coef), n = TT)               
        } else {
            ts <- rnorm(TT)
        }
        if (type == "drift") {
            ts <- ts + seq(from=0, length.out = TT, by=1)
        }
        return(ts)
    }

    # number of treatment status
    ntr <- length(tr.threshold) + 1


    if (r > 0) {
         # loadings
        if (Rtype %in% c("u","uniform")) {
            bound <- sqrt(3)
            lambda<-matrix(runif(N*r, min = 0, max = 2*bound), N, r, byrow=TRUE)
        } else {
            lambda<-matrix(rnorm(N*r, 1, 1), N, r, byrow=TRUE) # mean = 1
        }      

        # factors
        if (is.null(factor) == TRUE) {
            factor<-matrix(NA,TT,r)   
            if (length(Ftype)==1) {Ftype <- rep(Ftype, r)}
            for (i in 1:r) {
                type <- Ftype[i]
                factor[,i] <- getTS(type, TT, ar.coef)
            }  
        }      
        if (length(Fsize)==1)  {Fsize <- rep(Fsize,r)}
        for (i in 1:r) {
            factor[,i] <- factor[,i]*Fsize[i]
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
        unitFE <- matrix(rep(rnorm(N,sd=1),each=TT),TT,N)
    }
    if (force==2|force==3) { ## time fixed effectW
        ts <- arima.sim(list(order = c(1,0,0), ar = ar.coef), TT) 
        timeFE <- matrix(rep(ts,N),TT,N)
    }

    ## treatment assignment
    tr.raw <- 0.7*lambda[,1] + 0.3*lambda[,2] + rnorm(N, 0, tr.noise)
    D <- matrix(0, TT, N)
    treat <- rep(0, N) 
    T0 <- rep(TT, N)
    for (i in 1:(ntr-1)) {
        sd.adj <- sqrt(0.7^2 + 0.3^2 + tr.noise^2)
        threshold <- qnorm(tr.threshold[i], mean = 1, sd = sd.adj) 
        # the theoretical mean is 1
        tr.id <- which(tr.raw * 10000 >= threshold * 10000 + 1) 
            
        treat[tr.id] <- i             
        T0[tr.id] <- tr.start[i]-1
        for (j in tr.id) {
            D[tr.start[i]:TT,j] <- 1  
        }
    } 
    
    ## disturbances
    if (error.type == "n") {
        e <- rnorm(TT*N,sd=1)
    } else if (error.type == "g") {
        e <- (rgamma(TT*N, shape = 2, scale = 2) - 4)/sqrt(8)
    }
    e <- matrix(e, TT, N)
    
    
    ## outcome variable
    Y <- matrix(mu, TT, N)
    
    if (r>0) {
        Y <- Y + factor%*%t(lambda)
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
    for (i in 1:N){
        eff[, i] <- rnorm(TT, mean = cumsum(D[, i]) * eff.size, sd = eff.noise)
    }
    eff <- eff*D
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
            panel<-cbind(panel,rep(factor[,i],N))
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
```

# summary_function.R


```r
## summary function for Bayesian Causal 
## 09/08/2019: Xun made a change by adding niter to the other two funcetions
##
## 1. treated and conterfactual outcomes
## 2. estimated atts
## 3. estimated beta
## 4. multi-level part
## 5. dynamic part
## 6.

### The plot does not plot the fourth parameter 
coefSummary <- function(x,                 ## estimation results
                        burn = 1000) {     ## burn-in length

    niter <- dim(x$sigma2_i)[2]
    out <- NULL

    ## ----------------------------- ##
    ## 1. beta
    beta_i <- x$beta
    if (!is.null(beta_i)) {
        p <- dim(beta_i)[1]

        beta_i <- matrix(c(beta_i[, (burn + 1):niter]), p, niter - burn)

        est_beta_mean <- apply(beta_i, 1, mean)
        est_beta_ci <- t(apply(beta_i, 1, quantile, c(0.025, 0.975)))
        data <- cbind.data.frame(est_beta_mean, est_beta_ci)

        names(data) <- c("mean", "ci_l", "ci_u")
        est.beta <- data

        out <- c(out, list(est.beta = est.beta))

    }


    ## ------------------------------ ##
    ## 2. multi-level coefficient 

    tr.unit.pos <- x$tr.unit.pos ## label treated units

    alpha_i <- x$alpha 
    wa_i <- x$wa
    if (!is.null(alpha_i)) {

        N <- dim(alpha_i)[1]
        p <- dim(alpha_i)[2]

        for (i in (burn + 1):niter) {
            alpha_i[,, i] <- alpha_i[,,i] * matrix(rep(wa_i[, i], each = N), N, p)
        }

        est_alpha_ci_l <- est_alpha_ci_u <- est_alpha_mean <- matrix(NA, N, p)

        for (i in 1:N) {
            sub_alpha <- matrix(alpha_i[i,,], p, niter)
            if (p == 1) {
                sub_alpha <- matrix(sub_alpha[, (burn + 1):niter], 1)
            } else {
                sub_alpha <- as.matrix(sub_alpha[, (burn + 1):niter])
            }
            est_alpha_mean[i,] <- apply(sub_alpha, 1, mean)
            est_alpha_ci_l[i,] <- apply(sub_alpha, 1, quantile, 0.025)
            est_alpha_ci_u[i,] <- apply(sub_alpha, 1, quantile, 0.975)
        }

        data <- NULL

        for (i in 1:p) {
            subdata <- cbind.data.frame(est_alpha_mean[, i], est_alpha_ci_l[, i], est_alpha_ci_u[, i])
            colnames(subdata) <- c("mean", "ci_l", "ci_u")
            subdata$id <- N:1
            subdata$tr <- 0
            subdata[tr.unit.pos, "tr"] <- 1
            subdata$tr <- as.factor(subdata$tr)
            data <- c(data, list(subdata))
        }

        est.alpha <- data
        out <- c(out, list(est.alpha = est.alpha))
    }


    ## ------------------------------ ##
    ## 3. time-varying coefficient 

    xi_i <- x$xi 
    wxi_i <- x$wxi
    
    if (!is.null(xi_i)) {
        TT <- dim(xi_i)[1]
        p <- dim(xi_i)[2]

        for (i in (burn + 1):niter) {
            xi_i[,, i] <- xi_i[,,i] * matrix(rep(wxi_i[, i], each = TT), TT, p)
        }

        est_xi_ci_l <- est_xi_ci_u <- est_xi_mean <- matrix(NA, TT, p)

        for (i in 1:TT) {
            sub_xi <- matrix(xi_i[i,,], p, niter)
            if (p == 1) {
                sub_xi <- matrix(sub_xi[, (burn + 1):niter], 1)
            } else {
                sub_xi <- as.matrix(sub_xi[, (burn + 1):niter])
            }
            est_xi_mean[i,] <- apply(sub_xi, 1, mean)
            est_xi_ci_l[i,] <- apply(sub_xi, 1, quantile, 0.025)
            est_xi_ci_u[i,] <- apply(sub_xi, 1, quantile, 0.975)
        }

        data <- NULL

        for (i in 1:p) {
            subdata <- cbind.data.frame(est_xi_mean[, i], est_xi_ci_l[, i], est_xi_ci_u[, i])
            colnames(subdata) <- c("mean", "ci_l", "ci_u")
            subdata$time <- 1:TT
            data <- c(data, list(subdata))
        }

        est.xi <- data
        out <- c(out, list(est.xi = est.xi))
    }


    ## ------------------------------ ##
    ## 4. ar1 coefficients

    p_i <- x$p

    if (!is.null(p_i)) {
        k <- dim(p_i)[1]

        p_i <- matrix(c(p_i[, (burn + 1):niter]), k, niter - burn)

        est_phi_mean <- apply(p_i, 1, mean)
        est_phi_ci <- t(apply(p_i, 1, quantile, c(0.025, 0.975)))
        data <- cbind.data.frame(est_phi_mean, est_phi_ci)

        names(data) <- c("mean", "ci_l", "ci_u")

        est.phi.xi <- est.phi.f <- NULL
        r <- k1 <- 0
        if (!is.null(x$xi)) {
            k1 <- dim(x$xi)[2]
            est.phi.xi <- data[1:k1,]
            out <- c(out, list(est.phi.xi = est.phi.xi))

        }
        if (!is.null(x$f)) {
            r <- dim(x$f)[2]
            est.phi.f <- data[(k1+1):(k1+r),]
            out <- c(out, list(est.phi.f = est.phi.f))
        }

    }

    return(out)
} 



## --------------------------------------- ##
## 5. observed and counterfactual outcomes  
## 6. plot by period att
## 7. plot cumulative att

effSummary <- function(x, 
                       usr.id = NULL,         ## individual effect, if left blank, all treated units will be used
                       burn = 1000, 
                       cumu = FALSE,          ## whether to calculate cumulative effect
                       expo = FALSE,
                       rela.period = TRUE) {  ## aggregate by time relative to treatment

    niter <- dim(x$sigma2_i)[2] 

    if (cumu) {
        rela.period <- TRUE
    }

    id.tr <- x$raw.id.tr
    time.tr <- x$time.tr
    rela.time.tr <- x$rela.time.tr  


    ## id indicator
    id.pos <- NULL
    unique.tr <- c(unique(id.tr))
    if (is.null(usr.id)) {
        id.pos <- 1:length(c(id.tr))  
    } else {
        if (sum(usr.id %in% unique.tr) != length(usr.id)) {
            stop("Some specified ids are not in treated group, please check input.\n")
        }
        id.pos <- which(c(id.tr) %in% usr.id)
    }

    yo_t <- NULL
    if (expo) {
        yo_t <- exp(x$yo_t)
    } else {
        yo_t <- x$yo_t
    }
    yo_t <- yo_t[id.pos]
    
    time.tr <- time.tr[id.pos]
    rela.time.tr <- rela.time.tr[id.pos]

    yct_i <- NULL
    if (expo) {
        yct_i <- exp(x$yct)
    } else {
        yct_i <- x$yct
    }
    
    yct_i <- matrix(c(yct_i[id.pos, (burn + 1):niter]), length(id.pos), niter - burn)

    ## mean observed and counterfactual
    count.tr <- NULL ## num of observations at each period

    if (rela.period) { ## relative to treatment occurrence
        m_yo <- tapply(yo_t, rela.time.tr, mean)
        m_yct <- sapply(1:(niter - burn), function(i){tapply(yct_i[, i], rela.time.tr, mean)})
        count.tr <- as.numeric(table(rela.time.tr))
    } else { ## real time period
        m_yo <- tapply(yo_t, time.tr, mean)
        m_yct <- sapply(1:(niter - burn), function(i){tapply(yct_i[, i], time.tr, mean)})

        count.tr <- as.numeric(table(rela.time.tr))
    }


    ## outcomes -------------------

    m_yct_mean <- apply(m_yct, 1, mean)
    m_yct_ci_l <- apply(m_yct, 1, quantile, 0.025)
    m_yct_ci_u <- apply(m_yct, 1, quantile, 0.975)


    ## effect ---------------------

    eff_i <- matrix(rep(c(m_yo), niter - burn), length(c(m_yo)), niter - burn) - m_yct

    eff_mean <- apply(eff_i, 1, mean)
    eff_ci_l <- apply(eff_i, 1, quantile, 0.025)
    eff_ci_u <- apply(eff_i, 1, quantile, 0.975)


    data <- cbind.data.frame(m_yo, m_yct_mean, m_yct_ci_u, m_yct_ci_l, eff_mean, eff_ci_l, eff_ci_u)
    names(data) <- c("observed", "estimated_counterfactual", 
                     "counterfactual_ci_l", "counterfactual_ci_u",
                     "estimated_ATT", "estimated_ATT_ci_l", "estimated_ATT_ci_u")
    if(rela.period) {
        data$time <- sort(unique(rela.time.tr))
        data$count <- count.tr
    } else {
        data$time <- sort(unique(time.tr))
    }

    est.eff <- data


    ## cumulative effects ---------
    est.cumu <- NULL
    if (cumu) {
        relatime <- sort(unique(rela.time.tr))

        st.pos <- which(relatime == 1) ## start point 

        eff_sub_i <- matrix(c(eff_i[st.pos:length(relatime), ]), length(relatime) - st.pos +1)

        eff_cumu_i <- matrix(NA, length(relatime) - st.pos +1, niter - burn)

        count.tr.sub <- count.tr[st.pos:length(relatime)]

        eff_cumu_i[1, ] <- eff_sub_i[1, ]
        if (length(relatime) - st.pos >= 2) {
            for (j in 2:(length(relatime) - st.pos +1)) {
                eff_cumu_i[j, ] <- sapply(1:(niter - burn), function(i) {sum(eff_sub_i[1:j, i] * count.tr.sub[1:j])/sum(count.tr.sub[1:j])} ) * j
            }
        }

        eff_cumu_mean <- apply(eff_cumu_i, 1, mean)
        eff_cumu_ci_l <- apply(eff_cumu_i, 1, quantile, 0.025)
        eff_cumu_ci_u <- apply(eff_cumu_i, 1, quantile, 0.975)

        data <- cbind.data.frame(eff_cumu_mean, eff_cumu_ci_l, eff_cumu_ci_u)
        names(data) <- c("mean", "ci_l", "ci_u")

        data$count <- count.tr[st.pos:length(relatime)]
        data$time <- relatime[st.pos:length(relatime)]

        est.cumu <- data
    }
    


    ## average effetcs ------------

    t.post <- which(rela.time.tr > 0)

    eff_avg_i <- sapply(1:(niter - burn), function(i) {mean(yo_t[t.post] - yct_i[t.post, i])})

    eff_avg_mean <- mean(eff_avg_i)
    eff_avg_ci_l <- quantile(eff_avg_i, 0.025)
    eff_avg_ci_u <- quantile(eff_avg_i, 0.975)


    est.avg <- cbind(eff_avg_mean, eff_avg_ci_l, eff_avg_ci_u)
    colnames(est.avg) <- c("mean", "ci_l", "ci_u")


    out <- list(est.eff = est.eff, 
                est.avg = est.avg)

    if (!is.null(est.cumu)) {
        out <- c(out, list(est.cumu = est.cumu))
    }

    return(out)

}
```

# plot_function.R


```r
library(ggplot2)
library(gridExtra)

## ---------------------------- 1.plot coefficient

coefPlot <- function(x,                      ## summary data
                     type,                   ## plot what: beta, alpha, xi,
                     plotlength = 5,         ## for multi-level and time-varying, by row or column
                     labelname = NULL,       ## annotation
                     main = NULL,            ## title
                     legend = TRUE,          ## show legend 
                     xlim = NULL,
                     ylim = NULL,
                     xlab = NULL,
                     ylab = NULL,
                     cex.xlab = 15,
                     cex.ylab = 15,
                     cex.xaxis = 12,
                     cex.yaxis = 12,
                     cex.main = 15, 
                     cex.legend = 12,  
                     rm1 = FALSE,
                     legend.pos = NULL) {    ## show intercept

    if (is.null(legend.pos)==TRUE) {
        legend.pos <- ifelse(legend, "bottom", "none")
    }
    
    main <- ifelse(is.null(main), type, main)
    if (type != "xi") {
        xlab <- ifelse(is.null(xlab), "Covar", xlab)
        if (type != "alpha") {
          ylab <- ifelse(is.null(ylab), "Coef", ylab)
        } else {
          ylab <- ifelse(is.null(ylab), "Units", ylab)
        }
    } else {
        xlab <- ifelse(is.null(xlab), "Time", xlab)
        ylab <- ifelse(is.null(ylab), "Coef", ylab)
    }

    if (type == "beta") {
        ## ------------------------- 1. plot beta
              
        data <- x$est.beta
        if (rm1 == TRUE) {
            data <- data[-1,]
        }

        data$id <- dim(data)[1]:1

        p <- ggplot(data, aes(mean, id)) 
        p <- p + geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50")

        p <- p + geom_point() + geom_errorbarh(aes(xmax = ci_u, xmin = ci_l, height = .2))
        p <- p + xlab(xlab) + ylab(ylab)
        p <- p + ggtitle(main)

        p <- p + scale_y_continuous(expand = c(0, 0), breaks = data$id, labels = labelname, limits = c(min(data$id) - 0.3, max(data$id) + 0.3))
        p <- p + theme_bw(base_size = 15)
        p <- p + theme(panel.grid.major = element_blank(),
                       panel.grid.minor = element_blank(),
                       legend.text = element_text(margin = margin(r = 10, unit = "pt"), size = cex.legend),
                       legend.position = legend.pos,
                       axis.title = element_text(size=12),
                       axis.title.x = element_text(margin = margin(t = 8, r = 0, b = 0, l = 0), size = cex.xlab),
                       axis.title.y = element_text(margin = margin(t = 0, r = 8, b = 0, l = 0), size = cex.ylab),
                       axis.text = element_text(color="black", size=8),
                       axis.text.x = element_text(size = cex.xaxis),
                       axis.text.y = element_text(size = cex.yaxis),
                       plot.title = element_text(size = cex.main,
                                                 hjust = 0.5,
                                                 face="bold",
                                                 margin = margin(10, 0, 10, 0)))
        if (!is.null(xlim)) {
            p <- p + coord_cartesian(xlim = xlim)
        }

        #if (!is.null(ylim)) {
        #    p <- p + coord_cartesian(ylim = ylim)
        #}
        p

    }
    else if (type == "phi_xi") {
        ## plus1: phi_xi
        data <- x$est.phi.xi
        if (intercept == FALSE) {
            data <- data[-1,]
        }

        data$id <- dim(data)[1]:1

        p <- ggplot(data, aes(mean, id)) 
        p <- p + geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50")

        p <- p + geom_point() + geom_errorbarh(aes(xmax = ci_u, xmin = ci_l, height = .2))
        p <- p + xlab(xlab) + ylab(ylab)
        p <- p + ggtitle(main)

        p <- p + scale_y_continuous(expand = c(0, 0), breaks = data$id, labels = labelname, limits = c(min(data$id) - 0.3, max(data$id) + 0.3))
        p <- p + theme_bw(base_size = 15)
        p <- p + theme(panel.grid.major = element_blank(),
                       panel.grid.minor = element_blank(),
                       legend.text = element_text(margin = margin(r = 10, unit = "pt"), size = cex.legend),
                       legend.position = legend.pos,
                       axis.title = element_text(size=12),
                       axis.title.x = element_text(margin = margin(t = 8, r = 0, b = 0, l = 0), size = cex.xlab),
                       axis.title.y = element_text(margin = margin(t = 0, r = 8, b = 0, l = 0), size = cex.ylab),
                       axis.text = element_text(color="black", size=8),
                       axis.text.x = element_text(size = cex.xaxis),
                       axis.text.y = element_text(size = cex.yaxis),
                       plot.title = element_text(size = cex.main,
                                                 hjust = 0.5,
                                                 face="bold",
                                                 margin = margin(10, 0, 10, 0)))
        if (!is.null(xlim)) {
            p <- p + coord_cartesian(xlim = xlim)
        }

        #if (!is.null(ylim)) {
        #    p <- p + coord_cartesian(ylim = ylim)
        #}
        #main <- ifelse(is.null(main), "Phi_xi", main) 
        #p <- p + ggtitle(main)
        p
    }
    else if (type == "phi_f") {
        ## plus2: phi_f
        data <- x$est.phi.f
        data$id <- dim(data)[1]:1

        p <- ggplot(data, aes(mean, id)) 
        p <- p + geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50")

        p <- p + geom_point() + geom_errorbarh(aes(xmax = ci_u, xmin = ci_l, height = .2))
        p <- p + xlab(xlab) + ylab(ylab)
        p <- p + ggtitle(main)

        p <- p + scale_y_continuous(expand = c(0, 0), breaks = data$id, labels = labelname, limits = c(min(data$id) - 0.3, max(data$id) + 0.3))
        p <- p + theme_bw(base_size = 15)
        p <- p + theme(panel.grid.major = element_blank(),
                       panel.grid.minor = element_blank(),
                       legend.text = element_text(margin = margin(r = 10, unit = "pt"), size = cex.legend),
                       legend.position = legend.pos,
                       axis.title = element_text(size=12),
                       axis.title.x = element_text(margin = margin(t = 8, r = 0, b = 0, l = 0), size = cex.xlab),
                       axis.title.y = element_text(margin = margin(t = 0, r = 8, b = 0, l = 0), size = cex.ylab),
                       axis.text = element_text(color="black", size=8),
                       axis.text.x = element_text(size = cex.xaxis),
                       axis.text.y = element_text(size = cex.yaxis),
                       plot.title = element_text(size = cex.main,
                                                 hjust = 0.5,
                                                 face="bold",
                                                 margin = margin(10, 0, 10, 0)))

        if (!is.null(xlim)) {
            p <- p + coord_cartesian(xlim = xlim)
        }

        #if (!is.null(ylim)) {
        #    p <- p + coord_cartesian(ylim = ylim)
        #}
        
        #main <- ifelse(is.null(main), "Phi_f", main) 
        #p <- p + ggtitle(main)
        p

    }
    else if (type == "alpha") {
        ## -------------------------- 2. plot multi-level coefficient 
        ## Zname <- c()
        k <- length(x$est.alpha)
        p.all <- NULL
        start.pos <- ifelse(rm1, 2, 1)
        #if (!is.null(labelname) && intercept == TRUE) {
        #    labelname <- c("Intercept", labelname)
        #}
        k.length <- k - start.pos + 1

        if (!is.null(xlim)) {
        	if (class(xlim) == "matrix") {
        		if (dim(xlim)[1] == k.length) {
        			xlim <- xlim
        		} else {
        			xlim <- NULL
        		}
        	} else if (class(xlim) == "numeric") {
        		## xlim <- -abs(xlim)
        		if (length(xlim) == 2) {
        			## xlim <- matrix(rep(c(xlim, -xlim), each = k.length), k.length, 2)
        			xlim <- matrix(rep(xlim, each = k.length), k.length, 2)
        		} else {
        			## if (length(xlim) == k.length) {
        			##	xlim <- matrix(c(xlim, -xlim), k.length, 2)
        			##} else {
        				xlim <- NULL
        			##}

        		}
        	} else {
        		xlim <- NULL
        	}
        }

        
        jj <- 1
        for (i in start.pos:k) {
            data <- x$est.alpha[[i]]
            ncount <- dim(data)[1]
            data <- rbind(data, c(NA, NA, NA, 0.2, 0))
            data <- rbind(data, c(NA, NA, NA, ncount+0.8, 0))

            p <- ggplot(data, aes(mean, id, colour = tr)) 
            p <- p + geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50")
            p <- p + geom_point() + geom_errorbarh(aes(xmax = ci_u, xmin = ci_l, height = .2))
            ## p <- p + xlab("Coef") + ylab("Units")
            p <- p + xlab(xlab) + ylab(ylab)
            p <- p + theme_bw(base_size = 15)

            set.limits <- c(1, 0)
            set.labels <- c("Treated", "Control")
            set.colors <- c("black", "grey80")

            p <- p + scale_colour_manual(limits = set.limits,
                                         labels = set.labels,
                                         values =set.colors) +
                     guides(colour = guide_legend(title=NULL, nrow=1))
            p <- p + scale_y_continuous(expand = c(0, 0), breaks = data$id, labels = NULL, limits = c(min(data$id) - 0.3, max(data$id) + 0.3))
            p <- p + xlab(xlab) + ylab(ylab)
            main <- ifelse(is.null(labelname), "", labelname[i]) 
            p <- p + ggtitle(main)
            p <- p + theme(panel.grid.major = element_blank(),
                       panel.grid.minor = element_blank(),
                       legend.text = element_text(margin = margin(r = 10, unit = "pt"), size = cex.legend),
                       legend.position = legend.pos,
                       axis.title = element_text(size=12),
                       axis.title.x = element_text(margin = margin(t = 8, r = 0, b = 0, l = 0), size = cex.xlab),
                       axis.title.y = element_text(margin = margin(t = 0, r = 8, b = 0, l = 0), size = cex.ylab),
                       axis.text = element_text(color="black", size=8),
                       axis.text.x = element_text(size = cex.xaxis),
                       axis.text.y = element_text(size = cex.yaxis),
                       axis.ticks.y = element_blank(),
                       plot.title = element_text(size = cex.main,
                                                 hjust = 0.5,
                                                 face="bold",
                                                 margin = margin(10, 0, 10, 0)))
            
            
            #p <- p + ggtitle(main)
            if (!is.null(xlim)) {
                p <- p + coord_cartesian(xlim = xlim[jj, ])
                jj <- jj + 1
            }
            p.all <- c(p.all, list(p))
        }
        q <- grid.arrange(grobs = p.all, ncol = plotlength)
        q
    }
    else if (type == "xi") {
        ## ------------------------- 3. plot time-varying coefficient
        ## Aname <- c()
        k <- length(x$est.xi)
        p.all <- NULL
        start.pos <- ifelse(rm1, 2, 1)
        #if (!is.null(labelname)) {
        #    labelname <- c("intercept", labelname)
        #}
        
        for (i in start.pos:k) {
            data <- x$est.xi[[i]]
            show <- NULL
            if (!is.null(xlim)) {
                show <- which(data$time >= xlim[1] & data$time <= xlim[2])
            }

            data$type <- rep("m", dim(data)[1])

            p <- ggplot(data) 
            p <- p + xlab(xlab) + ylab(ylab) 
            p <- p + geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50")
            p <- p + geom_line(aes(time, mean, colour = type, size = type))
            p <- p + geom_ribbon(aes(x = time, ymin = ci_l, ymax = ci_u),alpha=0.2)

            set.limits <- c("m", "c")
            set.labels <- c("Mean", "95% CI")
            set.colors <- c("black", "#00000080")
            set.size <- c(1,3)

            p <- p + scale_colour_manual(limits = set.limits,
                                         labels = set.labels,
                                         values =set.colors) +
                     scale_size_manual(limits = set.limits,
                                       labels = set.labels,
                                       values = set.size) +
                     guides(colour = guide_legend(title=NULL, nrow=1),
                            size = guide_legend(title=NULL, nrow=1))

            p <- p + theme_bw(base_size = 15)
            main <- ifelse(is.null(labelname), "", labelname[i]) 
            p <- p + ggtitle(main)

            p <- p + theme(panel.grid.major = element_blank(),
                       panel.grid.minor = element_blank(),
                       legend.text = element_text(margin = margin(r = 10, unit = "pt"), size = cex.legend),
                       legend.position = legend.pos,
                       axis.title = element_text(size=12),
                       axis.title.x = element_text(margin = margin(t = 8, r = 0, b = 0, l = 0), size = cex.xlab),
                       axis.title.y = element_text(margin = margin(t = 0, r = 8, b = 0, l = 0), size = cex.ylab),
                       axis.text = element_text(color="black", size=8),
                       axis.text.x = element_text(size = cex.xaxis),
                       axis.text.y = element_text(size = cex.yaxis, angle = 90),
                       plot.title = element_text(size = cex.main,
                                                 hjust = 0.5,
                                                 face="bold",
                                                 margin = margin(10, 0, 10, 0)))

            
            if (!is.null(ylim)) {
                if (is.list(ylim)==TRUE) {
                    p <- p + coord_cartesian(ylim = ylim[[i]])
                } else {
                    p <- p + coord_cartesian(ylim = ylim)
                }
                
            }
            p.all <- c(p.all, list(p))
        }
        q <- grid.arrange(grobs = p.all, nrow = plotlength)

    }

}



## ------------------------2. plot treatment effect and outcomes

effPlot <- function(x,                      ## summary data
                    type,                   ## plot what: outcome, eff,
                    plotlength = 5,         ## for multi-level and time-varying, by row or column
                    main = NULL,            ## title
                    legend = TRUE,          ## show legend 
                    xlim = NULL,
                    ylim = NULL,
                    xlab = NULL,
                    ylab = NULL,
                    cex.xlab = 13,
                    cex.ylab = 13,
                    cex.xaxis = 10,
                    cex.yaxis = 10,
                    cex.main = 15, 
                    cex.legend = 10,  
                    x1.pos = 0,
                    x2.pos = NULL,             ## for placebo
                    y.pos = 0,
                    legend.pos = NULL,
                    legend.labs = NULL,
                    CI = TRUE) {

    if (is.null(legend.pos)==TRUE) {
        legend.pos <- ifelse(legend, "bottom", "none")
    }
    xlab <- ifelse(is.null(main), "Time", xlab)
    ylab <- ifelse(is.null(main), "Coef", ylab)
    ## ----------------------- 4. observed and counterfactual outcomes
    if (type == "outcome") {
        main <- ifelse(is.null(main), "Observed and Estimated Counterfactual Outcomes", main) 

        data <- x$est.eff
        data$type <- rep("o", dim(data)[1])

        show <- NULL
        if (!is.null(xlim)) {
            show <- which(data$time >= xlim[1] & data$time <= xlim[2])
            data <- data[show, ]
        }

        p <- ggplot(data) 
        p <- p + xlab(xlab) + ylab(ylab) 
        
        if(!is.null(x1.pos)) {
            p <- p + geom_vline(xintercept = x1.pos, linetype = "dashed", colour = "grey50")
        }
        if(!is.null(x2.pos)) {
            p <- p + geom_vline(xintercept = x2.pos, linetype = "dashed", colour = "grey50")
        }
        if(!is.null(y.pos)) {
            p <- p + geom_hline(yintercept = y.pos, linetype = "dashed", colour = "grey50")
        }
        if (CI == TRUE) {
            p <- p + geom_ribbon(aes(x = time, ymin = counterfactual_ci_l, 
             ymax = counterfactual_ci_u), fill = "#77777730")
        }
        p <- p + geom_line(aes(time, estimated_counterfactual), size = 0.8, 
            colour = "#777777", linetype = "dashed")
        p <- p + geom_line(aes(time, observed, colour = type, size = type))
        

        set.labels <- legend.labs
        if (CI == TRUE) {
            if (is.null(set.labels)==TRUE) {
                set.labels <- c("Observed", "Estimated Y(0)", "95% CI")
            } 
            set.limits <- c("o", "m", "c")
            set.colors <- c("black", "#777777", "#77777730")
            set.size <- c(1, 1, 3)
        } else {
            if (is.null(set.labels)==TRUE) {
                set.labels <- c("Observed", "Estimated Y(0)")
            }
            set.limits <- c("o", "m")
            set.colors <- c("black", "#777777")
            set.size <- c(1, 1)
        }    
        

        p <- p + scale_colour_manual(limits = set.limits,
                                     labels = set.labels,
                                     values =set.colors) +
                 scale_size_manual(limits = set.limits,
                                   labels = set.labels,
                                   values = set.size) +
                 guides(colour = guide_legend(title=NULL, nrow=3),
                        size = guide_legend(title=NULL, nrow=3))

        p <- p + ggtitle(main)

        p <- p + theme_bw(base_size = 15)

        p <- p + theme(panel.grid.major = element_blank(),
                       panel.grid.minor = element_blank(),
                       legend.text = element_text(margin = margin(r = 8, unit = "pt"), size = cex.legend),
                       legend.position = legend.pos,
                       axis.title = element_text(size=12),
                       axis.title.x = element_text(margin = margin(t = 8, r = 0, b = 0, l = 0), size = cex.xlab),
                       axis.title.y = element_text(margin = margin(t = 0, r = 8, b = 0, l = 0), size = cex.ylab),
                       axis.text = element_text(color="black", size=8),
                       axis.text.x = element_text(size = cex.xaxis),
                       axis.text.y = element_text(size = cex.yaxis, angle = 90),
                       plot.title = element_text(size = cex.main,
                                                 hjust = 0.5,
                                                 face="bold",
                                                 margin = margin(10, 0, 10, 0)))

        if (!is.null(ylim)) {
            p <- p + coord_cartesian(ylim = ylim)
        }
        p

    }
    else if (type == "eff") {
        ## ------------------------------- 5. att
        main <- ifelse(is.null(main), "Estimated Treatment Effects", main)

        data <- x$est.eff

        show <- NULL
        if (!is.null(xlim)) {
            show <- which(data$time >= xlim[1] & data$time <= xlim[2])
            data <- data[show, ]
        }

        p <- ggplot(data) + xlab(xlab) + ylab(ylab) 
        if(!is.null(x1.pos)) {
            p <- p + geom_vline(xintercept = x1.pos, linetype = "dashed", colour = "grey50")
        }
        if (!is.null(x2.pos)) {
            p <- p + geom_vline(xintercept = x2.pos, linetype = "dashed", colour = "grey50")
        }
        if(!is.null(y.pos)) {
            p <- p + geom_hline(yintercept = y.pos, linetype = "dashed", colour = "grey50")
        }
        p <- p + geom_line(aes(time, estimated_ATT), size = 1)
        p <- p + geom_ribbon(aes(x = time, ymin = estimated_ATT_ci_l, 
                                           ymax = estimated_ATT_ci_u), alpha = 0.2)

        p <- p + ggtitle(main)
        p <- p + theme_bw(base_size = 15)
        p <- p + theme(panel.grid.major = element_blank(),
                       panel.grid.minor = element_blank(),
                       legend.text = element_text(margin = margin(r = 10, unit = "pt"), size = cex.legend),
                       legend.position = legend.pos,
                       axis.title = element_text(size=12),
                       axis.title.x = element_text(margin = margin(t = 8, r = 0, b = 0, l = 0), size = cex.xlab),
                       axis.title.y = element_text(margin = margin(t = 0, r = 8, b = 0, l = 0), size = cex.ylab),
                       axis.text = element_text(color="black", size=8),
                       axis.text.x = element_text(size = cex.xaxis),
                       axis.text.y = element_text(size = cex.yaxis, angle = 90),
                       plot.title = element_text(size = cex.main,
                                                 hjust = 0.5,
                                                 face="bold",
                                                 margin = margin(10, 0, 10, 0)))
        if (!is.null(ylim)) {
            p <- p + coord_cartesian(ylim = ylim)
        }
        p
    }
    else if (type == "cumu") {
        ## --------------------------------- 6. cumulative effects
        data <- x$est.cumu 
        data <- rbind(0, data)
        data$time <- 1:dim(data)[1] - 1

        show <- NULL
        if (!is.null(xlim)) {
            show <- which(data$time >= xlim[1] & data$time <= xlim[2])
            data <- data[show, ]

        }

        main <- ifelse(is.null(main), "Cumulative Effects", main)

        p <- ggplot(data) + xlab(xlab) + ylab(ylab) 
        if(!is.null(x1.pos)) {
            p <- p + geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50")
        }
        if(!is.null(y.pos)) {
            p <- p + geom_hline(yintercept = 0, linetype = "dashed", colour = "grey50")
        }
        p <- p + geom_line(aes(time, mean), size = 1)
        p <- p + geom_ribbon(aes(x = time, ymin = ci_l, 
                                           ymax = ci_u), alpha = 0.2)

        p <- p + ggtitle(main)
        p <- p + theme_bw(base_size = 15)
        p <- p + theme(panel.grid.major = element_blank(),
                       panel.grid.minor = element_blank(),
                       legend.text = element_text(margin = margin(r = 10, unit = "pt"), size = cex.legend),
                       legend.position = legend.pos,
                       axis.title = element_text(size=12),
                       axis.title.x = element_text(margin = margin(t = 8, r = 0, b = 0, l = 0), size = cex.xlab),
                       axis.title.y = element_text(margin = margin(t = 0, r = 8, b = 0, l = 0), size = cex.ylab),
                       axis.text = element_text(color="black", size=8),
                       axis.text.x = element_text(size = cex.xaxis),
                       axis.text.y = element_text(size = cex.yaxis),
                       plot.title = element_text(size = cex.main,
                                                 hjust = 0.5,
                                                 face="bold",
                                                 margin = margin(10, 0, 10, 0)))
        if (!is.null(ylim)) {
            p <- p + coord_cartesian(ylim = ylim)
        }
        p
    }
}


## ------------- 2.1 another att plot: with true eff: only for simulated data
## with true effect
plotTeff <- function(x, teff, 
                     x.pos = NULL, y.pos = NULL, 
                     xlim = NULL, ylim = NULL, main = NULL,
                     xlab = NULL, ylab = NULL,
                     legend = TRUE,
                     cex.xlab = 12,
                     cex.ylab = 12,
                     cex.xaxis = 8,
                     cex.yaxis = 8,
                     cex.main = 15, 
                     cex.legend = 12) {

    main <- ifelse(is.null(main), "True and Estimated Treatment Effects", main)
    xlab <- ifelse(is.null(main), "Time", xlab)
    ylab <- ifelse(is.null(main), "Coef", ylab)
    legend.pos <- ifelse(legend == TRUE, "bottom", "none")
    data <- x$est.eff
    data$type <- rep("o", dim(data)[1])

    ## teff <- rep(0, dim(data)[1]) ## from data
    data$teff <- teff

    show <- NULL
    if (!is.null(xlim)) {
        show <- which(data$time >= xlim[1] & data$time <= xlim[2])
        data <- data[show, ]

    }

    p <- ggplot(data) + xlab(xlab) + ylab(ylab) 
    if(!is.null(x.pos)) {
        p <- p + geom_vline(xintercept = x.pos, linetype = "dashed", colour = "grey50")
    }
    if(!is.null(y.pos)) {
        p <- p + geom_hline(yintercept = y.pos, linetype = "dashed", colour = "grey50")
    }
    p <- p + geom_line(aes(time, teff, colour = type, size = type))
    p <- p + geom_line(aes(time, estimated_ATT), size = 1, colour = "#0000FF")
    p <- p + geom_ribbon(aes(x = time, ymin = estimated_ATT_ci_l, 
                                       ymax = estimated_ATT_ci_u), 
                                       fill = "#0000FF", alpha = 0.2)

    set.limits <- c("o", "m", "c")
    set.labels <- c("True", "Estimated", "95% CI")
    set.colors <- c("black", "#0000FF", "#0000FF80")
    set.size <- c(1, 1, 3)

    p <- p + scale_colour_manual(limits = set.limits,
                                 labels = set.labels,
                                 values =set.colors) +
             scale_size_manual(limits = set.limits,
                               labels = set.labels,
                               values = set.size) +
             guides(colour = guide_legend(title=NULL, nrow=1),
                    size = guide_legend(title=NULL, nrow=1))

    p <- p + ggtitle(main)

    p <- p + theme_bw(base_size = 15)

    p <- p + theme(panel.grid.major = element_blank(),
                       panel.grid.minor = element_blank(),
                       legend.text = element_text(margin = margin(r = 10, unit = "pt"), size = cex.legend),
                       legend.position = legend.pos,
                       axis.title = element_text(size=12),
                       axis.title.x = element_text(margin = margin(t = 8, r = 0, b = 0, l = 0), size = cex.xlab),
                       axis.title.y = element_text(margin = margin(t = 0, r = 8, b = 0, l = 0), size = cex.ylab),
                       axis.text = element_text(color="black", size=8),
                       axis.text.x = element_text(size = cex.xaxis),
                       axis.text.y = element_text(size = cex.yaxis),
                       plot.title = element_text(size = cex.main,
                                                 hjust = 0.5,
                                                 face="bold",
                                                 margin = margin(10, 0, 10, 0)))
    if (!is.null(ylim)) {
        p <- p + coord_cartesian(ylim = ylim)
    }
    p

}


## -------------------------- 3. density plot for omega 
omegaPlot <- function(x,                      ## summary data
                      oxlim = NULL,
                      type,                   ## plot what: alpha, xi, f
                      burn = 0,
                      plotlength = 5,         ## for multi-level and time-varying, by row or column
                      labelname = NULL,       ## annotation
                      legend = TRUE,
                      xlab = NULL,
                      ylab = NULL,
                      cex.xlab = 12,
                      cex.ylab = 12,
                      cex.xaxis = 8,
                      cex.yaxis = 8,
                      cex.main = 15, 
                      cex.legend = 12,  
                      intercept = FALSE) {    ## show intercept

    legend.pos <- ifelse(legend == TRUE, "bottom", "none")
    xlab <- ifelse(is.null(xlab), "Omega", xlab)
    ylab <- ifelse(is.null(ylab), "Density", ylab)
    niter <- length(x$sigma2_i)
    if (niter <= (burn + 1)) {
      stop("Burn too much...\n")
    }
    if (type == "alpha") {
        data <- as.matrix(x$wa[,(burn+1):niter])
    } else if (type == "xi") {
        data <- as.matrix(x$wxi[,(burn+1):niter])
    } else {
        data <- as.matrix(x$wg[,(burn+1):niter])
    }

    if (1 %in% dim(data)) {
        data <- t(data)
    }

    if (type %in% c("alpha", "xi")) {
        if (intercept) {
            if (!is.null(labelname)) {
                labelname <- c("Intercept", labelname)
            }
        } else {
            data <- as.matrix(data[-1,])
        }
    }

    if (0 %in% dim(data)) {
      stop("Cannot plot.\n")
    } else {
      if (1  %in% dim(data)) {
        data <- t(data)
      }
    }

    k <- dim(data)[1]
    p.all <- NULL

    for (i in 1:k) {
        subdata <- cbind.data.frame(data[i,])
        names(subdata) <- "omega"

        p <- ggplot(subdata,aes(omega))
        p <- p + geom_vline(xintercept = 0, linetype = "dashed", colour = "grey50")

        p <- p + geom_line(stat = "density") + expand_limits(y = 0)
        if (!is.null(oxlim)) {
            p <- p + xlim(oxlim[1], oxlim[2]) 
        }

        p <- p + xlab(xlab) + ylab(ylab)
        p <- p + theme_bw(base_size = 15)
        p <- p + theme(panel.grid.major = element_blank(),
                       panel.grid.minor = element_blank(),
                       legend.text = element_text(margin = margin(r = 10, unit = "pt"), size = cex.legend),
                       legend.position = "bottom",
                       axis.title = element_text(size=12),
                       axis.title.x = element_text(margin = margin(t = 8, r = 0, b = 0, l = 0), size = cex.xlab),
                       axis.title.y = element_text(margin = margin(t = 0, r = 8, b = 0, l = 0), size = cex.ylab),
                       axis.text = element_text(color="black", size=8),
                       axis.text.x = element_text(size = cex.xaxis),
                       axis.text.y = element_text(size = cex.yaxis),
                       plot.title = element_text(size = cex.main,
                                                 hjust = 0.5,
                                                 face="bold",
                                                 margin = margin(10, 0, 10, 0)))
        
        
        main <- ifelse(is.null(labelname), "", labelname[i]) 
        p <- p + ggtitle(main)
        p.all <- c(p.all, list(p))

    }

    q <- grid.arrange(grobs = p.all, nrow = plotlength)
}

## ggsave('/Users/llc/Desktop/bayes_plot/mg.pdf', q, width = 18, height = 15)

## ggsave('/Users/llc/Desktop/bayes_plot/mg.pdf', q, width = 18, height = 15)
```


## Example: ADH (2015)

Actual example script:

```r
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
dataprep.out <- dataprep(foo = data,
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
synth.out <- synth(data.prep.obj=dataprep.out, Margin.ipop=.005,Sigf.ipop=7,Bound.ipop=6)

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
Y.co.mat <- matrix(data.co$gdp, (N-1), TT, byrow = TRUE)
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
  xlab = "Time", ylab = "", by.timing = TRUE, main = "", axis.lab.gap = c(1,0))
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
                  legend.pos = c(0.2,0.85), 
                  legend.labs = c("West Germany","Estimated Y(0)", "95% CI"))
ggsave(p1.ct, file = "./graphs/adh_ct.pdf", height = 5, width = 6)



## plot results
p1.beta <- coefPlot(sout1, type = "beta", plotlength = 3, labelname = c("Intercept", Xname),             
                    legend = TRUE, xlim = NULL, ylim = NULL, rm1 = 0)
#ggsave(p1.beta, file = "./graphs/adh_beta.pdf", height = 7, width = 4)


ylim.v <- c(c(-20000,20000),rep(c(-100,100),6))
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
    plot(density(wg[i,]), main = paste0("Factor ",k), xlim = c(-7000,7000))
  } else {
    plot(density(wg[i,]), main = paste0("Factor ",k), xlim = c(-3000,3000))
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
```