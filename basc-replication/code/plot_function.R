## plot functions

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


