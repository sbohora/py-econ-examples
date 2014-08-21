# OLS and IV regressions done manually in R

rm(list=ls())

#load data
data <- read.csv("card.csv", header=TRUE)

#extract variables
wage <- data$wage
constant <- matrix(1,nrow=length(wage),ncol=1)
lnwage <- log(wage)
educ <- data$educ
exper <- data$exper
exper2 <- exper^2
south <- data$south
black <- data$black

# a) OLS estimation

X <- cbind(constant,educ,exper,exper2,south,black)
BetaOLS <- solve(t(X)%*%X)%*%t(X)%*%lnwage

Yhat <- X%*%BetaOLS
n <- length(Yhat)
sigma2hat <- (1/n)*colSums((lnwage - Yhat)^2)
Q <- (1/n)*t(X)%*%X
Qinv <- solve(Q)
SE <- (1/sqrt(n))*sqrt(sigma2hat)*sqrt(diag(Qinv))

# b) 2SLS estimation

#instrument near4
near4 <- data$nearc4
Z <- cbind(constant,near4)
gammahat <- solve(t(Z)%*%Z)%*%t(Z)%*%educ
educhat <- Z %*% gammahat

Xhat <- cbind(constant,educhat,exper,exper2,south,black)
Beta2SLS <- solve(t(Xhat)%*%Xhat)%*%t(Xhat)%*%lnwage

YhatIV <- Xhat %*% Beta2SLS
sigma2hatIV <- (1/n)*colSums((lnwage-YhatIV)^2)
QIV <- (1/n)*t(Xhat)%*%Xhat
QIVinv <- solve(QIV)
SEIV <- (1/sqrt(n))*sqrt(sigma2hatIV)*sqrt(diag(QIVinv))
