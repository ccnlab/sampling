
# -------------------------------------------------------------------------
# RBF kernel regression with sampling by weight dropout
# Author: Kevin McKee (klmckee@ucdavis.edu)

# Description: An abstracted model of synaptic failure in biological neural nets
# The model uses a local learning rule with weights bound to the [0,1] interval.
# Weights map to dropout probability. 
# An exponent of 1.45 was chosen for mapping the weight matrix to the failure probability matrix,
# but the resulting accuracy of distributions is still not perfect.
# -------------------------------------------------------------------------



library(doSNOW)
library(foreach)

#Functions ---------------------------------------------------------------
RBF<-function(x,mu,sig) exp(-(x-mu)^2/sig)
invRBF<-function(y, H, pp) {
  if(!is.null(dim(y))){
    apply(y, 1, function(x) pp[H%*%x==max(H%*%x)])
  }else{
    pp2[H%*%y==max(H%*%y)]
  }
}

# Generate data -----------------------------------------------------------
set.seed(123)

#Simple linear model: 
# n<-1000
# x<-runif(n, -5, 5)
# y<-x + 0.1*rnorm(n)

#Data model 2: Expanding variance
set.seed(123)
nTrain<-300
nTest<-100
n<-nTrain+nTest
x<-runif(n, -5, 5)
beta<-.4
y<-rnorm(n,  beta*x-2, 0.1+0.2*(x+5))
plot(x,y)

#Bimodal
# nTrain<-300
# nTest<-100
# n<-nTrain+nTest
# x<-runif(n, -5, 5)
# beta<-.25
# y1<-rnorm(n,  -2, 0.2)
# y2<-rnorm(n,  beta*x, 0.2)
# pvec <- plogis(0.25*x)
# cvec<-rbinom(n, 1, pvec)+1
# y<-x
# y[cvec==1]<-y1[cvec==1]
# y[cvec==2]<-y2[cvec==2]
# plot(x,y)

# Kernel conversion -------------------------------------------------------
j<-50
k<-20000
sig<-.2
pp<-seq(-6,6,l=j)
pp2<-seq(-6,6,l=k)
H<-RBF(pp2, matrix(pp, k, j, byrow=T), sig)
X<-RBF(x, matrix(pp, n, j, byrow=T), sig)
Y<-RBF(y, matrix(pp, n, j, byrow=T), sig)


# RBF regression by local learning with weight failure -----------------------------------------------------------------
W<-matrix(rnorm(j*j, 0.0, sd=0.00),j,j)

epochs<-50
LR<-10
err<-c()
for(h in 1:epochs){
  errSum<-0
  for(i in 1:nrow(X)){
    # cat("\r",h,i,"\t\t")
    Yp<-X[i,]%*%W
    e<-Y[i,]-Yp  
    W<-W+LR*X[i,]%*%e/n
    W[W<0]<-0
    W[W>1]<-1
    errSum<- errSum + sum(e^2)/n
  }
  err<-c(err, errSum)
  plot(err, type="o")
}


# Predict for values across x and sample by synaptic failure----------------------------------------------------------------
np<-300
ppt<-seq(-5,5,l=np)
Xh<-RBF(ppt, matrix(pp, np, j, byrow=T), sig)

#Parallel sampling for speed. Change %dopar% to %doseq% for sequential.
cl<-makeCluster(4)
registerDoSNOW(cl)
pts<-foreach(h = 1:np, .inorder=F, .combine=rbind, .init=NULL)%dopar%{
  samples<-c()
  for(i in 1:10){
    P<-matrix(rbinom(j*j, 1, W^1.45 ),j,j) #Failure matrix
    s<- Xh[h,]%*%(W*P)
    samples<-rbind(samples,  s)
  }
  samples<-samples[rowSums(samples)>0,]
  samples<-unlist(lapply(invRBF(samples, H, pp2), '[[', 1)) #Invert sample to produce continuous value
  return(cbind(ppt[h], samples))
}
stopCluster(cl)

#Plot result
plot(x,y, col="red", pch=16)
points(pts, col="blue", pch=16)






