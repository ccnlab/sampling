# -------------------------------------------------------------------------
# Locally Learned Synaptic Failure for Complete Bayesian Inference
# Author: Kevin McKee (klmckee@ucdavis.edu)

# Description: An abstracted model of synaptic failure in biological neural nets
# The model uses a local learning rule with weights bound to the [0,1] interval.
# Weights map to dropout probability. 
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
probFromWeight<-function(x){
  n<-ncol(x)
  p<-t(apply(x, 1, function(z){
    zs<-sort(z/sum(z), decreasing=T)
    fw<-zs/(1-cumsum(c(0,zs[-n])))
    fw_unsort<-fw[order(order(z, decreasing=T))]
    return(fw_unsort)
  } ))
  p[is.infinite(p)]<-1
  p[is.nan(p)]<-1
  p[p>1]<-1
  return(p)
}

# Generate data -----------------------------------------------------------
set.seed(1111)
n<-4000
np<- 50 #Number of bins
ns<- 250#Number of samples per bin
j<-100 #Number of neurons
k<-5000 #Grid resolution for inversion
sig<-0.05 #RBF Sigma
ppt<-seq(-6,6,l=np)

#Heteroskedastic discrete data model
nBin<-5
x<- rep(seq(-4, 4, l=nBin), each=n/nBin)
beta<-0
y<-rnorm(length(x),  0, 0.2+0.2*(x+4))

# Kernel conversion -------------------------------------------------------
pp<-seq(-6,6,l=j)
pp2<-seq(-6,6,l=k)
H<-RBF(pp2, matrix(pp, k, j, byrow=T), sig)
X<-RBF(x, matrix(pp, n, j, byrow=T), sig)
Y<-RBF(y, matrix(pp, n, j, byrow=T), sig)

#Train by simply accumulating matches vs mismatches:
A0<-matrix( 1e-3*runif(j*j) + .025, j,j)  #Prior positives
B0<-matrix( 1e-3*runif(j*j) + .1, j,j)  #Prior negatives
A<-A0 + t(X)%*%Y   
B<-B0 + t(X)%*%(1-Y)  
W<-A/(A+B) #No error

LR<-1 #For beta distribution
K<-mean(rowSums(X))

#Aleatoric uncertainty derived analytically
Q0<-probFromWeight(W)/K
image(Q0, col=gray.colors(1000))

#Epistemic uncertainty derived from beta distribution variance
Psi <- (sqrt((-A^2-A*B-A)^2 + 4*B*(A^2+A*B+A)) - A^2-A*B-A )/(2*B)

# Local learning of probabilities -----------------------------------------
qL<-matrix(runif(j*j, 0.7, 0.71), j,j, byrow=T)#Prior positives
Xh<-RBF(ppt, matrix(pp, np, j, byrow=T), sig)
Xthr<-mean(X[1,])
nEpoch<-5000
LRq<-0.01
for(epoch in 1:nEpoch){
  for(h in 1:nrow(Xh) ){
    selInp<-which(Xh[h,]>Xthr)  #Try to learn several 'x' at once
    Wx<-matrix(W[selInp,], nrow=length(selInp))
    nX<-nrow(Wx)#sum(Xh[h,]>Xthr )
    qLx<-qL[selInp,]
    p<-matrix(rbinom(nX*j,1, qLx ), nX, j )
    sub<-p*Wx
    while(sum(sub>0) > 0){
      m<-which(Wx==max(sub, na.rm=T))
      subE<-sub
      g<-subE[m]/sum(subE) #Percentage of agent consumed...
      qLx[m]<- qLx[m] - LRq*(qLx[m] - g^8 ) #BEST
      sub[m]<-0
      qLx[qLx>1]<-1
      qLx[qLx<0]<- 1e-5
    }
    qL[Xh[h,]>Xthr,]<-qLx#matrix(qLx, nrow=sum(Xh[h,]>Xthr), ncol=j, byrow=T)
    # qL[selInp,]<-qLx #One 'x' at a time
  }
  if(epoch%%10==0){
    cat("\r",epoch,LRq,"\t\t")
    par(mfrow=c(1,3))
    image(Q0, col=gray.colors(1000))
    image(qL, col=gray.colors(1000))
    plot(Q0[25,], qL[25,], log="xy" );abline(0,1)
    cat("\t\t", cor(Q0[25,], qL[25,]),"\t\t")
  }
}

#Can use either analytic (Q0) or learned (QL) probabilities
Q<-Q0
# Q<-qL

par(mfrow=c(1,3), mai=c(.1,.1,.1,.1))
image(W, col=gray.colors(1000), xaxt="n", yaxt="n")
image(Q0, col=gray.colors(1000), xaxt="n", yaxt="n")
image(qL, col=gray.colors(1000), xaxt="n", yaxt="n")

# Run simulation ----------------------------------------------------------
Xh<-RBF(ppt, matrix(pp, np, j, byrow=T), sig)
cl<-makeCluster(5)
registerDoSNOW(cl)
pts<-foreach(h = 1:np, .inorder=F, .combine=rbind, .init=NULL)%dopar%{
  samples.total.dropout<-samples.total.beta<-samples.resid<-samples.mean.dropout<-samples.mean.beta<-matrix(NA, ns, j)
  for(i in 1:ns){
    
    #Total distribution: Dropout
    M<-matrix(rbinom(j*j, 1, Psi*Q ),j,j) 
    s<- (Xh[h,]%*%(W*M))
    samples.total.dropout[i,]<-s
    
    #Total distribution: Beta
    Ws<-matrix(rbeta(j*j, 1+A*LR, 1+B*LR), j, j)
    M<-matrix(rbinom(j*j, 1, Q ),j,j) 
    s<- Xh[h,]%*%(Ws*M)
    samples.total.beta[i,]<-s
    
    #Residuals:
    M<-matrix(rbinom(j*j, 1, Q ),j,j) 
    s<- Xh[h,]%*%(W*M)
    samples.resid[i,]<-s
    
    #MAPs: beta
    Ws<-matrix(rbeta(j*j, 1+A*LR, 1+B*LR), j, j)
    s<- Xh[h,]%*%Ws
    samples.mean.beta[i,]<-s
    
    #MAPs: dropout
    M<-matrix(rbinom(j*j, 1, Psi),j,j) 
    s<- Xh[h,]%*%(W*M)
    samples.mean.dropout[i,]<-s
    
  }
  #Invert samples to produce continuous value
  samples.total.beta<-unlist(lapply(invRBF(samples.total.beta, H, pp2), '[[', 1)) 
  samples.total.dropout<-unlist(lapply(invRBF(samples.total.dropout, H, pp2), '[[', 1))
  samples.resid<-unlist(lapply(invRBF(samples.resid, H, pp2), '[[', 1)) 
  samples.mean.beta<-unlist(lapply(invRBF(samples.mean.beta, H, pp2), '[[', 1)) 
  samples.mean.dropout<-unlist(lapply(invRBF(samples.mean.dropout, H, pp2), '[[', 1)) 
  
  return(cbind(ppt[h], samples.total.beta,samples.total.dropout, samples.resid, samples.mean.beta, samples.mean.dropout))
}
stopCluster(cl)
dim(pts)

# Plots -------------------------------------------------------------------
if(nrow(pts)<=np*ns){
  
  sd.total.beta<-sd.total.dropout<-sd.resid<-sd.mean.dropout<-sd.mean.beta<-c()
  for(pt in ppt){
    sd.total.beta<-c(sd.total.beta,           sqrt(var(pts[pts[,1]==pt,2]) ))
    sd.total.dropout<-c(sd.total.dropout,     sqrt(var(pts[pts[,1]==pt,3]) ))
    sd.resid<-c(sd.resid,                     sqrt(var(pts[pts[,1]==pt,4]) ))
    sd.mean.beta<- c(sd.mean.beta,            sqrt(var(pts[pts[,1]==pt,5]) ))
    sd.mean.dropout<- c(sd.mean.dropout,      sqrt(var(pts[pts[,1]==pt,6]) ))
  }
  
  trueSD<-.2+.2*(ppt+4) 
  
  par(mfrow=c(1,2), mai=c(1,1,.1,.1), cex.lab=1)
  plot(trueSD, sd.resid, xlim=c(0,max(c(sd.resid,trueSD)) ), ylim=c(0,max(c(sd.resid,trueSD)) ), type="o", xlab="True Standard Deviation", ylab="Estimated Standard Deviation", col="blue");abline(0,1)
  plot(sd.mean.beta, sd.mean.dropout, xlim=c(min(c(sd.mean.dropout,sd.mean.beta)) ,max(c(sd.mean.dropout,sd.mean.beta)) ), ylim=c(min(c(sd.mean.dropout,sd.mean.beta)) ,max(c(sd.mean.dropout,sd.mean.beta)) ), type="p", xlab="Beta Standard Deviation", ylab="Estimated Standard Deviation", col="blue", pch=16, log="xy");abline(0,1)
  
  #Plot result
  gscale<-6
  jit<-.1
  ptSize<-1
  ptsr<-pts[,1]+runif(nrow(pts), -1,1)*jit
  xr<-x+runif(n, -1,1)*jit

  pdf("simresults1.pdf",width=12,height=6)
  par(mfrow=c(1,2), mai=c(1,1,.1,.1), cex.lab=1.2)
  plot(xr,y, col="black", pch=".", xlim=c(-gscale, gscale), ylim=c(-gscale, gscale), cex=ptSize, xlab="X", ylab="Y")
  plot(ptsr,  pts[,3], col="blue", pch=".", xlim=c(-gscale,gscale), ylim=c(-gscale, gscale), cex=ptSize, xlab="X", ylab="Y predicted (Dropout-only model)")
  points(ptsr, pts[,6], col="red", pch=".", cex=ptSize)
  legend("topleft", pch=c(16, 16), c("Total distribution", "Epistemic distribution"), col=c("blue", "red"), bg="white", bty="o")
  dev.off()
}


