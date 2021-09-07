setwd("C:/UC Davis/Code/Axon/axon/examples/sae2")


x<-read.csv("spikes_7_3.csv")



x<-x[,51:60]

par(mfrow=c(3,1), mai=c(.65, 1,.1,.1), cex.lab=1.5)
plot(svd(x)$d, type="o", ylab="Eigenvalues")


chunksize<-20
y<-c()
for(i in 1:(nrow(x)/chunksize)) y<-rbind(y,colSums(x[(i-1)*chunksize+(1:chunksize ),]))

plot(svd(y)$d, type="o", ylab="Eigenvalues")


plot(1:nrow(y), y[,1], type="n", col="red", ylim=c(0, max(y)), ylab="Spikes per 100 cycles", xlab="Cycle (x 100)")
for(i in 1:nrow(y)) lines(1:nrow(y) + i/60 , y[,i], type="l", col=rainbow(10)[i])



min(cor(y), na.rm=T)

# 
# chunksize<-100
# y<-c()
# for(i in 1:(nrow(x)/chunksize)) y<-rbind(y,colSums(x[(i-1)*chunksize+(1:chunksize ),]))
# plot(1:nrow(y), y[,1], type="n", col="red", ylim=c(0, max(y)))
# for(i in c(30,19)) lines(1:nrow(y) + i/60 , y[,i], type="l", col=rainbow(30)[i], ylab="Spikes per 100 cycles", xlab="Cycle (x 100)")

image(y, ylab="Neuron", xlab="Cycle (x 100)")
