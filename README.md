# \alpha-ADARTS
# abstract
	The use of gradient guided search in Differentiable neural architecture search(DARTS) significantly reduces the cost of neural network architecture search. 
	However, DARTS suffers from imbalanced competition among candidate operations, and issues like poor architecture stability caused by the emergence of a large number of weight-free operations in the later stages of search. 
	This paper proposes a novel long scope attention mechanism based on partial channel connection for differentiable neural architecture search, termed $\alpha$-DARTS. 
	Channels are given weights by the attention mechanism, allowing only important channels to enter the mixing operation, which significantly enhance search efficiency and memory utilization. 
	Different from the existing methods limited to feature maps, its scope includes the feature maps and the architecture of computing cells. 
	By incorporating network architecture parameters into the scope of attention machines, the stability of long-term search processes has been greatly improved. 
	Experimental results demonstrate that we achieve an error rate of 2.49\% on CIFAR10 and 15.9\% on CIFAR100 using only 0.05 GPU-days.

# Structure obtained from search
![reduction cell](reduction.pdf "reduction cell")
![normal cell](normal.pdf "normal cell")

Part of results listed here, including the weight of cifar10 \& 100 ([bscifar10weights](bscifar10weights.pt "bscifar10weights"), [bscifar100weights](bscifar100weights.pt "bscifar100weights"))

The article has not been published yet. If you wish to communicate with me, you can communicate with me via email. Thank you for your visit.
