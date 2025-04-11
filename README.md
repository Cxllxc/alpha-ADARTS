# $\alpha$-ADARTS: Alpha channel attention for differentiable architecture search
https://ieeexplore.ieee.org/document/10933613
# abstract

The use of gradient guided search in Differentiable neural architecture search(DARTS) significantly reduces the cost of neural network architecture search. However, DARTS suffers from imbalanced competition among candidate operations, and issues like poor architecture stability caused by the emergence of a large number of weight-free operations in the later stages of search. This paper proposes a novel long scope attention mechanism based on partial channel connection for differentiable neural architecture search, termed $\alpha$ -DARTS. Channels are given weights by the attention mechanism, allowing only important channels to enter the mixing operation, which significantly enhance search efficiency and memory utilization. Different from the existing methods limited to feature maps, its scope includes the feature maps and the architecture of computing cells. By incorporating network architecture parameters into the scope of attention machines, the stability of long-term search processes has been greatly improved. Experimental results demonstrate that we achieve an error rate of 2.49% on CIFAR10 and 16.07 on CIFAR100 using only 0.05 GPU-days.

# Structure obtained from search

![reduction cell](reduction.pdf "reduction cell")

![normal cell](normal.pdf "normal cell")

Part of results listed here, including the weight of cifar10 \& 100.
[bscifar10weights](bscifar10weights.pt "bscifar10weights"), [bscifar100weights](bscifar100weights.pt "bscifar100weights")

# Core code(attention module)    
        out = self.fc2(self.relu1((torch.mm(weights_all.t(), weights_all).expand(len(x), -1, -1) @ (self.fc1(torch.cat((self.avg_pool(x), self.max_pool(x)), dim=1))[:, :, :, 0])).unsqueeze(-1)))

# verify
You can verify the results through [test.py](test.py "test.py"). The script is just for cifar10, if you want to test the weights for cifar100 please give it a little change.(just change the dataloader and the num of classes) :heart: :heart: :heart:

The evaluation and validation results of CIFAR100 are in [eval-EXP-20231201-135135](eval-EXP-20231201-135135 "eval-EXP-20231201-135135")
cifar10: [eval-EXP-20240118-165737](eval-EXP-20240118-165737 "eval-EXP-20240118-165737")


There may be slight differences between different versions of cuda and pytorch.
Performance slightly degradation observed in the latest versions of cuda and pytorch.

If you wish to communicate with me, you can communicate with me via email. Thank you for your visit.
