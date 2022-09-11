# dmdgp-mlm
Minimimal Learning Machine (MLM) applied to the Discretizable Molecular Distance Geometry Problem

## Gradient of the MLM training objective function 
From Eq. (7) of [1], we have
$$J(y)=\sum_{i=1}^{k}(\delta^2(y,t_i)-\hat\delta_i^2)^2=\sum_{i=1}^{k}c_i^2(y),$$
where each $\hat\delta_i\in\mathbb{R}$ is constant e $c_i(y)=\delta^2(y,t_i)-\hat\delta_i^2$.

From this, we get
$$\frac{\partial J(y)}{\partial y_j}=\sum_{i=1}^k\frac{\partial c_i^2(y)}{\partial y_j}=\sum_{i=1}^k2c_i(y)\frac{\partial c_i(y)}{\partial y_j}$$

If $\delta(y,t_i)=||y-t_i||_2$ (Euclidean distance), then
$$\frac{\partial c_i(y)}{\partial y_j}=\frac{\partial ((y-t_i)'(y-t_i) -\hat\delta^2_i)}{\partial y_j}=2(y_j-t_{ij}).$$

Finally, we conclude that
$$\frac{\partial J(y)}{\partial y_j}=4\sum_{i=1}^kc_i(y)(y_j-t_{ij})$$

# References
1. de Souza Junior, A. H., Corona, F., Barreto, G. A., Miche, Y., & Lendasse, A. (2015). Minimal learning machine: a novel supervised distance-based approach for regression and classification. Neurocomputing, 164, 34-44.

2. 'Lavor, Carlile. "On generating instances for the molecular distance geometry problem." Global optimization. Springer, Boston, MA, 2006. 405-414.
