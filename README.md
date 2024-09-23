##  Non-overlapping, Schwarz-type Domain Decomposition Method for Physics and Equality Constrained Artificial Neural Networks (PECANNs)
![plot](./forward/helmholtz/pic/psn_multi_xy55_2_u_pred.png)

## Currently under review [manuscript link](https://doi.org/10.1016/j.jcp.2022.111301)

### Abstract:  
We introduce a non-overlapping, Schwarz-type domain decomposition method employing a generalized interface condition, tailored for physics-informed machine learning of partial differential equations (PDEs) in both forward and inverse scenarios. Our method utilizes physics and equality constrained
artificial neural networks (PECANN) in each subdomain. Diverging from the
original PECANN method, which uses initial and boundary conditions to constrain
the PDEs alone, our method jointly employs both the boundary conditions and
PDEs to constrain a specially formulated generalized interface loss function
for each subdomain. This modification enhances the learning of
subdomain-specific interface parameters, while delaying information exchange
between neighboring subdomains, and thereby significantly reduces communication
overhead. By utilizing an augmented Lagrangian method with a conditionally
adaptive update strategy, the constrained optimization problem in each
subdomain is transformed into a dual unconstrained problem. This approach
enables neural network training without the need for ad-hoc tuning of model
parameters. We demonstrate the generalization ability and robust parallel
performance of our method across a range of forward and inverse problems, with
solid parallel scaling performance up to 32 processes using the Message Passing
Interface model. A key strength of our approach is its capability to solve both
Laplace's and Helmholtz equations with multi-scale solutions within a unified
framework, highlighting its broad applicability and efficiency.

## Citation
Please cite us if you find our work useful for your research:
```
@misc{hu2024nonoverlappingschwarztypedomaindecomposition,
  title={Non-overlapping, Schwarz-type Domain Decomposition Method for Physics and Equality Constrained Artificial Neural Networks}, 
  author={Qifeng Hu and Shamsulhaq Basir and Inanc Senocak},
  year={2024},
  eprint={2409.13644},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2409.13644}, 
}
```
### Funding Acknowledgment
This material is based upon work supported by the National Science Foundation under Grant No. 1953204 and in part in part by the University of Pittsburgh Center for Research Computing through the resources provided.
### Questions and feedback?
For questions or feedback feel free to reach us at [Qifeng Hu] (mailto:qih56@pitt.edu), [Inanc Senocak (mailto:senocak@pitt.edu),
