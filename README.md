# GSP-KalmanNet
Dynamic systems of graph signals are encountered in various applications, including social networks, power grids, and transportation. 
While such systems can often be described as state space (SS) models, tracking graph signals via conventional tools based on the Kalman filter (KF) and its variants
is typically challenging. This is due to the nonlinearity, high dimensionality, irregularity of the domain, and complex modeling associated with real-world dynamic systems of graph signals.
In this work, we study the tracking of graph signals using a hybrid model-based/data-driven approach. We develop the GSP-KalmanNet, which tracks the hidden graphical states from
the graphical measurements by jointly leveraging graph signal processing (GSP) tools and deep learning (DL) techniques. The derivations of the GSP-KalmanNet are based on extending the 
KF to exploit the inherent graph structure via graph frequency domain filtering, which considerably simplifies the computational complexity entailed in processing high-dimensional signals 
and increases the robustness to small topology changes. Then, we use data to learn the Kalman gain following the recently proposed KalmanNet framework, which copes with partial and
approximated modeling, without forcing a specific model over the noise statistics. Our empirical results demonstrate that the proposed GSP-KalmanNet achieves enhanced accuracy and run
time performance as well as improved robustness to model misspecifications compared with both model-based and data-driven benchmarks.
This article was submitted to IEEE Signal Processing Transaction.



In order to run the simulations all files except from data need to be on the same folder.

The Pypowermain runs Pypower on EKF mode, there are 4 boolean variables that can change for other types - EKF,GSP_EKF, GSP_KNet, KNet.

For other simulations except Pypower need to change the data in Pypower main.

For running with Colab you can register the link https://colab.research.google.com/drive/1kMzFZgOT5pRde7Sm43yme4cfRLHhHE6R?authuser=2#scrollTo=xgguNTR1nLz3
