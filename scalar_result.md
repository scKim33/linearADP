# PID Simulation of Pendulum System
***
![fig1](scalar_result/fig1.png)

System equation : $\ddot{\theta}+\frac{g}{l}\sin\theta=\frac{T_c}{ml^2}$

Use $l=20m$,$m=0.1kg$ and $g=9.81m/sec^2$

Initial state : position = 0 rad, angular velocity = 0.1 rad/s
## P Control
### 1. K = (1, 0, 0)
![1-0-0](scalar_result/1-0-0.png)
### 1. K = (10, 0, 0)
![10-0-0](scalar_result/10-0-0.png)
### 1. K = (15, 0, 0)
![15-0-0](scalar_result/15-0-0.png)
## PD Control
### 1. K = (15, 0, 3)
![15-0-3](scalar_result/15-0-3.png)
### 1. K = (15, 0, 10)
![15-0-10](scalar_result/15-0-10.png)
### 1. K = (15, 0, 20)
![15-0-20](scalar_result/15-0-20.png)
## PID Control
### 1. K = (15, 1, 10)
![15-1-10](scalar_result/15-1-10.png)
### 1. K = (15, 3, 10)
![15-3-10](scalar_result/15-3-10.png)
### 1. K = (15, 7, 10)
![15-7-10](scalar_result/15-7-10.png)