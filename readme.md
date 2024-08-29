## 离散部分

目前仅仅测试了lunarlander环境,实验的算法有：DQN，DDQN，C51，duelDQN，QRDQN，NoiseDQN，PriorDQN，NstepDQN，Rainbow，QRRainbow，SAC，PPO, DRQN

然后里面我实验了一些超参数，主要是3个优化，state_norm,reward_scale,lr_decay,上面的算法其他的所有公共参数，都是一样的。

然后我放弃了HDQN，因为不太适合这个环境。


## 连续部分

主要是mujoco环境，实验的算法有：DDPG，PPO，TD3，SAC


## 离线强化学习部分

主要是mujoco环境，实验的算法有：IQL，CQL，DT，EDAC，这个部分如果要实验，需要自己建立一个dataset文件夹，下载d4rl数据集。



## 离线到在线强化学习部分

主要是mujoco环境，实验的算法有：AWAC，offlinetoonline，PEX，SO2，不过这里并没有达到论文里描述的结果。


## 暂时的总结

以上所有的算法都已收敛，主要是表达，复现的没有问题(当然，能跑出来也不一定没问题)。


## 待办

目前所有的算法都是从头开始写的，其实可以提取公共部分，设置公共的函数或者类，这样效率更高，现在暂时没啥时间，先这样吧。

还有后续一些我感兴趣的算法我都会放在这里。




## 环境

python == 3.9.18  
torch == 2.1.0  
mujoco == 3.1.1  
gymnasium == 0.29.1  
numpy == 1.26.0  
h5py == 3.10.0  