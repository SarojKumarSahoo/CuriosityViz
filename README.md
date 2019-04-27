# CuriosityViz
This project explores various visualization techniques that could help us visualize the policy development of the RL agent under different reward settings.

## Dependencies
* numpy
* pytorch
* scipy
* cv2
* gym[atari]

## Credits
https://github.com/chagmgang/pytorch_ppo_rl/blob/master/Breakout_ppo.py

https://github.com/jcwleo/random-network-distillation-pytorch

## Instruction to run Extrinsic

To Train Model:

```
python extrinsic.py 
```
## Instruction to run Intrinsic Model

To Train Model:

```
python train.py 
```


To run vizualization:

Goto d3 Viz subfolder and start a http server
```
python -m http.server 
```
## Javascript Files for Different Plots : 

Line Chart - ``` episodeStats.js```

Pie Chart - ``` pie_dist.js```

Scatter Plot - ``` overallStats.js```

Dot Plot - ``` ep_dist.js``` and ``` ep_dist1.js```

## Data Files
Intrinsic : 

```actions_final_int2.csv```

```rewards_final_int2.csv```

```episode_data_final_int2.csv```

```feature_data_final_int2.csv```

Extrinsic :

```actions_final.csv```

```rewards_final.csv```

```episode_data_final.csv```

```feature_data_final.csv```

## Report

FinalReport.pdf
