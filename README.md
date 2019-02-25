# CuriosityViz
This project explores various visualization techniques that could help us visualize the policy development of the RL agent under different reward settings.

## Dependencies
* numpy
* pytorch
* scipy
* cv2
* gym[atari]

## Installing open AI gym.
```
pip install gym[atari]
```
The above installation doesn't work on windows. I found the following article useful:
* [Installing Gym on Windows](https://medium.com/@SeoJaeDuk/archive-post-how-to-install-open-ai-gym-on-windows-1f5208c16179)

## Instruction to run

To Train Model:
python main.py (set params as needed)

To Generate Saliency Maps(will saved in Frames folder):
python viz_saliency.py
