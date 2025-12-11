## Introduction

This repository contains an unofficial mitsuba 3 implementation of retro-reflective BRDF model proposed by Guo et al. ([link](https://www.sciencedirect.com/science/article/abs/pii/S1524070318300018)).


![Retro-reflective BRDF visualization](asset/retro_vis.png)

## Setup
```
# install requirements
pip install numpy imageio tqdm
# install mitsuba
pip install mitsuba
```

## Test
```
python val_render.py
```