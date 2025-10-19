Assignment 3 - Nerf - Volume Rendering - 3D Vision Course CMSC848F
===================================

The original assignment repository with problem descriptions can be found [here](https://github.com/848f-3DVision/assignment3)

The report documentation for this project is also hosted as Github Pages and can be found [here](https://darshit-desai.github.io/NeRF-VolumeRendering-3DVision)

##  0. Setup

### 0.1 Environment setup
You can use the python environment you've set up for past assignments, or re-install it with our `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate l3d
```

If you do not have Anaconda, you can quickly download it [here](https://docs.conda.io/en/latest/miniconda.html), or via the command line in with:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### 0.2 Data

The data for this assignment is provided in the github repo under `data/`. You do not need to download anything yourself.

##  1. Code Running Instructions

### 1.3 For Problem 1.3 xy grid and rays visualization 

```bash
python main.py --config-name=box
```
The results will be stored at `images/prob1.3_grid.png` for xy_grid and `images/prob1.3_rays.png` for rays

###  1.4. For Problem 1.4 Point sampling (10 points)

The command remains the same as above and it is not needed to be run separately for this problem as running the above command would generate results for 1.3, 1.4 and 1.5

```bash
python main.py --config-name=box
```
The results will be stored at `images/raypoints_0_points.png` for rays point sampling.

###  1.5. Volume rendering (30 points)

The command remains the same as above and it is not needed to be run separately for this problem as running the above command would generate results for 1.3, 1.4 and 1.5

```bash
python main.py --config-name=box
```
The results will be stored at `images/part_1.gif` for volume rendering of the box and `images/part_1_depth.png` for depth render image.


###  2. For Problem 2 Implicit Volume Rendering

Run the following command:

```bash
python main.py --config-name=train_box
```

The result for the box dimensions would be printed in the terminal and is also mentioned in the documentation, the gif spiral render is generated and stored here `images/part_2.gif`.

###  3. For Problem 3 NeRF rendering
Run the following command:
```bash
python main.py --config-name=nerf_lego
```

This will create a NeRF with the `NeuralRadianceField` class in `implicit.py`, and use it as the `implicit_fn` in `VolumeRenderer`. It will also train a NeRF for 250 epochs on 128x128 images.

After training, a spiral rendering will be written to `images/part_3.gif`.

