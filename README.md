# ReVoRF

Learning with Unreliability: Fast Few-shot Voxel Radiance Fields with Relative Geometric Consistency (CVPR2024)

[paper](https://arxiv.org/pdf/2403.17638v1)


### Installation
```
git clone https://github.com/HKCLynn/ReVoRF.git
cd ReVoRF
pip install -r requirements.txt
```
[Pytorch](https://pytorch.org/) and [torch_scatter](https://github.com/rusty1s/pytorch_scatter) installation is machine dependent.

### Dataset Download
[NeRF Synthetic Dataset](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and [LLFF Dataset](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

### Directory structure for the datasets

<details>
  <summary> (click to expand;) </summary>

    data
    ├── nerf_synthetic     
    │   └── [chair|drums|ficus|hotdog|lego|materials|mic|ship]
    │       ├── [train|val|test]
    │       │   └── r_*.png
    │       └── transforms_[train|val|test].json
    │
    ├── nerf_llff_data     
        └── [fern|flower|fortress|horns|leaves|orchids|room|trex]
</details>


## Quick Start

- Get Depth Maps
	

	If the datasets are set in the above formats. Get DPT model and generate the depth maps. 
	```bash
	$ python download.py
	```

- Training
    ```bash
    $ python run.py --config configs/nerf/hotdog.py --render_test
    ```
    
- Evaluation
  
    ```bash
    $ python run.py --config configs/nerf/hotdog.py --render_only --render_test \
                                                  --eval_ssim --eval_lpips_vgg
    ```

## Acknowledgement
The code is heavily based on [DVGOv2](https://github.com/sunset1995/DirectVoxGO) implementation, and some functions are modified from [VGOS](https://github.com/SJoJoK/VGOS). Thank you!
