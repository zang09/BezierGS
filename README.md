# BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting

### [[Project]]() [[Paper]](http://arxiv.org/abs/2506.22099) 

> [**BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting**](http://arxiv.org/abs/2506.22099)  
> [Zipei Ma](https://xiao10ma.github.io/)<sup>⚖</sup>, [Junzhe Jiang](https://selfspin.github.io/)<sup>⚖</sup>, [Yurui Chen](https://github.com/fumore), [Li Zhang](https://lzrobots.github.io)<sup>✉</sup>  
> **Shanghai Innovation Institute; School of Data Science, Fudan University**<br>
> **ICCV 2025**

**Official implementation of "BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting".** 

## 🛠️ Pipeline
<div align="center">
  <img src="assets/pipeline.png"/>
</div><br/>

## 🎞️ Demo

**BézierGS.mp4**

[![BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting, BézierGS.mp4 - YouTube](https://res.cloudinary.com/marcomontalbano/image/upload/v1751600146/video_to_markdown/images/youtube--lSMn9V2rBLc-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=lSMn9V2rBLc "BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting, BézierGS.mp4 - YouTube")

**pedestrian.mp4**

[![BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting, pedestrian.mp4 - YouTube](https://res.cloudinary.com/marcomontalbano/image/upload/v1751600597/video_to_markdown/images/youtube--sMb0xTdMumg-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://www.youtube.com/watch?v=sMb0xTdMumg "BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting, pedestrian.mp4 - YouTube")

## 🚀 Get started
### Environment
```bash
# Clone the repo.
git clone https://github.com/fudan-zvg/BezierGS --recursive
cd BezierGS

# Make a conda environment.
conda create --name bezier python=3.10 -y
conda activate bezier

# Install requirements.
pip install -r requirements.txt

# Install simple-knn
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git
pip install ./simple-knn

# a modified gaussian splatting (for feature rendering)
git clone --recursive https://github.com/SuLvXiangXin/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# Install nvdiffrast (for Envlight)
git clone https://github.com/NVlabs/nvdiffrast
pip install ./nvdiffrast
```

### Data preparation

Create a directory for the data: `mkdir dataset`. We provide some processed data [here](https://drive.google.com/drive/folders/1Uo-cNq6mSRCk1zbddKcORsFOoKQbp61k?usp=drive_link).

<details> <summary>Prepare Waymo Open Dataset.</summary>

We provide the split file following [EmerNeRF](https://github.com/NVlabs/EmerNeRF). You can refer to this [document](https://github.com/NVlabs/EmerNeRF/blob/main/docs/NOTR.md) for download details.

#### Preprocess the data (prepare_1.sh & prepare_2.sh)
---
Create waymo env
```bash
conda create -n waymo python=3.8 -y
conda activate waymo

pip install requirements-data.txt
pip install git+https://github.com/gdlg/simple-waymo-open-dataset-reader.git
```
---

Preprocess the example scenes
```bash
conda deactivate
conda activate waymo

python script/waymo/waymo_converter.py --root_dir TRAINING_SET_DIR --save_dir SAVE_DIR --split_file script/waymo/waymo_splits/demo.txt --segment_file script/waymo/waymo_splits/segment_list_train.txt
```

Generating LiDAR depth
```bash
conda deactivate
conda activate waymo

python script/waymo/generate_lidar_depth.py --datadir DATA_DIR
```

---
Create grounding_sam env (following this [repo](https://github.com/xiao10ma/Grounded-Segment-Anything))
```bash
conda create -n grounded_sam python=3.9 -y
conda activate grounded_sam

pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

cd Grounded-Segment-Anything
pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
pip install --upgrade 'diffusers[torch]'

git submodule update --init --recursive
cd grounded-sam-osx && bash install.sh

git clone https://github.com/xinyu1205/recognize-anything.git
pip install -r ./recognize-anything/requirements.txt
pip install -e ./recognize-anything/

pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel termcolor imageio

# Optional (if invoke torch error)
pip install --force-reinstall "numpy==1.26.4" "transformers==4.35.2"
```

Download checkpoint
```bash
mkdir weights && cd weights
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```

---
Generating sky mask
```bash
conda deactivate
conda activate grounding_sam

python script/waymo/generate_sky_mask.py --datadir DATA_DIR --sam_checkpoint SAM_CHECKPOINT
```

Generating instance segmentation
```bash
conda deactivate
conda activate grounding_sam

bash waymo_segmentation.sh $SCENE_ID
```

#### Regenerating pointcloud.npz (regenerate.sh)
If the extrinsics are fixed, you need to regenerate the point cloud and camera projections accordingly.
```bash
conda deactivate
conda activate bezier

bash regenerate.sh $SCENE_ID
```

</details>


### Training (run.sh)

```
CUDA_VISIBLE_DEVICES=0 python train.py \
--config configs/waymo/017.yaml \
source_path=dataset/017 \
model_path=eval_output/waymo_nvs/017
```

After training, evaluation results can be found in `{EXPERIMENT_DIR}/eval_output` directory.

### Rendering

```
CUDA_VISIBLE_DEVICES=0 python render.py \
--config configs/waymo/017.yaml \
source_path=dataset/017 \
model_path=eval_output/waymo_nvs/017 \
checkpoint=eval_output/waymo_nvs/017/chkpnt30000.pth \
render_split=test \
render_camera_id=0 \
render_interp_steps=10 \
render_fps=30
```

### Evaluating (eval.sh)

You can also use the following command to evaluate.

```
CUDA_VISIBLE_DEVICES=0 python evaluate.py \
--config configs/waymo/017.yaml \
source_path=dataset/017 \
model_path=eval_output/waymo_nvs/017 \
checkpoint=eval_output/waymo_nvs/017/chkpnt30000.pth
```

## 📜 BibTeX

``` bibtex
@inproceedings{Ma2025BezierGS,
  title={BézierGS: Dynamic Urban Scene Reconstruction with Bézier Curve Gaussian Splatting},
  author={Ma, Zipei and Jiang, Junzhe and Chen, Yurui and Zhang, Li},
  booktitle={ICCV},
  year={2025},
}
```
