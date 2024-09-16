### 💾 Installation
changed from [make it real]{https://github.com/Aleafy/Make_it_Real/}
module load if it is possible
```
module load sqlite3/3.42.0 tensorrt/8.6.1.6-cuda-12.X cmake/3.28.3  openblas/0.3.23 cudnn/v8.9.7.29-prod-cuda-12.X cuda/12.1 gcc/11.5.0-binutils-2.43    blender/3.6.2
```

1. Install basic modules: torch and packages in requirements.txt
   
   ```bash
   cd seg_material

   conda create -n mkreal python=3.8 
   pip install torch torchvision torchaudio
   pip install -r requirements.txt
   ```

2. Install rendering utils:  Kaolin
   
   ```bash
   pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.10.1_cu111.html
   ```
   
3. Install segment & mark utils:
   ```
   # install Semantic-SAM
   pip install git+https://github.com/UX-Decoder/Semantic-SAM.git@package
   # install Deformable Convolution for Semantic-SAM
   cd som/ops && sh make.sh && cd ..
   # install detectron2
   python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
   ```
   Get model weight through `bash som/ckpts/download_ckpt.sh`