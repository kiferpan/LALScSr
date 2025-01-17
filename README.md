# LALScSr: Layer-wise Adaptive Learning of Spectral Correction and Spatial Rebuild Network for Hyperspectral Image Enhancement 

Implementation of Paper: "LALScSr: Layer-wise Adaptive Learning of Spectral Correction and Spatial Rebuild Network for Hyperspectral Image Enhancement" in PyTorch


## Prepare Training Dataset

1. CAVE:  [website](https://www1.cs.columbia.edu/CAVE/databases/multispectral/)
2. Houston: [website](https://hyperspectral.ee.uh.edu/?page_id=1075)

## Network Structure
<p>
  <img src='Fig/Network_structure.png'/>
</p>

## Experimental Results
### CAVE:
<p>
  <img src='Fig/CAVE_result.png' />
</p>
Fig. The distribution of SAM value in CAVE dataset for partial data by different methods. (a) RGB, (b) CNMF, (c) Bayesian, (d) GFPCA, (e) DDLPS, (f) GuidedNet, (g) ADKNet, (h) BUSIFusion, (i) the proposed method. Color bar range is [0,0.2] to highlight the spectral differences more clearly.

### Houston:
<p>
  <img src='Fig/Houston_result.png' />
</p>
Fig. SThe distribution of SAM value in Houston dataset for partial data by different methods. (a) RGB, (b) CNMF, (c) Bayesian, (d) GFPCA, (e) DDLPS, (f) GuidedNet, (g) ADKNet, (h) BUSIFusion, (i) the proposed method. Color bar range is [0,0.2] to highlight the spectral differences more clearly.

## Further Results
Natural image superresolution (SR) model applied in our method.
<p>
  <img src='Fig/Further_result.png'/>
</p>
Fig. The results of different SR models. (a) Ground Truth, (b) Bicubic, (c) ESRNet, (d) LapSRN, (e) Piexlshuffle + DSHFCN, (f) (b) + DSHFCN (default), (g) (c) + DSHFCN, (h) (d) + DSHFCN. The first line is the spectral distortion map, where the color bar range is [0,0.2], and the second line is the visual HSI from the CAVE dataset, where [r,g,b]=[30,20,10].

## Training Code
Coming soon...

If you have any questions, please contact the author at fuyihao@stumail.nwu.edu.cn


