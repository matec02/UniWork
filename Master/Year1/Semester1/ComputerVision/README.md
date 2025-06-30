# Computer Vision – Laboratory Exercises

---

## Lab 1: Image Geometric Transformations

**File:** `lab1_mj.py`

Covers:
- **Affine transformation** with both **nearest-neighbor** and **bilinear** interpolation
- **RMSE** comparison between interpolation methods
- **Diamond-based affine recovery**
- **Projective (homography) transformation** using point correspondences
- Demonstrates transformations on sample images using synthesized and recovered mappings

---

## Lab 2: Edge and Corner Detection

### Canny Edge Detector

**File:** `canny.py`

Implements the full **Canny edge detection pipeline**:
- Gaussian smoothing
- Gradient computation (Sobel filters)
- Non-maximum suppression
- Double thresholding and **hysteresis** edge tracking

### Harris Corner Detector

**File:** `harrison.py`

Implements the **Harris corner detection** algorithm:
- Computes image gradients and structure tensor
- Calculates **Harris response**
- Applies **thresholding**, **non-maximum suppression**, and selects top `k` corners
- Displays detected corners overlaid on the original image

---

## Lab 4: Generative Modeling with Normalizing Flows

**File:** `RacunalniVid_LAB4.ipynb`

Focuses on **generative modeling** via **normalizing flows**:
- Implements bijective transformations (`BijectiveLinear2D`, `AffineCouplingLayer`) and architectures such as **SimpleNF** and **SimpleRealNVP**
- Trains flows to learn:
    - **2D Gaussian Mixture** distributions
    - **MNIST digit distributions** using RealNVP with residual blocks and dequantized input
- Shows how learned models can generate new data points from the modeled distribution
- Discusses **KL divergence**, **change of variable formula**, and **log-likelihood maximization**

This lab bridges classical computer vision and modern generative deep learning techniques.

---
