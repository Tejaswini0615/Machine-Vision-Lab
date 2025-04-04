# Machine-Vision-Lab
EXERCISE1:
1.	Arithmetic operations and Logical operations
Perform Arithmetic operations (Addition, Subtraction, Multiplication, Division) and Logical operations (AND, OR, XOR,NOT) on images (two color and two grey scale images) using open CV and without open CV.

2.	Resizing and Cropping Images
•	Resize the image to half its original dimensions.
•	Resize the image to a specific size (e.g., 200x200 pixels).
•	Crop a portion of the image and save the cropped region.

Image Rotation and Flipping
•	Rotate the image by 90, 180, and 270 degrees.
•	Flip the image horizontally and vertically.

3.	Histogram Analysis
•	Compute and display the histogram for a grayscale image.
•	Compute and display the color histograms (R, G, B) for a color image.
•	Perform histogram equalization to enhance the contrast.

EXERCISE2:
1.	Image cropping
•	Load four images and crop specific regions from each.
•	Arrange the cropped regions into a 2x2 grid to create a photo collage.
•	Save the collage as a new image  and print.

2.	Brightness Adjustment with Dynamic  Constraints
 Adjust the brightness of an image dynamically based on the average intensity of the pixels.
Steps:
1.	Compute the average intensity of the image.
2.	If the average intensity is below 100, brighten the image by adding an intensity value (e.g., +50).
3.	If the average intensity is above 200, darken the image by subtracting an intensity value (e.g., -50).
4.	Clip pixel values to stay within the valid range [0, 255].
NOTE: Implement the intensity adjustment without using built-in functions like cv2.add() or cv2.subtract().
Output: Display the original and adjusted images side by side.

3.	Cropping Using Interactive Mouse Input
Allow the user to interactively select and crop a region of interest using a mouse.

•	Load an image and display it in a window.
•	Capture mouse click-and-drag events to let the user select an ROI (Region of Interest).
•	Crop and save the selected ROI as a new image.
NOTE: Implement a "reset" option to allow multiple ROI selections in the same session.
Output: Display the cropped ROI in a separate window after the user selection.

EXERCISE3:
1.	Piece-Wise Linear Transformation (Contrast Stretching and Clipping)
 Implement piece-wise linear transformations for contrast stretching and intensity clipping.
a.	Use contrast stretching to improve the contrast of an image with low and high-intensity ranges.
b.	Experiment with intensity clipping to highlight specific intensity ranges (e.g., isolate mid-tone regions).
c.	Dynamically adjust the piece-wise linear transformation points based on the input image's histogram.
 Output:
Enhanced images and a comparison of histograms before and after transformation.
2.	Enhanced Bit-Plane Fusion with Custom Weighting

Develop a method to reconstruct the image by fusing specific bit-planes with custom weights. Analyze how different weights for bit-planes influence the reconstructed image and compare it with the original.
Steps:
1.	Decompose the Image: Break the grayscale image into 8 bit-planes, as in standard bit-plane slicing.
2.	Custom Weighted Fusion:
o	Assign weights to each bit-plane (e.g., higher-order planes contribute more to the final image).
o	Combine the bit-planes with the assigned weights to reconstruct the image. Example:
	Weight w7=0.5w7=0.5 for the 8th bit-plane (MSB),
	Weight w6=0.3 for the 7th bit-plane, etc.
3.	Reconstruct the Image: Use the formula: reconstructed=w7⋅P7+w6⋅P6+⋯+w0⋅P0  where Pi is the i-th bit-plane.
4.	Analyze Reconstruction:
o	Compare reconstructed images for different weight combinations using:
	Mean Squared Error (MSE).
	Peak Signal-to-Noise Ratio (PSNR).
	Structural Similarity Index (SSIM).
o	Visualize differences in image details as weights for lower-order planes are reduced or excluded.
Output:
•	Visualization of original and reconstructed images with varying weights.
•	Quantitative analysis of reconstruction quality.
•	Insights into the importance of specific bit-planes in image reconstruction.
3. Multi-Level Histogram Shifting with Overlap Zones
Apply multi-level histogram sliding on overlapping regions of the image and analyze the impact of overlap on the transitions.
Steps:
1.	Divide the Image:
o	Divide the image into overlapping regions (e.g., 3×33 \times 33×3 grid where adjacent regions share pixels).
o	Each region has a 50% overlap with neighboring regions.
2.	Histogram Sliding with Varying Shifts:
o	Region 1: Shift pixel values by +60+60+60.
o	Region 2: Shift pixel values by −40-40−40.
o	Region 3: Shift pixel values by +20+20+20.
o	Other regions: Apply random shifts between −50-50−50 and +50+50+50.
o	Clip values to remain within the valid range [0,255].
3.	Blend Overlap Zones:
o	In overlapping zones, compute the pixel value as the average of the intensity values modified by each region.
4.	Analysis:
o	Visualize the histogram of each region and overlapping zones before and after modifications.
o	Observe the effect of overlapping transitions on image continuity and abruptness.
Output:
•	Visualization of the modified image after applying histogram shifts and blending in overlaps.
•	Histograms of regions and overlaps before and after sliding.
•	Analysis of artifacts and image continuity in overlapping zones.

EXERCISE3_1:
1.	Noise Reduction with Moving Average Filters

(a) Implement and analyze the effect of moving average filters for image smoothing while balancing noise reduction and detail preservation.
Steps:
•	Load a grayscale or color image and add Gaussian noise and salt-and-pepper noise.
•	Apply moving average filters with different kernel sizes (e.g., 3×3 , 5×5, 7×7).
•	Compare the filtered images to observe the trade-offs between noise reduction and blurring of edges.
•	Compute PSNR(Peak Signal-to-Noise Ratio ) and SSIM (Structural Similarity Index) to quantitatively evaluate the filtering performance.
•	Test the effect of using non-square kernels (e.g., 3×5).
(b) Design an adaptive moving average filter where the kernel size adjusts based on local variance in the image.
 Steps to Design the Filter
•	Calculate Local Variance:
o	Divide the image into overlapping windows (e.g., 3x3, 5x5, etc.).
o	For each window, calculate the local variance: 
o	 

o	Where:
	xi: Intensity value of each pixel in the window.
	μ: Mean intensity value of the window.
	n: Total number of pixels in the window.
•	Define Adaptive Kernel Size:
o	Set a threshold for variance (e.g., Vlow, Vhigh).
o	Use a smaller kernel (e.g., 3x3) for high-variance regions (edges and textures) and a larger kernel (e.g., 7x7 or 9x9) for low-variance regions (smooth areas).
•	Apply the Moving Average Filter:
o	For each pixel, determine the kernel size based on the local variance.
o	Apply the moving average filter using the determined kernel size: 
o	 
o	Where:
	k: Kernel size (adaptive).
•	Merge the Results:
o	Construct the output image by applying the filter with adaptive kernel sizes across the entire image.
o	
2.	Convolution-Based Feature Extraction for Pattern Detection
 Use convolution with custom filters to detect patterns in an image (e.g., lines, corners, or specific shapes).
Steps:
•	Design custom convolution kernels for detecting horizontal, vertical, and diagonal lines,anti diagonal, corner .
•	Example horizontal line kernel: 
 −1 -1 −1
   2   2   2
  -1  -1  -1 
•	Convolve the kernels with an image to highlight specific patterns.
•	Experiment with detecting corners using a combination of line-detection filters.
•	Test the filters on natural and synthetic images (e.g., checkerboards, grids).
•	Visualize the outputs and analyze how convolution kernels enhance specific features.
•	 Extend the experiment to detect complex patterns, such as circles or custom shapes, using a combination of convolution kernels.

EXERCISE4:
1.	Combine machine learning techniques with traditional filtering methods to adaptively choose the best denoising filter.
2.	Develop a multi-stage non-linear filtering pipeline to effectively remove impulse noise (salt-and-pepper noise) from an image.  Add varying levels of salt-and-pepper noise to a grayscale image (e.g., 10%, 30%, 50% noise).
3.	Implement a real-time hybrid filtering approach combining linear and non-linear filters for denoising video frames. (try for Different type of Noises)

EXERCISE5:
Edge Metrics
a.	Edge Strength (Gradient Magnitude)
b.	Edge Sharpness (Variance of Laplacian)
c.	Edge Density
d.	Edge Contrast (Michelson Contrast)

1.	Investigate how edge detection performance is affected by varying illumination conditions (low vs. high light levels) in both synthetic and natural images. Using edge metrics compare them.
2.	Explore the fusion(Weighted average Fusion, Max/Min Fusion, Logical Operations (Binary Fusion)- OR , AND)  of edge maps from different edge detection algorithms( sobel, prewit, Robert) to improve performance under varying illumination conditions.

EXERCISE6:
1.	Automatic Threshold Selection for Canny Edge Detection 
•	Implement automatic threshold selection using Otsu’s method. 
•	Compute Otsu’s thresholding and apply it to Canny. 
•	Compare results with manually chosen thresholds. Test on images with varying contrast.

2.	Implement region-growing segmentation by selecting a seed point and expanding based on similarity criteria. Modify region growing to automatically select multiple seeds based on edge detection.
3.	Implement a quadtree-based region split and merge method for segmenting an image. Modify region splitting based on texture features (entropy, variance, Laplacian) instead of pixel intensity alone.

EXERCISE7:
1.	Multi-Scale Morphological Segmentation for Satellite Image Enhancement
•	Implement multi-scale morphological filtering for terrain segmentation.
o	 Apply small SE for fine details (e.g., roads, buildings).
o	Apply large SE for coarse structures (e.g., forests, rivers).
o	Merge results using weighted fusion.
•	Use adaptive structuring elements to process different landscapes.
•	 Urban Areas → Use rectangular SEs (aligned with buildings).
•	 Forests & Vegetation → Use circular SEs (preserve tree clusters).
•	 Mountains & Hills → Use elliptical SEs (capture sloped terrains).
•	Water Bodies → Use disk-shaped SEs (eliminate boundary noise).

2.	Dynamic Graph-Based Image Segmentation with Adaptive Edge Weights
•	Implement a graph-based segmentation algorithm that dynamically updates edge weights based on local contrast.
•	Use a self-learning mechanism to refine segment boundaries iteratively.

3.	Morphological Hit-or-Miss Transform for Document Forgery Detection
•	Implement hit-or-miss transform to detect forgery marks in scanned documents.
•	Design custom structuring elements for common fraud patterns.

EXERCISE8:
1.	Implement thickening to enhance blood vessels in X-ray angiograms.
•	Extract initial vessel structure using edge detection.
•	Apply morphological thickening iteratively.
•	Analyze vessel connectivity improvements.
•	Evaluate on real medical images (X-ray/MRI).

2.	Fill missing regions in satellite images using adaptive techniques.
•	Select satellite images with cloud-covered or missing areas.
•	Implement adaptive region filling using texture synthesis and interpolation.
•	Compare results using different structuring elements.
•	Evaluate the performance using terrain similarity metrics.

3.	Use skeletonization to preprocess handwritten text for OCR systems.
•	Use a dataset of handwritten characters.
•	Apply skeletonization to reduce characters to a single-pixel width structure.
•	Extract keypoints and compare with template matching.
•	Evaluate performance improvements in OCR accuracy.

EXERCISE9:
1.	A satellite image processing system needs to detect and classify buildings, forests, and water bodies using Convex Hull-based shape approximation.
•	Apply clustering-based segmentation to detect multiple objects.
•	Compute the Convex Hull for each detected object.
•	Analyze the convexity properties to distinguish between natural vs. man-made structures.
•	Validate results using a dataset of labeled aerial images.

2.	An industrial inspection system must detect defects in fabric rolls using GLCM-based texture analysis while ensuring rotation-invariant feature extraction.
•	Compute GLCM matrices at different orientations (0°, 45°, 90°, 135°).
•	Normalize and average features across orientations.
•	Train a support vector machine (SVM) to classify textures into defective and non-defective.
•	Evaluate performance using rotated test images.

3.	A biometric authentication system needs to differentiate between real human faces and spoofed images (e.g., printed photos used to bypass facial recognition).
•	Capture face images and apply LBP feature extraction.
•	Train a deep learning model to classify real vs. spoofed faces.
•	Implement real-time processing for anti-spoofing detection.
•	Test the system against different spoofing attacks (printed images, 3D masks).

EXERCISE10:
1.	Hybrid Image Representation using FFT and DCT
•	Apply FFT to extract low-frequency components from Image A.
•	Apply DCT to extract high-frequency components from Image B.
•	Combine the low-frequency and high-frequency components to generate a hybrid image (e.g., visible from different distances).
•	Analyze the perceptual effect of hybrid images at varying resolutions.
o	Downscale and upscale the hybrid image.
o	Observe how different frequencies dominate at different distances.

2.	Object Detection using SIFT & Haar Wavelets

•	 Apply SIFT feature extraction on an image.
•	 Use Haar-based wavelet decomposition to remove low-energy noise.
•	Perform feature matching between a query image and multiple test images.
•	Compare recognition accuracy before and after noise filtering.
