# Lane-marking-and-Object-detection
Lane marking and object detection from a car Dash camera view

## steps
1. Load the image and scale it to 80% of the original size using bilinear
interpolation
2. Normalize the image for intensity and contrast
3. Use gaussian filter, median filter and mean filter to filter noise in the image
4. Detect edges using Canny edge detector
5. Detect the road-lane markings from the edge-detected image using Hough Transform
6. Superimpose the lane markings in RED colour
7. Segment the image using region-growing algorithm
