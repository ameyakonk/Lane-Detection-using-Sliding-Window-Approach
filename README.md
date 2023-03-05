# Lane-Detection-using-Sliding-Window-Approach

1.  Developed a project to detect road lanes by creating Sliding windows on the lanes and classified degree of turn based on the curvature of the detected lane.
2.  Considered a polygon between two lanes in an image frame and performed homography transformation on it to convert it to an aerial view. Thresholded the image and predicted  window centers around the lane patches.
3.  Used the window centers to predict the lane coordinates and thus the angle of curvature. 

## Personnel

**Ameya Konkar**
**M.Eng. in Robotics  University of Maryland, College Park**

## Results
#### Original Image
<img src="https://user-images.githubusercontent.com/78075049/222947402-0db3b611-ebe7-4da2-ab31-fe4a340041bf.png" width="700" height="380">

#### After appling homography transformations
<img src="https://user-images.githubusercontent.com/78075049/222947469-3a409a55-5391-4e4e-a483-0fc8b264748b.png" width="700" height="380">

#### Final result
<img src="https://user-images.githubusercontent.com/78075049/222947342-5e4a342a-4e3f-4dd3-8ab4-9a97812cebcd.png" width="900" height="480">


