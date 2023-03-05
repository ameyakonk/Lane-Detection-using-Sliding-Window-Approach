# Lane-Detection-using-Sliding-Window-Approach

1.  Developed a project to detect road lanes by creating Sliding windows on the lanes and classified degree of turn based on the curvature of the detected lane.
2.  Considered a polygon between two lanes in an image frame and performed homography transformation on it to convert it to an aerial view. Thresholded the image and predicted  window centers around the lane patches.
3.  Used the window centers to predict the lane coordinates and thus the angle of curvature. 
