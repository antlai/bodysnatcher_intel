An http daemon to trigger depth camera/projector calibration and calculate 3D location information of human body parts. Targets Intel Realsense cameras.

Requirements:

cherrypy
numpy
opencv-python
pyrealsense2 (https://github.com/IntelRealSense/librealsense)
tensorflow


Trained Models from Alireza Shafaei and  James J. Little, UBC, Vancouver, Canada
 https://github.com/ashafaei/dense-depth-body-parts.git

If you use this model please give credit to the original authors (not me):

    @inproceedings{Shafaei16,
        author = {Shafaei, Alireza and Little, James J.},
        title = {Real-Time Human Motion Capture with Multiple Depth Cameras},
        booktitle = {Proceedings of the 13th Conference on Computer and Robot Vision},
        year = {2016},
        organization = {Canadian Image Processing and Pattern Recognition Society (CIPPRS)},
        url = {http://www.cs.ubc.ca/~shafaei/homepage/projects/crv16.php}
    }
