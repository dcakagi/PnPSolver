# PnPSolver
Based on code from the paper "Revisiting the PnP Problem: A Fast, General and Optimal Solution" by Zheng et al. available at https://openaccess.thecvf.com/content_iccv_2013/papers/Zheng_Revisiting_the_PnP_2013_ICCV_paper.pdf.

MATLAB code performing the OPnP approach outlined in the paper above is called within the PoseEstimator Python class and requires installation of the MATLAB engine API (instructions at https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).


In Linux:

`cd "<matlabroot>/extern/engines/python`

`python setup.py install`
