from posixpath import expanduser
import airsim
from airsim.client import TiltrotorClient
import numpy as np
import time

client = airsim.TiltrotorClient("10.32.114.184")
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
state = client.getTiltrotorState()

landing_point = np.array([-35880.0,-41940.0,4870.0+350])
start_location = np.array([-56858.316406,-43683.730469,5117.776855])

# end = np.array([20978.316406, 1743.730469, -247.776855])
end_wrt_start = landing_point-start_location

# zoom = 4
# width = 3840
# FOV = 90
# focal_length = width / (2 * np.tan(np.deg2rad(FOV)/2))


# FOV_new = np.rad2deg(2 * np.arctan(width/(2*focal_length * zoom)))
# client.simSetCameraFov("0",FOV_new)

height = 50*100
dist = 300*100
# start = np.subtract(end, [dist, 0, np.tan(np.deg2rad(7))*dist-height])

startx = end_wrt_start[0] - dist
starty = end_wrt_start[1]
startz = end_wrt_start[2] - height - np.arctan(np.deg2rad(7))*dist

dx = 1
xs = np.arange(startx,end_wrt_start[0]+dx,dx)
ys = np.array([end_wrt_start[1]]*len(xs))

x = np.arange(-(len(xs)-1),dx,dx)
zs = np.arctan(np.deg2rad(7))*x - height + end_wrt_start[2]

waypoints = np.array([xs,ys,zs],dtype=float)
waypoints = np.swapaxes(waypoints,0,1)
# waypoints = np.array([[150,10,-20]],dtype=float)
# waypoints = np.array([[0,5,-1]],dtype=float)
# waypoints = np.array([[150,10,-40]],dtype=float)
# waypoints = np.array([[208.7019,17.4373,-3.2370],[2,2,-2],[3,3,-3]],dtype=float)

N = 10000
zs = np.arange(zs[-1],end_wrt_start[2],.5)
xs = np.array([xs[-1]]*len(zs))
ys = np.array([ys[-1]]*len(zs))

end_waypoints = np.array([xs,ys,zs],dtype=float)
end_waypoints = np.swapaxes(end_waypoints,0,1)
end_waypoints = np.vstack((end_waypoints,np.array(end_waypoints[-1].tolist()*5000).reshape(5000,3)))
waypoints = np.vstack((waypoints,end_waypoints))
# waypoints = waypoints[0]
# client.simPause(True)
destination_point = landing_point - start_location
# zoom_prev = zoom+2
for w in waypoints:

    
    # if zoom != zoom_prev:
    #     f = open("FOV", "w")
    #     f.write(str(FOV_new))
    #     f.close()
    
    position = airsim.Vector3r(w[0]/100,w[1]/100,w[2]/100)
    rot = airsim.to_quaternion(np.deg2rad(-10.0),0,0)
    orientation = rot
    pose = airsim.Pose(position,orientation)
    start = time.time()
    total = .01
    tilt =  [float("NaN"),float("NaN"),float("NaN")]
    
    # while(True):
    while(time.time()-start < total):
        client.simSetTiltrotorPose(pose,tilt,True,vehicle_name='0',spin_props=True)
    pose = client.getTiltrotorState().kinematics_true

    # pose.position = airsim.Vector3r(0,0,0)
    dx = destination_point[0] - pose.position.x_val*100
    dz = destination_point[2] - pose.position.z_val*100

    aircraft_pitch = -np.rad2deg(airsim.to_eularian_angles(pose.orientation)[0])
    pitch = 90 - aircraft_pitch -  np.rad2deg(np.arctan(abs(dx/dz)))
    # print()

    pose_cam = client.simGetCameraInfo("0").pose
    pose_cam.orientation = airsim.to_quaternion(np.deg2rad(-pitch),0,0)
    pose_cam.position = airsim.Vector3r(0,0,0)
    client.simSetCameraPose("0",pose_cam)
    # print(-pitch)
