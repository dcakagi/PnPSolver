import airsim
import numpy as np
import time

client = airsim.TiltrotorClient("10.32.114.170")

client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
state = client.getTiltrotorState()

landing_point = np.array([-35880.0, -41940.0, 4870.0 + 350])
start_location = np.array([-56858.316406, -43683.730469, 5117.776855])

end_wrt_start = landing_point - start_location

pixels_height = 2160
pixels_width = 3840
down_fov = 5 * np.pi / 6  # Fisheye lens for downward facing camera - 150 deg FOV
forward_fov = np.pi / 2  # 90 deg FOV for forward facing camera

fixed_camera_pitch = -7.0 * np.pi / 180  # from body of plane, in radians

client.simSetCameraFov("3", np.rad2deg(down_fov))

height = 60 * 100  # Start at height of 60 m above ending location

startx = end_wrt_start[0]
starty = end_wrt_start[1]
startz = end_wrt_start[2] - height

dx = 0.5  # in cm
z_coords = np.arange(startz, end_wrt_start[2] + dx, dx).reshape(1, -1)
y_coords = np.ones((1, z_coords.shape[1])) * starty
x_coords = np.ones((1, z_coords.shape[1])) * startx

waypoints = np.concatenate((x_coords, y_coords, z_coords), axis=0).T

for w in waypoints:
    position = airsim.Vector3r(w[0]/100, w[1]/100, w[2]/100)
    rot = airsim.to_quaternion(0, 0, 0)  # Aircraft is level during descent
    orientation = rot
    pose = airsim.Pose(position, orientation)

    start = time.time()
    total = 0.1

    tilt = [float("NaN"), float("NaN"), float("NaN")]

    while time.time() - start < total:
        client.simSetTiltrotorPose(pose, tilt, True, spin_props=True, vehicle_name='0')

    pose_aircraft = client.getTiltrotorState().kinematics_true
    pose_camera = client.simGetCameraInfo("3").pose  # Downward facing camera

    dx = end_wrt_start[0] - pose.position.x_val * 100  # Convert to cm
    dy = end_wrt_start[1] - pose.position.y_val * 100  # Should be 0 for pure vertical descent (same as dx)
    dz = end_wrt_start[2] - pose.position.z_val * 100
    total_dist = np.sqrt(dx**2 + dy**2 + dz**2)

    aircraft_pitch = -airsim.to_eularian_angles(pose_aircraft.orientation)[0]
    cam_pitch = np.pi/2 - aircraft_pitch + fixed_camera_pitch

    pose_camera.orientation = airsim.to_quaternion(-cam_pitch, 0, 0)
    pose_camera.position = airsim.Vector3r(0, 0, 0)
    client.simSetCameraPose("3", pose_camera)





