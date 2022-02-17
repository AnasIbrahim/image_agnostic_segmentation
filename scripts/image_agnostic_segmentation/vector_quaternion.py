import numpy as np

# https://answers.ros.org/question/228896/quaternion-of-a-3d-vector/
# http://lolengine.net/blog/2013/09/18/beautiful-maths-quaternion-from-vectors
def pose_from_vector3D(waypoint):
    x = waypoint[0]
    y = waypoint[1]
    z = waypoint[2]
    #calculating the half-way vector.
    u = [1,0,0]
    norm = np.linalg.norm(waypoint)
    v = np.asarray(waypoint)/norm
    if (np.array_equal(u, v)):
        w = 1
        x = 0
        y = 0
        z = 0
    elif (np.array_equal(u, np.negative(v))):
        w = 0
        x = 0
        y = 0
        z = 1
    else:
        half = [u[0]+v[0], u[1]+v[1], u[2]+v[2]]
        w = np.dot(u, half)
        temp = np.cross(u, half)
        x = temp[0]
        y = temp[1]
        z = temp[2]
    norm = np.math.sqrt(x*x + y*y + z*z + w*w)
    if norm == 0:
        norm = 1
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    return np.array([x, y, z, w])
