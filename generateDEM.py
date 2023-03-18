import numpy as np
import cv2
import open3d as o3d
import copy
import matplotlib.pyplot as plt

binPath = "./sampleKittiData/LiDAR/000010.bin"

# read and center pcd
def readPCD(filePath):
    pcd = o3d.geometry.PointCloud()
    point_list = []
    
    f = open(filePath, "rb")
    point_list = np.fromfile(f, dtype=np.float32)
    point_list = np.reshape(point_list, (-1, 4))
    point_list  = np.delete(point_list, 3, -1)

    point_list -= np.mean(point_list)

    pcd.points = o3d.utility.Vector3dVector(point_list)
    return pcd

def generateWorldPlane(width, height, numPoints):
    worldPlane = np.zeros([numPoints * numPoints, 3])
    
    w = float(width)
    h = float(height)

    x = np.linspace(-w/2, w/2, numPoints)
    y = np.linspace(-h/2, h/2, numPoints)
    xv, yv = np.meshgrid(x, y)

    worldPlane[:, 0] = xv.flatten()
    worldPlane[:, 1] = yv.flatten()

    planePcd = o3d.geometry.PointCloud()
    planePcd.points = o3d.utility.Vector3dVector(worldPlane)
    return planePcd

def getC(plane, numPoints=50):
    C = np.zeros((numPoints, 3))
    
    p = np.array(plane)
    distFromOrigin = plane[3]

    if distFromOrigin > 1.0:
        return C
    
    # calculate radius of the circle of dist 1 points
    norm = np.linalg.norm(p[0:3])
    projOrigin = -p[0:3] * distFromOrigin/norm

    r = np.sqrt(1-d**2)

    a = np.cross(p[0:3], np.array([1,0,0]))
    b = np.cross(p[0:3], a)

    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)

    a *= r
    b *= r

    for i, angle in enumerate(np.linspace(0, 2*np.pi, numPoints)):
        point = projOrigin + a*np.cos(angle) + b*np.sin(angle)
        C[i] = point

    # print(C)
    return C

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

# read pcds
pcd = readPCD(binPath)
# o3d.visualization.draw_geometries([pcd])

# get planes n's and c's
worldPlane = generateWorldPlane(150, 150, 50)
worldPlane.paint_uniform_color([1, 0.706, 0])

# segment planes
n_c, inliers = pcd.segment_plane(distance_threshold=0.25,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = n_c
n = [a, b, c]
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd.select_by_index(inliers, invert=True).paint_uniform_color([0,0,1])


# obtain plane parameterisations, n's and c's
C_c = getC(n_c)
circle_c = o3d.geometry.PointCloud()
circle_c.points = o3d.utility.Vector3dVector(C_c)
circle_c.paint_uniform_color([0, 1.0, 0])

n_w = [0, 0, 1.0, 0]
C_w = getC(n_w)
circle_w = o3d.geometry.PointCloud()
circle_w.points = o3d.utility.Vector3dVector(C_w)
circle_w.paint_uniform_color([0, 0, 1.0])


# o3d.visualization.draw_geometries([circle_c, inlier_cloud])
# o3d.visualization.draw_geometries([circle_w, worldPlane])
# o3d.visualization.draw_geometries([worldPlane, inlier_cloud])
# o3d.visualization.draw_geometries([circle_c, circle_w])
# o3d.visualization.draw_geometries([circle_c, worldPlane, inlier_cloud, circle_w])


# coarse canonicalizaton
alpha = np.arctan(-n_c[0]/n_c[2])                                               # roll
beta = np.arctan(n_c[1]/(n_c[0] * np.cos(alpha)  -  n_c[0]*np.sin(alpha)))      # pitch

# icp b/w circle_c and circle_w
initRot = np.identity(4)
R = o3d.geometry.get_rotation_matrix_from_zyx(np.array([beta, 0, alpha]))
initRot[:3,:3] = R

# draw_registration_result(pcd, worldPlane, np.identity(4))
# draw_registration_result(pcd, worldPlane, initRot)

# draw_registration_result(circle_c, circle_w, np.identity(4))
# draw_registration_result(circle_c, circle_w, initRot)

# for p in np.asarray(circle_c.points):
#     print(p, np.linalg.norm(p))

reg_p2p = o3d.pipelines.registration.registration_icp(
    circle_c, circle_w, 2, initRot,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=500))
finalTransform = reg_p2p.transformation

print("init rot: ", initRot)
print("final rot: ", reg_p2p.transformation)

print(reg_p2p)

# draw_registration_result(circle_c, circle_w, finalTransform)

# apply canonicalization
canonicalPCD = pcd.transform(finalTransform)

# draw_registration_result(pcd, worldPlane, np.identity(4))
# draw_registration_result(canonicalPCD, worldPlane, np.identity(4))


## -- generate DEM --

def generateDEM(canonicalPCD, G_w, G_h, d):
    DEM = np.ones([G_w, G_h])
    DEM *= -np.inf

    # subsample
    points = np.array(canonicalPCD.points)
    
    points[:, :2] /= d
    points[:, :2] = np.floor(np.round(points[:, :2]))
    points[:, :2] *= d
    points = np.unique(points, axis=0)

    # sort by x, y and then z
    points = points[points[:,2].argsort(kind='mergesort')]
    points = points[points[:,1].argsort(kind='mergesort')]
    points = points[points[:,0].argsort(kind='mergesort')]

    minX = points[:,0].min()
    maxX = points[:,0].max()
    Xdist = maxX - minX
    Xres = float(Xdist)/G_w

    minY = points[:,1].min()
    maxY = points[:,1].max()
    Ydist = maxY - minY
    Yres = float(Ydist)/G_h

    for point in points:
        demLocation = [0,0]
        demLocation[0] = int((point[0] - minX) // Xres) - 1
        demLocation[1] = int((point[1] - minY) // Yres) - 1

        if DEM[demLocation[0], demLocation[1]] < point[2]:
            DEM[demLocation[0], demLocation[1]] = point[2]

    # set all missing values to the minimum height
    minHeight = np.inf
    for i in range(DEM.shape[0]):
        for j in range(DEM.shape[1]):
            if DEM[i,j] != -np.inf and DEM[i,j] < minHeight:
                minHeight = DEM[i,j]

    for i in range(DEM.shape[0]):
        for j in range(DEM.shape[1]):
            if DEM[i,j] == -np.inf:
                DEM[i,j] = minHeight

    DEM -= minHeight
    DEM /= DEM.max()

    # subsampledPCD = canonicalPCD.voxel_down_sample(voxel_size=d)
    subsampledPCD = o3d.geometry.PointCloud()
    subsampledPCD.points = o3d.utility.Vector3dVector(points)

    return DEM, subsampledPCD


mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
mesh = mesh.scale(20, [0,0,0])
o3d.visualization.draw_geometries([mesh, canonicalPCD])

DEM, subPCD = generateDEM(canonicalPCD, 500, 500, 0.2)
o3d.visualization.draw_geometries([mesh, subPCD])

DEM = cv2.resize(DEM, (800, 800), interpolation = cv2.INTER_AREA)
# DEM = cv2.blur(DEM, (1,10))

print(DEM)
print(DEM.min())
print(DEM.max())

DEM /= DEM.max()

DEM_display = cv2.applyColorMap(np.array(DEM * 256, dtype=np.uint8), cv2.COLORMAP_JET)
cv2.imshow("Kitti DEM", DEM_display)
cv2.waitKey(0)
cv2.destroyAllWindows()