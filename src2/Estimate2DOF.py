import numpy as np 
import open3d as o3d
import copy
import math
import matplotlib.pyplot as plt 
import cv2
# import YawEstimator as YM
from scipy.optimize import minimize
import torch.nn as nn
import torch

import kornia as K
import kornia.feature as KF
from kornia_moons.feature import *
import time 


x = np.linspace(0, 1, 50, endpoint=False)

xv, yv = np.meshgrid(x, x )

xv = np.reshape( xv , ( 2500 , ) )
yv = np.reshape( yv , ( 2500 , ) )

original_pts = np.zeros( ( 2500  , 3  ) )

one = np.zeros( (2500, ))

original_pts[:, 0] = xv
original_pts[:, 1] = yv
original_pts[:, 2] = one

C = math.cos 
S = math.sin

original_pts -= np.mean( original_pts , axis =0 )

Numpts = len( original_pts )


for i in range( Numpts ):

	original_pts[i] = original_pts[i]/ np.linalg.norm( original_pts[i] )
sift =  cv2.SIFT_create()


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def determine_R_matrix(original_pcd, moved_pcd, correspondance_list):

	# original_pcd = np.asarray(original_pcd.points)
	# moved_pcd = np.asarray(moved_pcd.points)
	num_pairs = len(correspondance_list)
	original_pcd = np.asarray(original_pcd)
	moved_pcd = np.asarray(moved_pcd) 
	pt2_temp_original = np.zeros((3, 1))
	pt1_temp_moved = np.zeros((3, 1))
	cov = np.zeros((3, 3))
	weight = 1
	for index_in_moved, index_in_original in correspondance_list:

		pt1_temp_moved[0][0] = moved_pcd[index_in_moved][0]
		pt1_temp_moved[1][0] = moved_pcd[index_in_moved][1]
		pt1_temp_moved[2][0] = moved_pcd[index_in_moved][2]  
		pt2_temp_original[0][0] = original_pcd[index_in_original][0]
		pt2_temp_original[1][0] = original_pcd[index_in_original][1]
		pt2_temp_original[2][0] = original_pcd[index_in_original][2]
		dist = np.linalg.norm(pt1_temp_moved - pt2_temp_original)  
		if( dist >  0.02):
			weight = 1     
		cov_temp = pt2_temp_original.dot(pt1_temp_moved.T)
		cov += cov_temp*weight   
	u ,s,v = np.linalg.svd( cov)
	R = u.dot(v) 
	return R



def determine_correspondance_yaw(original_pcd, moved_pcd):

	# original_pcd = np.asarray(original_pcd.points)
	# moved_pcd = np.asarray(moved_pcd.points)

	correspondance_index = []
	original_pcd = np.asarray(original_pcd)
	moved_pcd = np.asarray(moved_pcd)
	original_size = np.shape(original_pcd)[0]
	moved_size = np.shape(moved_pcd)[0]
	pt2  = np.zeros( (3,1))

	for i in range(moved_size):

		pt2_temp = moved_pcd[i][:]

		if( pt2_temp[2]  > 0.2):

			# pt2_temp = moved_pcd[i][:]
			pt2[0]  = pt2_temp[0] 
			pt2[1]  = pt2_temp[1] 
			pt2[2]  = pt2_temp[2] 
			dist_vec = original_pcd.T - pt2
			dist_vec = np.linalg.norm(dist_vec , axis = 0)**2
			index = np.argmin(dist_vec)
			index_corresp = (i, index) # ( index_in_moved , index_in_original )
			correspondance_index.append(index_corresp)

	return correspondance_index

def determine_correspondance(original_pcd, moved_pcd):

	# original_pcd = np.asarray(original_pcd.points)
	# moved_pcd = np.asarray(moved_pcd.points)

	correspondance_index = []
	original_pcd = np.asarray(original_pcd)
	moved_pcd = np.asarray(moved_pcd)
	original_size = np.shape(original_pcd)[0]
	moved_size = np.shape(moved_pcd)[0]
	pt2  = np.zeros( (3,1))

	for i in range(moved_size):


		

			pt2_temp = moved_pcd[i][:]
			pt2[0]  = pt2_temp[0] 
			pt2[1]  = pt2_temp[1] 
			pt2[2]  = pt2_temp[2] 
			dist_vec = original_pcd.T - pt2
			dist_vec = np.linalg.norm(dist_vec , axis = 0)**2
			index = np.argmin(dist_vec)
			index_corresp = (i, index) # ( index_in_moved , index_in_original )
			correspondance_index.append(index_corresp)

	return correspondance_index




def Initilize_Roll_Pitch( Normal ):

	gamma = math.atan( Normal[1] / Normal[2] )

	K = Normal[1]*math.sin(gamma) + Normal[2]*math.cos(gamma) 

	K1 = K - Normal[0]
	K2 = K + Normal[0]

	beta = math.asin( 1 / np.linalg.norm([ K1, K2 ])  ) + math.atan( (K2 )/ (K1 ) )

	R_init = np.asarray([ [ C(beta) , 0 , S(beta)  ] , 
						  [ S(gamma)*S(beta) , C(gamma) , -C(beta)*S(gamma)  ], 
						  [ -C(gamma)*S(beta) , S(gamma) , C(beta)*C(gamma) ] ])

	return R_init




voxel_size = 0.15 





def InitilizeYawPitch( Normal ):

	beta = np.arctan( - Normal[0]/Normal[2])

	K1 = np.sin(beta)
	K2 = np.cos(beta)

	gamma = np.arctan( Normal[1]/ (  Normal[2]*K2 - Normal[0]*K1 ) )

	C= np.cos 
	S = np.sin 

	R = np.asarray([ [ C(beta) , 0 , S(beta) ] , 
					[ S(gamma)*S(beta) , C(gamma) , -S(gamma)*C(beta) ] , 
					[ -S(beta)*C(gamma) , S(gamma) , C(gamma)*C(beta) ] ])

	return R 

def FilterPCD( PCD ):

	Pts = PCD.points
	mean = np.mean(Pts , axis = 0)
	Pts = Pts -mean
	PCD.points = o3d.utility.Vector3dVector( Pts)

	# pcd_tree = o3d.geometry.KDTreeFlann(PCD)
	# [k, idx, _] = pcd_tree.search_radius_vector_3d([0,0,0], 25.0)

	NewPCD = o3d.geometry.PointCloud()

	# NewPCD.points = o3d.utility.Vector3dVector( np.asarray(PCD.points)[idx[1:], :] )

	# mean = np.mean(np.asarray(NewPCD.points ) , axis = 0)
	# Pts = np.asarray(NewPCD.points ) - mean 
	# NewPCD.points = o3d.utility.Vector3dVector( Pts)


	# Pts = np.asarray( PCD.points )
	Distance_from_Center = np.linalg.norm(Pts , axis=1)

	indices = np.argwhere( Distance_from_Center < 25 )

	NewPCD = o3d.geometry.PointCloud()
	NumPts = len(indices )
	Points  = np.reshape(  Pts[indices] , ( NumPts , 3) )
	NewPCD.points = o3d.utility.Vector3dVector( Points )


	return NewPCD


def Get_2DOF( PCD ):

	OriginalPCD = copy.deepcopy(PCD)
	OriginalPCD = FilterPCD(OriginalPCD)
	PCD = FilterPCD(PCD)


	PlanePointCLoud =  copy.deepcopy(PCD)
	plane_model, inliers = PCD.segment_plane(distance_threshold=0.001,
                                         ransac_n=3,
                                         num_iterations=1000)

	R_init = InitilizeYawPitch(plane_model  )

	N = plane_model[0:3]



	PlanePointCLoud = PCD.select_by_index(inliers)

	PlanePointCLoud2 = copy.deepcopy(PlanePointCLoud)

	PlanePoints = np.asarray(PlanePointCLoud.points)
	Center_plane =  np.mean( np.asarray( PlanePoints) , axis =0 )
	PlanePoints -= np.mean( PlanePoints , axis =0 )

	

	NumPlanePts = len(PlanePoints)

	for i in range( NumPlanePts ):


		PlanePoints[i] = PlanePoints[i]/ np.linalg.norm( PlanePoints[i] )

	PlanePointCLoud.points = o3d.utility.Vector3dVector(PlanePoints)
	
	RefPCD = copy.deepcopy(PCD)
	RefPCD.points = o3d.utility.Vector3dVector(original_pts)

	H = np.eye(4)
	t = [0, 0, 0]
	H[0:3,0:3] = np.ones(3)
	H[0:3, 3] = t
	H[3, :] = [0, 0 ,0 ,1]

	threshold = 0.02

	trans_init = H

	correspondance_list = determine_correspondance( np.asarray( PlanePointCLoud.points ), np.asarray( RefPCD.points)  )

	R =  determine_R_matrix( np.asarray( PlanePointCLoud.points ), np.asarray( RefPCD.points), correspondance_list) #reg_p2p.transformation[0:3,0:3]

	# print("**********")
	# print(R.T)
	# print(R_init)
	# print("---------")


	Center = np.mean(  np.asarray(PlanePointCLoud2.points) , axis=0  )
	Z = Center[2]

	OriginalPCD =  OriginalPCD.rotate(R_init, center=(Center[0], Center[1], Center[2] ))
	Pts = np.asarray(OriginalPCD.points)

	plane_model, inliers = OriginalPCD.segment_plane(distance_threshold=0.001,
                                         ransac_n=3,
                                         num_iterations=1000)


	xmax , xmin = np.amax( Pts[:, 0 ]  ) , np.amin( Pts[:, 0 ]  )
	ymax , ymin = np.amax( Pts[:, 1 ]  ) , np.amin( Pts[:, 1 ]  )
	zmax , zmin = np.amax( Pts[:, 2 ]  ) , np.amin( Pts[:, 2 ]  )

	NumGrids = 500
	resolution = 0.1 #( xmax - xmin)/NumGrids
	NumGrids = int( ( xmax - xmin)/resolution )

	HeightImage = np.zeros( ( NumGrids+1 , NumGrids+1 ) )

	A = np.asarray(plane_model)


	# for pt in range(len(Pts)):


	# 	x = min( int( (Pts[pt][0] -xmin )/ resolution ) , NumGrids)   
	# 	y = min(int( (Pts[pt][1] -ymin)/ resolution ) , NumGrids  )
	# 	Pt = [ Pts[pt][0] , Pts[pt][1] , Pts[pt][2] , 1 ]

	# 	dist_from_plane = max( A.T@Pt , 0) 

	# 	HeightImage[x][y] = max( dist_from_plane , HeightImage[x][y] )

	for pt in range(len(Pts)):

		x = min( int( (Pts[pt][0]  )/ resolution ) , NumGrids/2)    +250
		y = min(int( (Pts[pt][1] )/ resolution ) , NumGrids/2  ) +250
		Pt = [ Pts[pt][0] , Pts[pt][1] , Pts[pt][2] , 1 ]

		dist_from_plane = max( A.T@Pt , 0) *10


		if( (x < NumGrids) and (y < NumGrids) ):
			HeightImage[int(x)][int(y)] = max( dist_from_plane , HeightImage[int(x)][int(y)] )


	R = R_init

	return R , Z , HeightImage, xmin , ymin , resolution



def ExractPointClouds(  DEM1 , DEM2 , Xmin , Ymin , resolution , Pts1, Pts2):


	NumPts = len(Pts1)#np.shape(DEM1)[0]

	PCD1 = np.zeros( ( NumPts ,3 ) )
	PCD2 = np.zeros( ( NumPts ,3 ) )

	cntr = 0 
	
	for i in range(NumPts):

		X1 = Pts1[i][0]* resolution[0]  + Xmin[0] #Pixels1[i][0]*resolution[0] + Xmin[0]
		Y1 = Pts1[i][1]* resolution[0]  + Ymin[0]  #Pixels1[i][1]*resolution[0] + Ymin[0]

		X2 = Pts2[i][0]* resolution[1]  + Xmin[1]  #Pixels2[i][0]*resolution[1] + Xmin[1]
		Y2 = Pts2[i][1]* resolution[1]  + Ymin[1] #Pixels2[i][1]*resolution[1] + Ymin[1]


		PCD1[cntr] = [ X1, Y1, DEM1[ int(Pts1[i][0]) ][ int(Pts1[i][1] ) ] ]
		PCD2[cntr] = [ X2, Y2, DEM2[ int(Pts2[i][0]) ][ int(Pts2[i][1] ) ] ]

		cntr += 1


	return PCD1, PCD2 


def ExtractCompletePointClouds( DEM1 , DEM2 , Xmin , Ymin ,resolution ):

	NumPts = np.shape(DEM1)[0] -1

	PCD1 = np.zeros( ( NumPts**2 ,3 ) )
	PCD2 = np.zeros( ( NumPts**2 ,3 ) )

	cntr = 0 


	for i in range(NumPts):	

		for j in range(NumPts):

			X1 = i* resolution[0]  + Xmin[0]
			Y1 = j* resolution[0]  + Ymin[0]

			X2 = i* resolution[1]  + Xmin[1]
			Y2 = j* resolution[1]  + Ymin[1]


			if(DEM1[ i ][ j ] > 0.1 ):
				PCD1[cntr] = [ X1, Y1, DEM1[ i ][ j ] ]
			if(DEM2[ i ][ j ] > 0.1 ):
				PCD2[cntr] = [ X2, Y2, DEM2[ i][j ] ]

			cntr +=1


	return PCD1 - np.mean(PCD1 , axis=0) , PCD2 - np.mean(PCD2 , axis=0)

def hess(p,src,dst):
    n  = np.size(src,0)
    T = np.matrix([[np.cos(p[2]),-np.sin(p[2]),p[0]],
                   [np.sin(p[2]), np.cos(p[2]),p[1]],
                   [0           ,0            ,1   ]])
    n  = np.size(src,0)
    xt = np.ones([n,3])        
    xt[:,:-1] = src
    xt = (xt*T.T).A
    d  = np.zeros(np.shape(src))
    d[:,0] = xt[:,0]-dst[:,0]
    d[:,1] = xt[:,1]-dst[:,1]

    H = np.zeros([3,3])
    
    dUdth_R = np.matrix([[-np.sin(p[2]),-np.cos(p[2])],
                        [ np.cos(p[2]),-np.sin(p[2])]])
    dUdth = (src*dUdth_R.T).A

    H[0,0] = n*2
    H[0,1] = 0
    H[0,2] = np.sum(2*dUdth[:,0])
    
    H[1,0] = 0
    H[1,1] = n*2
    H[1,2] = np.sum(2*dUdth[:,1])
    
    H[2,0] = H[0,2]
    H[2,1] = H[1,2]

    d2Ud2th_R = np.matrix([[-np.cos(p[2]), np.sin(p[2])],
                           [-np.sin(p[2]),-np.cos(p[2])]])
    d2Ud2th = (src*d2Ud2th_R.T).A
    
    H[2,2] = np.sum(2*(np.square(dUdth[:,0])+np.square(dUdth[:,1]) + d[:,0]*d2Ud2th[:,0]+d[:,0]*d2Ud2th[:,0]))
    return H



def jac(p,src,dst):
    T = np.matrix([[np.cos(p[2]),-np.sin(p[2]),p[0]],
                   [np.sin(p[2]), np.cos(p[2]),p[1]],
                   [0           ,0            ,1   ]])

    n  = np.size(src,0)
    xt = np.ones([n,3])
        
    xt[:,:-1] = src
    xt = (xt*T.T).A
    d  = np.zeros(np.shape(src))
    d[:,0] = xt[:,0]-dst[:,0]
    d[:,1] = xt[:,1]-dst[:,1]

    #look at square as g(U)=sum U_i^TU_i, U_i=f_i([t_x,t_y,theta]^T)
    dUdth_R = np.matrix([[-np.sin(p[2]),-np.cos(p[2])],
                        [ np.cos(p[2]),-np.sin(p[2])]])
    dUdth = (src*dUdth_R.T).A
    g = np.array([  np.sum(2*d[:,0]),
                    np.sum(2*d[:,1]),
                    np.sum(2*(d[:,0]*dUdth[:,0]+d[:,1]*dUdth[:,1])) ])

    return g


def res(p,src,dst):
    T = np.matrix([[np.cos(p[2]),-np.sin(p[2]),p[0]],
                   [np.sin(p[2]), np.cos(p[2]),p[1]],
                   [0           ,0            ,1   ]])
    n  = np.size(src,0)
    xt = np.ones([n,3])        
    xt[:,:-1] = src
    xt = (xt*T.T).A
    d  = np.zeros(np.shape(src))
    d[:,0] = xt[:,0]-dst[:,0]
    d[:,1] = xt[:,1]-dst[:,1]

    r = np.sum(np.square(d[:,0])+np.square(d[:,1]))
    return r


def least_squared_2d_transform(src,dst,p0):
    '''
    Find the translation and roation (matrix) that
    gives a local optima to
    
    sum (T(src[i])-dst[i])^T*(T(src[i])-dst[i])
    src: (nx2) [x,y]
    dst: (nx2) [x,y]
    p0:  (3x,) [x,y,theta]
   
    '''
   

    #least squares want's 1d functions
    #result = leastsq(res,p0,Dfun=jac,col_deriv=1,full_output=1)
        
    #p_opt  = fmin_bfgs(res,p0,fprime=jac,args=(src,dst),disp=1)        
    result  = minimize(res,p0,args=(src,dst),method='Newton-CG',jac=jac,hess=hess)
    p_opt = result.x
    T_opt  = np.array([[np.cos(p_opt[2]),-np.sin(p_opt[2]),p_opt[0]],
                       [np.sin(p_opt[2]), np.cos(p_opt[2]),p_opt[1]]])
    return p_opt,T_opt



def DetermineYaw( Img1, Img2 , Z , Xmin , Ymin , resolution, viz=False):

	# Img2 = cv2.GaussianBlur(Img2,(5,5),cv2.BORDER_DEFAULT)
	# Img1 = cv2.GaussianBlur(Img1,(5,5),cv2.BORDER_DEFAULT)

	m = nn.Sigmoid()
	Img1 = m( torch.tensor(Img1)).numpy()
	Img2= m( torch.tensor(Img2)).numpy()

	DEM1 = Img1
	DEM2 = Img2

	W1, H1 = np.shape(Img1)
	W2, H2 = np.shape(Img2)


	Img1 = cv2.normalize(Img1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
	Img2 = cv2.normalize(Img2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

	keypoints_1, descriptors_1 = sift.detectAndCompute(Img1,None)
	keypoints_2, descriptors_2 = sift.detectAndCompute(Img2,None)

	bf = cv2.BFMatcher()
	matches = bf.knnMatch(descriptors_1,descriptors_2, k=2)

	good = []

	Image1_feature_points =[]
	Image2_feature_points = []

	pts1 = []
	pts2 = []

	for m,n in matches:
		if m.distance < 0.7*n.distance:
			good.append([m])
			img1_idx = m.queryIdx
			img2_idx = m.trainIdx
			(x1, y1) = keypoints_1[img1_idx].pt
			(x2, y2) = keypoints_2[img2_idx].pt

			A1 = [x1 - W1/2, y1 -H1/2 ]
			A2 =  [x2 -W2/2 , y2 -H2/2] 

			Image1_feature_points.append( [x1 - W1/2, y1 -H1/2 ] )
			Image2_feature_points.append(  [x2 -W2/2 , y2 -H2/2] )
			pts1.append( A1)
			pts2.append( A2  )



	if viz == True:
		img3 = cv2.drawMatchesKnn(Img1,keypoints_1,Img2,keypoints_2,good,None , flags=2 )
		plt.imshow(img3)
		plt.show()

	if len(pts1) == 0 or len(pts2) == 0:
		return np.eye(3), [0,0,1]

	p,T = least_squared_2d_transform(np.asarray(pts1),np.asarray(pts2),[0,0,0])

	R_yaw = np.eye(3) #np.asarray( [ [ T[0][0] ] ])

	R_yaw[0:2, 0:2] = T[0:2, 0:2]

	# SCALE METRIC
	t = [ T[0][2]/10 , T[1][2]/10, 1 ]



	Pt1  =np.ones( ( len(pts1) , 3))
	Pt2  =np.ones( ( len(pts2) , 3))

	Pt1[: , 0:2] =  pts1
	Pt2[: , 0:2] =  pts2

	return R_yaw , t





