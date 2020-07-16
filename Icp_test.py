# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 16:23:58 2020

@author: mille
"""


import numpy as np
import matplotlib.pyplot as plt
from math import ceil, cos, radians 
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
try: 
    import torch
    from sklearn.neighbors import NearestNeighbors
except:
    print("Importado o no instalado")
import os
import seaborn as sn

path_txt = os.path.join('E:\Descargas','datos.txt')
data     = pd.read_csv(path_txt,sep='\t',header=None)
i=data.fillna(0)
data_size=data.shape[0]
data1= np.array(i)

print (data.shape)
N=640
XYZINrRSaU_1=np.zeros([data_size*N,9])
XYZINrRSaU_2=np.zeros([data_size*N,9])
XYZINrRSaU_3=np.zeros([data_size*N,9])
PI=np.pi

XYZ=np.zeros([1,3])
Fs=30
Ts=1/Fs
eps=1e-9


# def gen(data,echo):
#     hokuyo_start=7
#     Number_of_returns=data.iloc[:,hokuyo_start+N*6:hokuyo_start+N*7]
#     Number_of_returns=Number_of_returns.fillna(-1)
#     Number_of_returns=np.array(Number_of_returns.values)
#     scan_angle=data.iloc[:,hokuyo_start+N*7:hokuyo_start+N*8]
#     scan_angle=np.array(scan_angle.values)
#     if echo==1:
#         print("1th echo")
#         Range       = data.iloc[:,hokuyo_start:hokuyo_start+N]
#     elif echo==2:
#         print(".........2th echo")
#         Range           = data.iloc[:,hokuyo_start+N:hokuyo_start+N*2]
#     elif echo==3:
#         print("..................3th echo")
#         Range       = data.iloc[:,hokuyo_start+N*2:hokuyo_start+N*3]
#     else:
#         print("Error")
#     Range=Range.replace(60,100)
#     Range=Range.replace(0,100)
#     Range=np.array(Range)
#     #np.nan_to_num(0)
#     #Range=np.nan_to_num(Range)
#     return (Range)

def proceso(R):
    global XYZ
    samples=R.shape[0]*R.shape[1]
    X=R*np.cos(beta)
    Y=R*np.sin(beta)
    input_matrix_k = np.zeros([4,N])
    input_matrix_k_1 = np.zeros([4,N])
    L=R.shape[0]  #Total de escaneos
    L=500      #Final forzado
    n=1           # Intervalo
    inicio=3    # escaneo de incio
    alpha=45*PI/180.0 # Angulo dynamixel
    Tx1  = 0
    Ty1  =-0.04980
    Tz1  = 0.11350
    # Matriz traslacion:
    T              = np.array([[1,0,0,Tx1],[0,1,0,Ty1],[0,0,1,Tz1],[0,0,0,1]])
    Rot            = np.array([[np.cos(alpha), 0, np.sin(alpha), 0], [0,1,0,0], [-np.sin(alpha), 0, np.cos(alpha), 0], [0,0,0,1]])
    ## ETEST
    T2             = np.array([[1,0,0,1],[0,1,0,2],[0,0,1,3],[0,0,0,1]])
    append_matrix  = np.empty([3,N])
    delta_x=[0]
    delta_y=[0]
    delta_z=[0]
    for k in range(inicio,L,n):
        input_matrix_k[0,:] = X[k,:]
        input_matrix_k[1,:] = Y[k,:]
        input_matrix_k[3,:] = np.ones([1,N])
        P2_k=np.dot(T,input_matrix_k)
        P3_k=np.dot(Rot,P2_k)
        #P3_k=torch.matmul(T2,P3_k)
        #
        input_matrix_k_1[0,:] = X[k-1,:]
        input_matrix_k_1[1,:] = Y[k-1,:]
        input_matrix_k_1[3,:] = np.ones([1,N])
        P2_k_1=np.dot(T,input_matrix_k_1)
        P3_k_1=np.dot(Rot,P2_k_1)
        T_odom,yaw,pitch,roll=ICP(P3_k,P3_k_1)
        delta_x.append(float(T_odom[0]))
        delta_y.append(float(T_odom[1]))
        delta_z.append(float(T_odom[2]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(P3_k[0,:],P3_k[1,:],P3_k[2,:],s=1)
    ax.scatter(P3_k_1[0,:],P3_k_1[1,:],P3_k_1[2,:],s=1)
    #ax.scatter(P3_k[0,pk],P3_k[1,pk],P3_k[2,pk],s=10)
    #ax.scatter(P3_k_1[0,pk_1],P3_k_1[1,pk_1],P3_k_1[2,pk_1],s=10)
    ax.view_init(elev=0, azim=-90)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    return P3_k, np.array(delta_x), np.array(delta_y), np.array(delta_z)



def get_geom_feat(X,Y,Z,P):
    mean_x = np.mean(X)
    mean_y = np.mean(Y)
    mean_z = np.mean(Z)
    X = X.T #- mean_x
    Y = Y.T #- mean_y
    Z = Z.T #- mean_z
    cov_mat = np.cov([X,Y,Z])
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
    eig_val = np.sort(eig_val_cov)
    E1 = eig_val[2]
    E2 = eig_val[1]
    E3 = eig_val[0]
    # Anistropy, planarity, sphericity, Lniearity, curvature....
    # https://www.isprs-ann-photogramm-remote-sens-spatial-inf-sci.net/II-3/9/2014/isprsannals-II-3-9-2014.pdf
    e1=E1/(E1+E2+E3+eps)
    e2=E2/(E1+E2+E3+eps)
    e3=E3/(E1+E2+E3+eps)
    if(e1>0 and e2>0 and e3>0):
        p0 = 1.0*(e1-e2)/(e1 + eps)  # Linearity
        p1 = 1.0*(e2-e3)/(e1 + eps)  # Planarity
        p2 = 1.0*e3/(e1 + eps)      # Sphericity
        p3 = 1.0*pow(e1*e3*e3,1/3.0) # Omnivariance
        p4 = 1.0*(e1-e3)/(e1 + eps)  # Anisotropy
        p5 = -1.0*( e3*np.log(e3) + e2*np.log(e2) + e1*np.log(e1) ) #Eigenentropy
        p6 = 1.0*(e1 +e3 + e3)         # sumatory
        p7 = 1.0*e3/(e1+e2+e3 + eps) #  change of curvature
        # Calculate of angles
        mp = np.argmin(eig_val_cov)
        nx, ny, nz  = eig_vec_cov[:,mp] # https://mediatum.ub.tum.de/doc/800632/800632.pdf
        phi =abs(np.arctan(nz/(ny+eps)))
        theta =abs(np.arctan((pow(nz,2)+pow(ny,2))/nx))
    else:
        print(X,Y,Z)
        print(cov_mat)
        print(P)
        p0 = float("NaN")
        p1 = float("NaN")
        p2 = float("NaN")
        p3 = float("NaN")
        p4 = float("NaN")
        p5 = float("NaN")
        p6 = float("NaN")
        p7 = float("NaN")
        # Calculate of angles
        mp = float("NaN")
        phi =float("NaN")
        theta =float("NaN")
    descriptors=np.array([p0,p1,p2,p3,p4,p5,p6,p7,phi,theta])
    #print(descriptors)
    return descriptors


def rotation_matrix(axis, theta):
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],[2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],[2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])


def format_transformation_matrix (T):
    return ["{},{}".format(i, j) for i, j in T]


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''
    assert A.shape == B.shape
    # get number of dimensions
    m = A.shape[1]
    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)
    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t
    return T, R, t


def nearest_neighbor(src, dst):
    '''Find the nearest (Euclidean) neighbor in dst for each point in src
    
    Parameters
    ----------
        src: Pxm array of points
        dst: Qxm array of points
    Returns
    -------
        distances: P-elements array of Euclidean distances of the nearest neighbor
        indices: P-elements array of dst indices of the nearest neighbor
    '''
    # Asserting src and dst have the same dimension. They can have different point count.
    assert src.shape[1] == dst.shape[1]

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def filter_points_by_angle (A, angle_in_degrees):
    """Reduce the number of items in a point cloud by measuring the poitional vector of each point to the Z axis
    Parameters
    ----------
    A : numpy array
        Pxm array for a list with P items of m-Dimensioned Points
    angle_in_degrees : float
        points located within this angle to the Z-Axis is kept 
    Returns
    -------
    numpy array
        Qxm array a list of m-Dimensioned Points, item count reduced from P to Q, where Q = ceil(P/everyNth)
    """
    indices = []
    angle = cos(radians(angle_in_degrees))
    for i in range(len(A)):
        if (np.dot([0,0,1],A[i])/np.linalg.norm(A[i]) > angle):
            indices.append(i)
    #This is a very inefficient implementation, someone kwno knows Numpy better should have a better way of dealing with this,
    return A[indices]


def decimate_by_sequence (A, everyNth = 2):
    """Reduce the number of points in a point cloud
    Parameters
    ----------
    A : numpy array 
        Pxm array for a list with P items of m-Dimensioned Points
    everyNth : int, optional
        Return one point for every x Number of points  (the default is 2, which is to return every other point)
    Returns
    -------
    numpy array
        Qxm array a list of m-Dimensioned Points, item count reduced from P to Q, where Q = ceil(P/everyNth)
    """
    return A[range(0,len(A),everyNth)]


def icp(A, B, standard_deviation_range = 0.0, init_pose=None, max_iterations=100, convergence=0.001, quickconverge=1, filename="results.txt"):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Pxm numpy array of source m-Dimensioned points
        B: Qxm numpy array of destination m-Dimensioned point
        standard_deviation_range: If this value is not zero, the outliers (defined by the given standard Deviation) will be ignored.
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        convergence: convergence criteria
        quickconverge: streategy to overapply transformation at each iteration. Value of 2 means two transforamtions are made.
        filename: fileName to save the In-Progress data for preview
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    # Asserting A and B have the same dimension. They can have different point count.
    assert A.shape[1] == B.shape[1]
    # get number of dimensions
    m = A.shape[1]
    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0
    delta_error = 0

    f= open(filename,"w+")
    if standard_deviation_range > 0.0:
        f.write("Outlier purge active, standard deviation range: %f\n", standard_deviation_range)
        #logging.debug("Outlier purge active, standard deviation range: %f", standard_deviation_range)
    else:
        f.write("Outlier purge not active.\n")
        #logging.debug("Outlier purge not active.")

    for i in range(max_iterations):
        f.write("ITERATION:%d\n"%i)
        #logging.debug("Iteration: %d "%i)
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)
        #Compute Mean Error and Standard Deviation
        mean_error = np.mean(distances)
        stde_error = np.std(distances)
        #Ignore distances that are outlers
        if standard_deviation_range > 0.0:
            trimmed_dst_indices = []
            trimmed_src_indices = []
            for j in range(len(indices)):
                if distances[j] > (mean_error + standard_deviation_range * stde_error):
                    continue
                if distances[j] < (mean_error - standard_deviation_range * stde_error):
                    continue
                trimmed_dst_indices.append(indices[j])
                trimmed_src_indices.append(j)
            # compute the transformation between the selected values in source and nearest destination points
            T,R,t = best_fit_transform(src[:m,trimmed_src_indices].T, dst[:m,trimmed_dst_indices].T)
            #Recomputing Mean Error based on selected distances
            mean_error = np.mean(distances[trimmed_src_indices])
        else:
            # compute the transformation between the source and nearest destination points
            T,R,t = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        #Transform the current source
        if quickconverge > 1.0:
            if i > 2:
                if delta_error > 0.0:
                    quickconverge = quickconverge - 0.4
                    #logging.debug("quickconverge: %.2f " % quickconverge)

        f.write(" QUICK_CONVERGE: %.1f" % quickconverge)
        f.write("\n")
        for p in range(int(ceil(quickconverge))):
            src = np.dot(T, src)
        #Log the current transformation matrix
        currentT,currentR,current_t = best_fit_transform(A, src[:m,:].T)
        f.write(" TRANSFORMATION:")
        f.write(np.array2string(currentT.flatten(),formatter={'float':lambda x: "%.8f" % x},separator=",").replace('\n', ''))
        f.write("\n")

        # logging.debug(np.array2string(T.flatten(),separator=","))

        # check error to see if convergence cretia is met
        delta_error = (mean_error - prev_error)
        f.write(" MEAN_ERR:%.8f\n" % mean_error)
        f.write(" STD_DEV:%.8f\n" % stde_error)
        f.write(" DELTA_ERROR:%.8f\n" % delta_error)

        if np.abs(delta_error) < convergence:
            break
        prev_error = mean_error
    f.close() 
    # calculate final transformation
    T,R,t = best_fit_transform(A, src[:m,:].T)
    return T, R, t, distances, i   

def R2abg(R):
    alpha=np.arctan(R[1,0]/R[0,0])
    beta =np.arctan(-R[2,0]/np.sqrt( np.power(R[2,1],2)+np.power(R[2,2],2)  ) )
    gamma=np.arctan(R[2,1]/R[2,2])
    return alpha, beta, gamma

def path_plot(delta_x,delta_y,delta_z,T):
    L=T.shape[2]
    X=np.zeros([1,L])
    Y=np.zeros([1,L])
    Z=np.zeros([1,L])
    delta_X=[0]
    delta_Y=[0]
    delta_Z=[0]
    pose=np.array([[0],[0],[0],[1]])
    for k in range(0,L):
        pose=np.dot(T[:,:,k],pose)
        delta_X.append(float(pose[0]))
        delta_Y.append(float(pose[1]))
        delta_Z.append(float(pose[2]))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(delta_X,delta_Y,delta_Z,s=1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

if __name__ == "__main__":
    N=641
    beta = np.linspace(-29.883638317,30.038820215,N)*-PI/180.0
    # R1=gen(data,1)
    # R2=gen(data,2)
    # R3=gen(data,3)
    # R=R1+R2+R3
    # print (R1.shape)
    # print (R2.shape)
    # print (R3.shape)
    ##
    samples=data1.shape[0]*data1.shape[1]
    X=data1*np.cos(beta)
    Y=data1*np.sin(beta)
    print (X.shape)
    input_matrix_k   = np.zeros([4,N])
    input_matrix_k_1 = np.zeros([4,N])
    L=data1.shape[0]  # Total de escaneos
    # L=100         # Final forzado
    n=1           # Intervalo
    inicio=1      # escaneo de incio
    alpha=0# Angulo dynamixel
    Tx1  = 0
    Ty1  =-0.04980
    Tz1  = 0.11350
    # Matriz traslacion:
    T              = np.array([[1,0,0,Tx1],[0,1,0,Ty1],[0,0,1,Tz1],[0,0,0,1]])
    Rot            = np.array([[np.cos(alpha), 0, np.sin(alpha), 0], [0,1,0,0], [-np.sin(alpha), 0, np.cos(alpha), 0], [0,0,0,1]])
    ## ETEST
    append_matrix  = np.empty([3,N])
    delta_x=[0]
    delta_y=[0]
    delta_z=[0]
    delta_yaw=[0]
    delta_pitch=[0]
    delta_roll=[0]
    mode=1
    inc=1
    Transformada=np.eye(4)
    new_dst=np.zeros([641,4])
    Trs=np.zeros([4,4,L])
    sumax=np.array([0])
    Xs=[0]
    Ys=[0]
    for k in range(n,L,1):
        N=641 
        # v=1
        input_matrix_k[0,:] = X[k,:]
        input_matrix_k[1,:] = Y[k,:]
        input_matrix_k[3,:] = np.ones([1,N])
        #
        #
        P2_k=np.dot(T,input_matrix_k)
        P3_k=np.dot(Rot,P2_k)
        #
        input_matrix_k_1[0,:] = X[k-inc,:]
        input_matrix_k_1[1,:] = Y[k-inc,:]
        input_matrix_k_1[3,:] = np.ones([1,N])
        P2_k_1=np.dot(T,input_matrix_k_1)
        P3_k_1=np.dot(Rot,P2_k_1)
        # if mode == 0:
            # fig = plt.figure()
            # 1z3 = np.zeros(461)
# Agrrgamos un plano 3D
            # ax = fig.add_subplot(111,projection='3d')
            # fig = plt.figure()
            # # ax= fig.add_subplot()
            # # ax.scatter(X, Y )
            # # ax.set_xlabel('X Label')
            # # ax.set_ylabel('Y Label')
            # # ax.set_zlabel('Z Label')
            # plt.plot(X,Y, P3_k[2,:])
            # plt.show()
            # ax.scatter(X, Y, z3)

# Mostramos el gráfico
            # plt.show()
        if mode==0:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            #ax.scatter(P3_k[0,:],P3_k[1,:],P3_k[2,:],'o',s=1)
            ax.scatter(P3_k_1[0,:],P3_k_1[1,:],P3_k_1[2,:],'+',s=1)
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
        #src=np.dot(Transformada,P3_k_1[0:4,:])
        # print (P3_k_1[2,:])
        src=P3_k_1[0:3,:].T
        dst=P3_k[0:3,:].T
        Transformada,R,t,distances, iterations = icp(src, dst, standard_deviation_range = 0.0, init_pose=None, max_iterations=100, convergence=0.000001, quickconverge=1, filename="results.txt")
        Trs[:,:,k-1]=Transformada
        yaw,pitch,roll=R2abg(R)
        delta_x.append(t[0])
        delta_y.append(t[1])
        delta_z.append(t[2]) 
        delta_pitch.append(pitch)
        delta_roll.append(roll)
        delta_yaw.append(yaw)
        z= np.zeros(10)
        # print (distances)
        # print (delta_yaw)
        
        print (k)
        sumx= sum(delta_x)
        Xs.append(sumx)  
        # print (Xs)
        # print (sumx)
        sumy= sum(delta_y)
        Ys.append(sumy)  
        # print (Ys)
        # print (sumy)
    print("FINISH")
    path_plot(delta_x,delta_y,z,Trs)
    fig = plt.figure()
    ax= fig.add_subplot()
    ax.scatter(Xs, Ys )
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    plt.show()
   
