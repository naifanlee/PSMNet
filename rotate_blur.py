import os, cv2, sys
import numpy as np
import math


def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

def remap(intrinsic_file, img, theta ):
    if os.path.isfile(intrinsic_file):
        fp = open(intrinsic_file, 'r')
        param_list = [l.split() for l in fp.readlines()]
        fx = float(param_list[0][1])
        fy = float(param_list[0][6])
        cx = float(param_list[0][3])
        cy = float(param_list[0][7])

        print ('fx = ', fx)
        print ('fy = ', fy)
        print ('cx = ', cx)
        print ('cy = ', cy)

        camera_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]], dtype=np.float32)

        camera_matrix_out = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]], dtype=np.float32)

        distortion = np.zeros(5)

        img_height = img.shape[0]
        img_width = img.shape[1]

        rotation = eulerAnglesToRotationMatrix(theta)

        map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion, rotation, camera_matrix_out,
                                                 (img_width, img_height), cv2.CV_32FC1)

        return map1, map2
        # pitch =  * math.pi/180.0
        # roll =
        # rotation = np.array([[ 1.0,0,0],[0,math.cos(pitch), -math.sin(pitch)],[0, math.sin(pitch), math.cos(pitch)]], dtype=np.float32)


def remap(fx, fy, cx, cy, img, theta ):

    print ('fx = ', fx)
    print ('fy = ', fy)
    print ('cx = ', cx)
    print ('cy = ', cy)

    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)

    camera_matrix_out = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)

    distortion = np.zeros(5)

    img_height = img.shape[0]
    img_width = img.shape[1]

    rotation = eulerAnglesToRotationMatrix(theta)

    map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, distortion, rotation, camera_matrix_out,
                                             (img_width, img_height), cv2.CV_32FC1)

    return map1, map2

def rectify(img, map1, map2):
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

def main():

    datapath = sys.argv[1]
    imgL_path = os.path.join(datapath, 'left')
    imgR_path = os.path.join(datapath, 'right')
    imgL_remap_path = os.path.join(datapath, 'left_remap')
    imgR_remap_path = os.path.join(datapath, 'right_remap')
    imgL_blur_path = os.path.join(datapath, 'left_blur')
    imgR_blur_path = os.path.join(datapath, 'right_blur')

    for subpath in [imgL_remap_path, imgR_remap_path, imgL_blur_path, imgR_blur_path]:
        os.system('rm -rf {}'.format(subpath))
        os.mkdir(subpath)

    for fn in sorted(os.listdir(imgL_path))[:20]:
        imgL_fpath = os.path.join(imgL_path, fn)
        imgR_fpath = os.path.join(imgR_path, fn)
        for img_fpath in [imgL_fpath, imgR_fpath]:
            print(img_fpath)
            img = cv2.imread(img_fpath)
    
            fx = 707.0493
            fy = 707.0493
            cx = 604.0814
            cy = 180.5066

            angle_x = 0.5
            angle_y = 0
            angle_z = 0

            kx_blur = 5
            sigma_x = 5

            angle_x *= math.pi/180.0
            angle_y *= math.pi/180.0
            angle_z *= math.pi/180.0



            map1, map2 = remap(fx, fy, cx, cy, img, [angle_x, angle_y, angle_z])

            dst = rectify(img, map1, map2)

            dst_blur = cv2.GaussianBlur(img, (kx_blur, kx_blur), sigmaX = sigma_x)

            if img_fpath == imgL_fpath:
                img_remap_fpath = os.path.join(imgL_remap_path, fn)
                img_blur_fpath = os.path.join(imgL_blur_path, fn)
                cv2.imwrite(img_remap_fpath, dst)
                cv2.imwrite(img_blur_fpath, dst_blur)
            else:
                img_remap_fpath = os.path.join(imgR_remap_path, fn)
                img_blur_fpath = os.path.join(imgR_blur_path, fn)
                cv2.imwrite(img_remap_fpath, img)
                cv2.imwrite(img_blur_fpath, img)

            

main()

