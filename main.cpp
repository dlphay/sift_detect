
/******************************************************************************
          *** orb特征点的运动目标检测 ***
******************************************************************************/

#include <iostream> 
#include "opencv2/core/core.hpp"   
#include "opencv2/features2d/features2d.hpp"   
#include "opencv2/highgui/highgui.hpp"   
#include "opencv2/legacy/legacy.hpp"
#include <iostream>   
#include <vector>  
#include <time.h>
// GPU
#include "opencv2/gpu/gpu.hpp"
#include "opencv2/nonfree/gpu.hpp"
//sift


/******************************************************************************
           *** 命名空间 ***
******************************************************************************/

using namespace cv;
using namespace std;
using namespace cv::gpu;

//DLPHAY
//#include "gpu_dlphay.h"

/******************************************************************************
        *** orb检测 ***
******************************************************************************/

#define GRAYSCALE  0

//orb匹配点系数
#define COEFF_DISTANCE 0.63

//累计计算的帧数
#define COUNT_sumNUM  3

//要检测几次？
#define COUNT_jiance_NUM  10

//间隔帧数
#define INTERVAL_IMAGE  9

//是否是第一次的检测   是0   不是1
#define FIRST_IS    0
#define FIRST_NOT    1

/******************************************************************************
       *** 图像本地路径 ***
******************************************************************************/

char *path_image[] = {
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3700.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3701.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3702.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3703.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3704.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3705.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3706.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3707.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3708.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3709.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3710.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3711.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3712.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3713.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3714.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3715.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3716.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3717.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3718.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3719.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3720.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3721.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3722.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3723.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3724.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3725.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3726.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3727.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3728.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3729.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3730.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3731.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3732.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3733.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3734.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3735.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3736.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3737.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3738.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3739.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3740.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3741.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3742.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3743.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3744.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3745.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3746.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3747.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3748.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3749.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3750.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3751.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3752.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3753.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3754.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3755.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3756.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3757.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3758.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3759.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3760.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3761.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3762.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3763.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3764.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3765.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3766.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3767.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3768.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3769.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3770.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3771.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3772.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3773.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3774.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3775.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3776.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3777.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3778.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3779.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3780.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3781.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3782.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3783.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3784.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3785.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3786.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3787.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3788.jpg","E:\\asn_test\\data\\capture\\images5\\A005.mpg3789.jpg",
	"E:\\asn_test\\data\\capture\\images5\\A005.mpg3790.jpg" 
};

char *result_image[] = {
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3700.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3701.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3702.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3703.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3704.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3705.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3706.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3707.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3708.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3709.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3710.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3711.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3712.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3713.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3714.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3715.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3716.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3717.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3718.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3719.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3720.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3721.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3722.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3723.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3724.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3725.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3726.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3727.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3728.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3729.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3730.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3731.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3732.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3733.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3734.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3735.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3736.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3737.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3738.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3739.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3740.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3741.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3742.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3743.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3744.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3745.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3746.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3747.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3748.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3749.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3750.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3751.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3752.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3753.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3754.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3755.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3756.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3757.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3758.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3759.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3760.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3761.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3762.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3763.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3764.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3765.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3766.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3767.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3768.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3769.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3770.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3771.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3772.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3773.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3774.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3775.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3776.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3777.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3778.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3779.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3780.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3781.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3782.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3783.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3784.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3785.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3786.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3787.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3788.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3789.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\A005.mpg3790.jpg"
};

char *sum_image[] = {
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\SUM.mpg3700.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\SUM.mpg3701.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\SUM.mpg3702.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\SUM.mpg3703.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\SUM.mpg3704.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\SUM.mpg3705.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\SUM.mpg3706.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\SUM.mpg3707.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\SUM.mpg3708.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\SUM.mpg3709.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\SUM.mpg3710.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\SUM.mpg3711.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\SUM.mpg3712.jpg" 
};

char *quzao_image[] = {
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\QUZAO.mpg3700.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\QUZAO.mpg3701.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\QUZAO.mpg3702.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\QUZAO.mpg3703.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\QUZAO.mpg3704.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\QUZAO.mpg3705.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\QUZAO.mpg3706.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\QUZAO.mpg3707.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\QUZAO.mpg3708.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\QUZAO.mpg3709.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\QUZAO.mpg3710.jpg","E:\\asn_test\\data\\capture\\images5\\orb\\result\\QUZAO.mpg3711.jpg",
	"E:\\asn_test\\data\\capture\\images5\\orb\\result\\QUZAO.mpg3712.jpg"
};

/******************************************************************************
      *** orb相关变量定义 ***
******************************************************************************/

ORB_GPU orb_gpu;
vector<KeyPoint> KeyPoints_1, KeyPoints_2;
GpuMat Descriptors_1, Descriptors_2;
BruteForceMatcher_GPU<Hamming> matcher;
vector<DMatch> matches;
double max_dist;   //计算距离
double min_dist;
vector<DMatch> good_matches;
std::vector<Point2f> obj;
std::vector<Point2f> scene;

/******************************************************************************
      *** 图像相关变量定义 ***
******************************************************************************/

int I_count = 0;
GpuMat xformed;
GpuMat save;
GpuMat img;
IplImage* ipl_save[10];
IplImage* ipl_img[10];
IplImage* SUM_image;
IplImage* SUM_image1;
IplImage* SUM_TEMP;
IplImage* SUM_image2_quzao;  //去噪
IplImage* SUM_image3_smooth;  //平滑
IplImage* SUM_image4_candy;   //边缘
cv::Mat SUM_image5_blob;      //blob
IplImage* ipl_save_TEMP;  //用于检测临时存放

/******************************************************************************
       *** 去噪尺度计算 ***
******************************************************************************/

int SCALE_NOISE;
int STRIDE_SLIDE_WINDOWS;
IplImage* BGR_image_cpu;
int COUNT;
int red_point_x;
int red_point_y;

/******************************************************************************
       *** 检测坐标点计算变量定义 ***
******************************************************************************/
int n = 0;
CvPoint pt1, pt2;
CvPoint pt3, pt4;
CvPoint pt5, pt6;
CvPoint pt7, pt8;

/******************************************************************************
       *** 其他变量定义 ***
******************************************************************************/

int is_first = FIRST_IS;  //第一次
int x, y, z, t, c;   //循环变量定义

//sift 跟踪
struct feature * fffff;

/******************************************************************************
		*** 链表内存释放 ***
******************************************************************************/

void CLEAR_VECTOR()
{
	KeyPoints_1.clear();
	KeyPoints_2.clear();
	matcher.clear();
	matches.clear();
	good_matches.clear();
	obj.clear();
	scene.clear();
}

/******************************************************************************
       *** gpu初始化 ***
******************************************************************************/

void gpu_init()
{
	int num_devices = gpu::getCudaEnabledDeviceCount();
	//cout << num_devices << endl;
	if (num_devices <= 0)
	{
		std::cerr << "There is no device." << std::endl;
		//return -1;
	}
	int enable_device_id = -1;
	for (int i = 0; i < num_devices; i++)
	{
		cv::gpu::DeviceInfo dev_info(i);
		if (dev_info.isCompatible())
		{
			enable_device_id = i;
		}
	}
	if (enable_device_id < 0)
	{
		std::cerr << "GPU module isn't built for GPU" << std::endl;
		//return -1;
	}
	gpu::setDevice(enable_device_id);
}


/******************************************************************************
       *** MAIN ***
******************************************************************************/

int main()
{
	gpu_init();
	for (I_count = 0; I_count < COUNT_sumNUM; I_count ++ )
	{
		Mat img_cpu_1 = imread(path_image[I_count], GRAYSCALE);
		Mat img_cpu_2 = imread(path_image[I_count + INTERVAL_IMAGE], GRAYSCALE);
		GpuMat img_gpu_1(img_cpu_1);
		GpuMat img_gpu_2(img_cpu_2);

		clock_t start, end;
		start = clock();
		orb_gpu(img_gpu_1, GpuMat(), KeyPoints_1, Descriptors_1);
		orb_gpu(img_gpu_2, GpuMat(), KeyPoints_2, Descriptors_2);
		end = clock();
		cout << "orb_gpu耗时： " << (double)(end - start) / 2 << "ms\n" << endl;

		//const GpuMat mask;
		matcher.match(Descriptors_1, Descriptors_2, matches, GpuMat());

		//-- Quick calculation of max and min distances between keypoints     
		for (int i = 0; i < Descriptors_1.rows; i++)
		{
			double dist = matches[i].distance;
			if (i == 0)
			{
				min_dist = dist;
				max_dist = dist;
			}
			if (dist < min_dist) min_dist = dist;
			if (dist > max_dist) max_dist = dist;
		}
		//printf("-- Max dist : %f \n", max_dist );
		//printf("-- Min dist : %f \n", min_dist );

		for (int i = 0; i < Descriptors_1.rows; i++)
		{
			if (matches[i].distance < COEFF_DISTANCE * max_dist)
			{
				good_matches.push_back(matches[i]);
			}
		}
		cout << good_matches.size() << endl;
		for (size_t i = 0; i < good_matches.size(); ++i)
		{
			// get the keypoints from the good matches
			obj.push_back(KeyPoints_1[good_matches[i].queryIdx].pt);
			scene.push_back(KeyPoints_2[good_matches[i].trainIdx].pt);
		}
		//const CvMat H = findHomography(obj, scene, CV_RANSAC);
		const Mat H_p = findHomography(obj, scene, CV_RANSAC);
		//CvMat H_pp = findHomography(obj, scene, CV_RANSAC);


		IplImage* HHHH_P;
		HHHH_P = &IplImage(H_p);
		for (int i = 0; i < H_p.rows; i++)
		{
			for (int j = 0; j < H_p.cols; j++)
			{
				double Img_pixelVal = cvGetReal2D(HHHH_P, i, j);
				cout << Img_pixelVal << " ";
			}
			cout << "\n" << endl;
		}




		if (NULL != 1)
		{
			gpu::warpPerspective(img_gpu_1, xformed, H_p, img_gpu_1.size(), INTER_LINEAR, BORDER_CONSTANT, cvScalarAll(0), Stream::Null());
			Mat save_cpu;
			Mat img_cpu;
			xformed.download(save_cpu);

			//imwrite("E:\\asn_test\\data\\capture\\images5\\orb\\xformed.bmp", save_cpu);
			gpu::threshold(xformed, img, 0, 255, 0);
			gpu::subtract(img_gpu_2, xformed, save, GpuMat(), 0);

			save.download(save_cpu);
			img.download(img_cpu);

			IplImage ipl_save_d = save_cpu;
			IplImage ipl_img_d = img_cpu;
			ipl_save[I_count] = cvCloneImage(&ipl_save_d);
			ipl_img[I_count] = cvCloneImage(&ipl_img_d);

			/*
			int x, y, c;
			for (y = 0; y<ipl_save[I_count]->height; y++)
				for (x = 0; x< ipl_save[I_count]->width; x++)
				{
					c = ((uchar*)(ipl_img[I_count]->imageData + ipl_img[I_count]->widthStep*y))[x];
					if (c == 0)
						((uchar*)(ipl_save[I_count]->imageData + ipl_save[I_count]->widthStep*y))[x] = (uchar)c;
				}
             */
		}
		is_first = FIRST_IS;
		CLEAR_VECTOR();
	}
	if (!FIRST_IS)
	{
		cvSaveImage("E:\\asn_test\\data\\capture\\images5\\orb\\ipl_save_0.bmp", ipl_save[0]);
		cvSaveImage("E:\\asn_test\\data\\capture\\images5\\orb\\ipl_save_1.bmp", ipl_save[1]);
		cvSaveImage("E:\\asn_test\\data\\capture\\images5\\orb\\ipl_save_2.bmp", ipl_save[2]);
		//cvSaveImage("E:\\asn_test\\data\\capture\\images5\\orb\\ipl_save_3.bmp", ipl_save[3]);
		//cvSaveImage("E:\\asn_test\\data\\capture\\images5\\orb\\ipl_save_4.bmp", ipl_save[4]);
		//cvSaveImage("E:\\asn_test\\data\\capture\\images5\\orb\\ipl_save_5.bmp", ipl_save[5]);
		//cvSaveImage("E:\\asn_test\\data\\capture\\images5\\orb\\ipl_save_6.bmp", ipl_save[6]);
		//cvSaveImage("E:\\asn_test\\data\\capture\\images5\\orb\\ipl_save_7.bmp", ipl_save[7]);
		//cvSaveImage("E:\\asn_test\\data\\capture\\images5\\orb\\ipl_save_8.bmp", ipl_save[8]);
		//cvSaveImage("E:\\asn_test\\data\\capture\\images5\\orb\\ipl_save_9.bmp", ipl_save[9]);

		//SUM 
		SUM_image = cvCreateImage(cvGetSize(ipl_save[0]), IPL_DEPTH_8U, 1);
		for (y = 0; y<SUM_image->height; y++)
			for (x = 0; x<SUM_image->width; x++)
			{
				c = ((uchar*)(ipl_save[0]->imageData + ipl_save[0]->widthStep*y))[x] + ((uchar*)(ipl_save[1]->imageData + ipl_save[1]->widthStep*y))[x]
					+ ((uchar*)(ipl_save[2]->imageData + ipl_save[2]->widthStep*y))[x];

				if (c<255)
					((uchar*)(SUM_image->imageData + SUM_image->widthStep*y))[x] = (uchar)c;
				else
				{
					c = 255;
					((uchar*)(SUM_image->imageData + SUM_image->widthStep*y))[x] = (uchar)c;
				}
			}
		cvSaveImage(sum_image[0], SUM_image);
		SUM_TEMP = SUM_image;
		SUM_image1 = SUM_image;		
		cvMorphologyEx(SUM_image, SUM_image1, SUM_TEMP, NULL, //default 3*3  
			CV_MOP_OPEN, //CV_MOP_CLOSE,
			1);
		SUM_image2_quzao = SUM_image1;
		cvSmooth(SUM_image1, SUM_image2_quzao, CV_GAUSSIAN, 3, 3, 0, 0);
		cvSaveImage("E:\\asn_test\\data\\capture\\images5\\orb\\SUM_image1.bmp", SUM_image1);
		//cvSaveImage("E:\\asn_test\\data\\capture\\images5\\orb\\SUM_image2_quzao.bmp", SUM_image2_quzao);
		//SUM_image2_quzao = SUM_image1;
		if (SUM_image->height > SUM_image->width)
		{
			SCALE_NOISE = (SUM_image->width) / 18;
			STRIDE_SLIDE_WINDOWS = (SUM_image->width) / 18;

		}
		else
		{
			SCALE_NOISE = (SUM_image->height) / 18;
			STRIDE_SLIDE_WINDOWS = (SUM_image->height) / 18;
		}


		//去除干扰因素
		clock_t start, end;
		start = clock();
		for (y = 10; y<SUM_image2_quzao->height - 10; y++)
			for (x = 10; x<SUM_image2_quzao->width - 10; x++)
			{
				c = ((uchar*)(SUM_image2_quzao->imageData + y * SUM_image2_quzao->widthStep))[x];
				if (c>10)
				{
					int COUNT_NUM = 0;

					//去除横线干扰因素！！
					for (z = 0; z<(SUM_image2_quzao->widthStep - x); z++)
					{
						if (((uchar*)(SUM_image2_quzao->imageData + y*SUM_image2_quzao->widthStep))[x + z] > 10)
						{
							COUNT_NUM++;
						}
						else
						{
							if (COUNT_NUM > SCALE_NOISE)
								for (t = 0; t<COUNT_NUM; t++)
									SUM_image2_quzao->imageData[y*SUM_image2_quzao->widthStep + x + t] = 0;
							COUNT_NUM = 0;
							break;
						}
					}

					//去除竖线干扰因素！！
					for (z = 0; z<(SUM_image2_quzao->height - y); z++)
					{
						if (((uchar*)(SUM_image2_quzao->imageData + (y + z)*SUM_image2_quzao->widthStep))[x] > 10)
						{
							COUNT_NUM++;
						}
						else
						{
							if (COUNT_NUM > SCALE_NOISE)
								for (t = 0; t<COUNT_NUM; t++)
									SUM_image2_quzao->imageData[(y + t)*SUM_image2_quzao->widthStep + x] = 0;
							COUNT_NUM = 0;
							break;
						}
					}
				}
			}
		//红点干扰因素！！
		BGR_image_cpu = cvLoadImage(path_image[10], 1);
		//提取 R通道 ！！
		for (y = 0; y < BGR_image_cpu->height; y++)
		{
			for (x = 0; x < BGR_image_cpu->width; x++)
			{
				if (((uchar*)(BGR_image_cpu->imageData + BGR_image_cpu->widthStep*y + x*BGR_image_cpu->nChannels))[0] < 10)
					if (((uchar*)(BGR_image_cpu->imageData + BGR_image_cpu->widthStep*y + x*BGR_image_cpu->nChannels))[1] < 10)
						if (((uchar*)(BGR_image_cpu->imageData + BGR_image_cpu->widthStep*y + x*BGR_image_cpu->nChannels))[2] > 160)
						{
							COUNT++;
							//if(COUNT%10 == 5) printf("(%d, %d)", x,y);
							if (COUNT == 1)
							{
								red_point_x = x;
								red_point_y = y;
								printf("red point ：X : %d   Y : %d  \n", red_point_x, red_point_y);
							}
						}
			}
		}
		printf("red point COUNT : %d  \n", COUNT);
		cvReleaseImage(&BGR_image_cpu);

		if (COUNT > 10)
			for (y = red_point_y - 80; y < red_point_y + 80; y++)
				for (x = red_point_x - 80; x < red_point_x + 80; x++)
				{
					if (((uchar*)(SUM_image2_quzao->imageData + y*SUM_image2_quzao->widthStep))[x] > 100)
						SUM_image2_quzao->imageData[y*SUM_image2_quzao->widthStep + x] = 0;
				}
		end = clock();
		cvSaveImage(quzao_image[0], SUM_image2_quzao);
		fprintf(stderr, "Remove interference time consuming  %d ms\n", (end - start));

		SUM_image3_smooth = SUM_image2_quzao;
		cvSmooth(SUM_image2_quzao, SUM_image3_smooth, CV_GAUSSIAN, 3, 3, 0, 0);
		cvSaveImage("E:\\asn_test\\data\\capture\\images5\\orb\\SUM_image3_smooth.bmp", SUM_image3_smooth);

		//
		for (y = 0; y<(SUM_image3_smooth->height); y++)
			for (x = (SUM_image3_smooth->width - 20); x<(SUM_image3_smooth->width); x++)
			{
				c = ((uchar*)(SUM_image3_smooth->imageData + y * SUM_image3_smooth->widthStep))[x];
				if (c > 0)
				{
					SUM_image3_smooth->imageData[y*SUM_image3_smooth->widthStep + x] = 0;
				}
			}
		for (y = 0; y<(SUM_image3_smooth->height); y++)
			for (x = 0; x<(20); x++)
			{
				c = ((uchar*)(SUM_image3_smooth->imageData + y * SUM_image3_smooth->widthStep))[x];
				if (c > 0)
				{
					SUM_image3_smooth->imageData[y*SUM_image3_smooth->widthStep + x] = 0;
				}
			}
		for (y = (SUM_image3_smooth->height - 10); y<(SUM_image3_smooth->height); y++)
			for (x = 0; x<(SUM_image3_smooth->width); x++)
			{
				c = ((uchar*)(SUM_image3_smooth->imageData + y * SUM_image3_smooth->widthStep))[x];
				if (c > 0)
				{
					SUM_image3_smooth->imageData[y*SUM_image3_smooth->widthStep + x] = 0;
				}
			}
		for (y = 0; y<20; y++)
			for (x = 0; x<(SUM_image3_smooth->width); x++)
			{
				c = ((uchar*)(SUM_image3_smooth->imageData + y * SUM_image3_smooth->widthStep))[x];
				if (c > 0)
				{
					SUM_image3_smooth->imageData[y*SUM_image3_smooth->widthStep + x] = 0;
				}
			}

		cvSaveImage("E:\\asn_test\\data\\capture\\images5\\orb\\SUM_image3_smooth.bmp", SUM_image3_smooth);



		SUM_image4_candy = SUM_image3_smooth;
		//cvCanny(SUM_image3_smooth, SUM_image4_candy, 250, 500, 3);
		cvCanny(SUM_image3_smooth, SUM_image4_candy, 250, 150, 3);

		//BLOB
		//Mat image  = imread(argv[1]);
		vector<KeyPoint> keypoints;
		SimpleBlobDetector::Params params;
		params.filterByArea = true;
		params.minArea = 2;
		params.maxArea = 100;
		IplImage* img_cpu_AA = cvLoadImage(path_image[INTERVAL_IMAGE + COUNT_sumNUM - 1], 1);


		SimpleBlobDetector blobDetector(params);
		blobDetector.create("SimpleBlob");
		blobDetector.detect(SUM_image4_candy, keypoints);
		drawKeypoints(img_cpu_AA, keypoints, SUM_image5_blob, Scalar(0, 0, 255));
		cvSaveImage("E:\\asn_test\\data\\capture\\images5\\orb\\SUM_image4_candy.bmp", SUM_image4_candy);
		cv::imwrite(result_image[2], SUM_image5_blob);


		//画框！！！
		for (y = 20; y<(SUM_image4_candy->height - 10); y++)
			for (x = 20; x<(SUM_image4_candy->width - 10); x++)
			{
				c = ((uchar*)(SUM_image4_candy->imageData + y * SUM_image4_candy->widthStep))[x];
				if (c>0)
				{
					n++;
					if (n == 1)
					{
						pt1.y = pt2.y = y;
						pt1.x = pt2.x = x;
					}
					if (x<pt1.x)
					{
						pt1.x = x;
					}
					if (y<pt1.y)
					{
						pt1.y = y;
					}
					if (x>pt2.x)
					{
						pt2.x = x;
					}
					if (y>pt2.y)
					{
						pt2.y = y;
					}
				}
			}
		pt1.x = pt1.x - 2, pt1.y = pt1.y - 2;
		pt2.x = pt2.x + 2, pt2.y = pt2.y + 2;
		cvRectangle(img_cpu_AA, pt1, pt2, cvScalar(0, 0, 255, 0), 1, 8, 0);
		cvSaveImage("E:\\asn_test\\data\\capture\\images5\\orb\\img_cpu_AA.bmp", img_cpu_AA);
		is_first = FIRST_NOT;
	}

	//jiance
		if (FIRST_NOT)
		{
			for (I_count = COUNT_sumNUM; I_count < (COUNT_sumNUM + COUNT_jiance_NUM); I_count++)
			{
				//开始逐帧检测
				Mat img_cpu_1 = imread(path_image[I_count], GRAYSCALE);
				Mat img_cpu_2 = imread(path_image[I_count + INTERVAL_IMAGE], GRAYSCALE);
				GpuMat img_gpu_1(img_cpu_1);
				GpuMat img_gpu_2(img_cpu_2);

				clock_t start, end;
				start = clock();
				orb_gpu(img_gpu_1, GpuMat(), KeyPoints_1, Descriptors_1);
				orb_gpu(img_gpu_2, GpuMat(), KeyPoints_2, Descriptors_2);
				end = clock();
				cout << " orb_gpu (1300*900)耗时：" << (double)(end - start) / 2 << " ms" << endl;

				//const GpuMat mask;
				matcher.match(Descriptors_1, Descriptors_2, matches, GpuMat());

				//-- Quick calculation of max and min distances between keypoints     
				for (int i = 0; i < Descriptors_1.rows; i++)
				{
					double dist = matches[i].distance;
					if (i == 0)
					{
						min_dist = dist;
						max_dist = dist;
					}
					if (dist < min_dist) min_dist = dist;
					if (dist > max_dist) max_dist = dist;
				}

				for (int i = 0; i < Descriptors_1.rows; i++)
				{
					if (matches[i].distance < COEFF_DISTANCE * max_dist)
					{
						good_matches.push_back(matches[i]);
					}
				}
				cout << good_matches.size() << endl;
				for (size_t i = 0; i < good_matches.size(); ++i)
				{
					// get the keypoints from the good matches
					obj.push_back(KeyPoints_1[good_matches[i].queryIdx].pt);
					scene.push_back(KeyPoints_2[good_matches[i].trainIdx].pt);
				}
				//const CvMat H = findHomography(obj, scene, CV_RANSAC);
				const Mat H_p = findHomography(obj, scene, CV_RANSAC);
				//CvMat H_pp = findHomography(obj, scene, CV_RANSAC);

				if (NULL != 1)
				{
					gpu::warpPerspective(img_gpu_1, xformed, H_p, img_gpu_1.size(), INTER_LINEAR, BORDER_CONSTANT, cvScalarAll(0), Stream::Null());
					Mat save_cpu;
					Mat img_cpu;
					xformed.download(save_cpu);

					//imwrite("E:\\asn_test\\data\\capture\\images5\\orb\\xformed.bmp", save_cpu);
					gpu::threshold(xformed, img, 0, 255, 0);
					gpu::subtract(img_gpu_2, xformed, save, GpuMat(), 0);

					save.download(save_cpu);
					img.download(img_cpu);

					IplImage ipl_save_d = save_cpu;
					IplImage ipl_img_d = img_cpu;
					ipl_save_TEMP = cvCloneImage(&ipl_save_d);
					ipl_img[I_count] = cvCloneImage(&ipl_img_d);
				}
				CLEAR_VECTOR();

				


				//累计检测
				ipl_save[0] = ipl_save[1];
				ipl_save[1] = ipl_save[2];
				ipl_save[2] = ipl_save_TEMP;

				//SUM 
				SUM_image = cvCreateImage(cvGetSize(ipl_save[0]), IPL_DEPTH_8U, 1);
				for (y = 0; y<SUM_image->height; y++)
					for (x = 0; x<SUM_image->width; x++)
					{
						c = ((uchar*)(ipl_save[0]->imageData + ipl_save[0]->widthStep*y))[x] + ((uchar*)(ipl_save[1]->imageData + ipl_save[1]->widthStep*y))[x]
							+ ((uchar*)(ipl_save[2]->imageData + ipl_save[2]->widthStep*y))[x];

						if (c<255)
							((uchar*)(SUM_image->imageData + SUM_image->widthStep*y))[x] = (uchar)c;
						else
						{
							c = 255;
							((uchar*)(SUM_image->imageData + SUM_image->widthStep*y))[x] = (uchar)c;
						}
					}
				cvSaveImage(sum_image[I_count-2], SUM_image);

				SUM_TEMP = SUM_image;
				SUM_image1 = SUM_image;
				cvMorphologyEx(SUM_image, SUM_image1, SUM_TEMP, NULL, //default 3*3  
					CV_MOP_OPEN, //CV_MOP_CLOSE,
					1);
				SUM_image2_quzao = SUM_image1;
				cvSmooth(SUM_image1, SUM_image2_quzao, CV_GAUSSIAN, 3, 3, 0, 0);
				//cvSaveImage("E:\\asn_test\\data\\capture\\images5\\orb\\SUM_image2_quzao.bmp", SUM_image2_quzao);
				//SUM_image2_quzao = SUM_image1;
				if (SUM_image->height > SUM_image->width)
				{
					SCALE_NOISE = (SUM_image->width) / 18;
					STRIDE_SLIDE_WINDOWS = (SUM_image->width) / 18;

				}
				else
				{
					SCALE_NOISE = (SUM_image->height) / 18;
					STRIDE_SLIDE_WINDOWS = (SUM_image->height) / 18;
				}


				//去除干扰因素
				//clock_t start, end;
				start = clock();
				for (y = 10; y<SUM_image2_quzao->height - 10; y++)
					for (x = 10; x<SUM_image2_quzao->width - 10; x++)
					{
						c = ((uchar*)(SUM_image2_quzao->imageData + y * SUM_image2_quzao->widthStep))[x];
						if (c>10)
						{
							int COUNT_NUM = 0;

							//去除横线干扰因素！！
							for (z = 0; z<(SUM_image2_quzao->widthStep - x); z++)
							{
								if (((uchar*)(SUM_image2_quzao->imageData + y*SUM_image2_quzao->widthStep))[x + z] > 10)
								{
									COUNT_NUM++;
								}
								else
								{
									if (COUNT_NUM > SCALE_NOISE)
										for (t = 0; t<COUNT_NUM; t++)
											SUM_image2_quzao->imageData[y*SUM_image2_quzao->widthStep + x + t] = 0;
									COUNT_NUM = 0;
									break;
								}
							}

							//去除竖线干扰因素！！
							for (z = 0; z<(SUM_image2_quzao->height - y); z++)
							{
								if (((uchar*)(SUM_image2_quzao->imageData + (y + z)*SUM_image2_quzao->widthStep))[x] > 10)
								{
									COUNT_NUM++;
								}
								else
								{
									if (COUNT_NUM > SCALE_NOISE)
										for (t = 0; t<COUNT_NUM; t++)
											SUM_image2_quzao->imageData[(y + t)*SUM_image2_quzao->widthStep + x] = 0;
									COUNT_NUM = 0;
									break;
								}
							}
						}
					}
				//红点干扰因素！！
				BGR_image_cpu = cvLoadImage(path_image[10], 1);
				//提取 R通道 ！！
				for (y = 0; y < BGR_image_cpu->height; y++)
				{
					for (x = 0; x < BGR_image_cpu->width; x++)
					{
						if (((uchar*)(BGR_image_cpu->imageData + BGR_image_cpu->widthStep*y + x*BGR_image_cpu->nChannels))[0] < 10)
							if (((uchar*)(BGR_image_cpu->imageData + BGR_image_cpu->widthStep*y + x*BGR_image_cpu->nChannels))[1] < 10)
								if (((uchar*)(BGR_image_cpu->imageData + BGR_image_cpu->widthStep*y + x*BGR_image_cpu->nChannels))[2] > 160)
								{
									COUNT++;
									//if(COUNT%10 == 5) printf("(%d, %d)", x,y);
									if (COUNT == 1)
									{
										red_point_x = x;
										red_point_y = y;
										printf("red point ：X : %d   Y : %d  \n", red_point_x, red_point_y);
									}
								}
					}
				}
				printf("red point COUNT : %d  \n", COUNT);
				cvReleaseImage(&BGR_image_cpu);

				if (COUNT > 10)
					for (y = red_point_y - 80; y < red_point_y + 80; y++)
						for (x = red_point_x - 80; x < red_point_x + 80; x++)
						{
							if (((uchar*)(SUM_image2_quzao->imageData + y*SUM_image2_quzao->widthStep))[x] > 100)
								SUM_image2_quzao->imageData[y*SUM_image2_quzao->widthStep + x] = 0;
						}
				end = clock();
				fprintf(stderr, "Remove interference time consuming  %d ms\n", (end - start));
				cvSaveImage(quzao_image[I_count - 2], SUM_image2_quzao);


				SUM_image3_smooth = SUM_image2_quzao;
				cvSmooth(SUM_image2_quzao, SUM_image3_smooth, CV_GAUSSIAN, 3, 3, 0, 0);


				//
				for (y = 0; y<(SUM_image3_smooth->height); y++)
					for (x = (SUM_image3_smooth->width - 20); x<(SUM_image3_smooth->width); x++)
					{
						c = ((uchar*)(SUM_image3_smooth->imageData + y * SUM_image3_smooth->widthStep))[x];
						if (c > 0)
						{
							SUM_image3_smooth->imageData[y*SUM_image3_smooth->widthStep + x] = 0;
						}
					}
				for (y = 0; y<(SUM_image3_smooth->height); y++)
					for (x = 0; x<(20); x++)
					{
						c = ((uchar*)(SUM_image3_smooth->imageData + y * SUM_image3_smooth->widthStep))[x];
						if (c > 0)
						{
							SUM_image3_smooth->imageData[y*SUM_image3_smooth->widthStep + x] = 0;
						}
					}
				for (y = (SUM_image3_smooth->height - 10); y<(SUM_image3_smooth->height); y++)
					for (x = 0; x<(SUM_image3_smooth->width); x++)
					{
						c = ((uchar*)(SUM_image3_smooth->imageData + y * SUM_image3_smooth->widthStep))[x];
						if (c > 0)
						{
							SUM_image3_smooth->imageData[y*SUM_image3_smooth->widthStep + x] = 0;
						}
					}
				for (y = 0; y<20; y++)
					for (x = 0; x<(SUM_image3_smooth->width); x++)
					{
						c = ((uchar*)(SUM_image3_smooth->imageData + y * SUM_image3_smooth->widthStep))[x];
						if (c > 0)
						{
							SUM_image3_smooth->imageData[y*SUM_image3_smooth->widthStep + x] = 0;
						}
					}


				SUM_image4_candy = SUM_image3_smooth;
				//cvCanny(SUM_image3_smooth, SUM_image4_candy, 250, 500, 3);
				cvCanny(SUM_image3_smooth, SUM_image4_candy, 250, 150, 3);

				//BLOB
				//Mat image  = imread(argv[1]);
				vector<KeyPoint> keypoints;
				SimpleBlobDetector::Params params;
				params.filterByArea = true;
				params.minArea = 1;
				params.maxArea = 10;
				params.minThreshold = 125;
				params.maxThreshold = 255;
				IplImage* img_cpu_AA = cvLoadImage(path_image[INTERVAL_IMAGE + I_count - 1], 1);


				SimpleBlobDetector blobDetector(params);
				blobDetector.create("SimpleBlob");
				blobDetector.detect(SUM_image4_candy, keypoints);
				drawKeypoints(img_cpu_AA, keypoints, SUM_image5_blob, Scalar(0, 0, 255));
				cv::imwrite(result_image[I_count], SUM_image5_blob);



			}
		}
	


	return 0;
}










//test

/*
Mat img_matches;
drawMatches(img_cpu_1, KeyPoints_1, img_cpu_2, KeyPoints_2,
good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
*/

/*
if (1)
{
// get the corners from the image_1
std::vector<Point2f> obj_corners(4);
obj_corners[0] = cvPoint(0, 0);
obj_corners[1] = cvPoint(img_cpu_1.cols, 0);
obj_corners[2] = cvPoint(img_cpu_1.cols, img_cpu_1.rows);
obj_corners[3] = cvPoint(0, img_cpu_1.rows);
std::vector<Point2f> scene_corners(4);

perspectiveTransform(obj_corners, scene_corners, H_p);

// draw lines between the corners (the mapped object in the scene - image_2)
line(img_matches, scene_corners[0] + Point2f(img_cpu_1.cols, 0), scene_corners[1] + Point2f(img_cpu_1.cols, 0), Scalar(255, 0, 0));
line(img_matches, scene_corners[1] + Point2f(img_cpu_1.cols, 0), scene_corners[2] + Point2f(img_cpu_1.cols, 0), Scalar(255, 0, 0));
line(img_matches, scene_corners[2] + Point2f(img_cpu_1.cols, 0), scene_corners[3] + Point2f(img_cpu_1.cols, 0), Scalar(255, 0, 0));
line(img_matches, scene_corners[3] + Point2f(img_cpu_1.cols, 0), scene_corners[0] + Point2f(img_cpu_1.cols, 0), Scalar(255, 0, 0));

}

//imwrite("E:\\asn_test\\data\\capture\\images5\\orb\\img_matches.bmp", img_matches);
*/






