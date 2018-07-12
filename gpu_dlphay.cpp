#include <iostream>   
#include "opencv2/gpu/gpu.hpp"

using namespace cv;
using namespace std;
using namespace cv::gpu;

int gpu_init()
{
	int num_devices = gpu::getCudaEnabledDeviceCount();
	//cout << num_devices << endl;
	if (num_devices <= 0)
	{
		std::cerr << "There is no device." << std::endl;
		return -1;
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
		return -1;
	}
	gpu::setDevice(enable_device_id);
}