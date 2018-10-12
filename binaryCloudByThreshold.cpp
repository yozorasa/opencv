#include <windows.h>
#include <opencv2/opencv.hpp>  
using namespace cv;
#include<iostream>
#include<fstream>
using namespace std;
#include <stdlib.h>
#include <string>

String loadImg = "cloud/0.jpg";
String saveImg = "cloud/0_b.jpg";
String saveImgROI = "cloud/0_broi.jpg";
String loadPos = "cloud/predictions.txt";
int bgrAbsThreshold = 15;
int bgrAvgThreshold = 160;
int greyThreshold = 160;

Mat cloudThreshold(const Mat& image) {
	Mat labels;//聚類後的標籤
			   //assert(image.channels() != 1);
	int rows = image.rows;
	int cols = image.cols;
	int channels = image.channels();

	//保存聚類後的圖
	Mat clusteredMat(rows, cols, CV_8UC3);
	clusteredMat.setTo(Scalar::all(0));

	//二值化 雲(前景) 非雲(背景)
	clusteredMat = image.clone();
	Scalar paintOn;
	//Color Mode
	for (int i = 0; i < rows; ++i)
	{
		const uchar *idata = image.ptr<uchar>(i);
		double bgrAvg;
		int bgrDisFlag;
		int cloudFlag = 0;
		for (int j = 0; j < cols*channels; j += channels)
		{
			bgrAvg = 0;
			bgrDisFlag = 0;
			cloudFlag = 0;
			for (int k = 0; k < 3; k++)
			{
				bgrAvg += idata[k + j];
			}
			bgrAvg /= 3;

			//計算B G R相差多少(色彩飽和度)
			for (int k = 0; k < 3; k++)
			{
				if (abs(bgrAvg - idata[k + j]) > bgrAbsThreshold)
				{
					//cout << "Dis = " << abs(bgrAvg[i] - clusterCenter[j]) << endl;
					bgrDisFlag = 1;
				}
			}

			//亮度高 飽和度低 判斷為雲
			if (bgrDisFlag == 0 && bgrAvg >= bgrAvgThreshold)
			{
				cloudFlag = 1;
			}

			if (cloudFlag != 1)
			{
				paintOn = Scalar(0, 0, 0);
				circle(clusteredMat, Point(j / channels, i), 1, paintOn); //非雲(背景)為黑
			}
		}
	}

	return clusteredMat;
}

Mat copyROI(const Mat& image, Mat& allROI, int pos[])
{
	int rows = image.rows;
	int cols = image.cols;
	int channels = image.channels();

	int left = pos[0];
	int right = pos[1];
	int top = pos[2];
	int bot = pos[3];

	for (int y = top; y < bot; ++y)
	{
		const uchar *gdata = image.ptr<uchar>(y);
		uchar *adata = allROI.ptr<uchar>(y);

		for (int x = left; x < right; ++x)
		{
			adata[3 * x] = gdata[3 * x];
			adata[3 * x + 1] = gdata[3 * x + 1];
			adata[3 * x + 2] = gdata[3 * x + 2];
		}
	}

	return allROI;
}

int main()
{
	//while (true) 
	for (int i = 1; i <= 452; i++)
	{
		cout << "predict" << endl;
		String fileName = to_string(i);
		String loadLocation = "test/" + fileName + ".jpg";
		String saveLocation = "binary/" + fileName + ".jpg";

		//Mat testImage = imread(loadImg, 0);
		Mat testImage = imread(loadLocation, 0);

		if (testImage.empty())
		{
			return -1;
		}

		Mat image;
		cvtColor(testImage, image, COLOR_GRAY2BGR);

		Mat binaryResult = cloudThreshold(image);
		Mat allROI(image.size(), CV_8UC3);
		allROI.setTo(Scalar::all(0));

		ifstream posFile;
		//posFile.open(loadPos);
		posFile.open("pos/" + fileName + ".txt");
		int pos[4] = { 0 };
		while (posFile >> pos[0])
		{
			posFile >> pos[1];
			posFile >> pos[2];
			posFile >> pos[3];
			allROI = copyROI(binaryResult, allROI, pos);
		}
		posFile.close();
		//imwrite(saveImgROI, allROI);
		imwrite(saveLocation, allROI);
	}
	return 0;
}