#include <opencv2/opencv.hpp>  
using namespace cv;
#include<iostream>
#include<fstream>
using namespace std;
#include <stdlib.h>
#include <string>

int imageStart = 1;
int imageFinish = 452;
String loadGT = "groundTruth/";
String loadBin = "binary/";
String loadFileType = ".jpg";
String saveImg = "IoU/";

float iou[3] = { 0 };

void IoU(const Mat& groundTruthImg, const Mat& image, String fileName)
{
	int iintersection = 0;
	int uunion = 0;

	Mat iouImg = groundTruthImg.clone();
	int rows = iouImg.rows;
	int cols = iouImg.cols;
	int channels = iouImg.channels();

	for (int y = 0; y < rows; ++y)
	{
		const uchar *idata = image.ptr<uchar>(y);
		uchar *iouData = iouImg.ptr<uchar>(y);

		for (int x = 0; x < cols; ++x)
		{
			if (idata[3 * x] == 0 && idata[3 * x + 1] == 0 && idata[3 * x + 2] == 0)
			{
				if (iouData[3 * x] == 0 && iouData[3 * x + 1] == 0 && iouData[3 * x + 2] == 0)
				{//都不覺得是雲 correct, no matter
					iouData[3 * x] = 0;
					iouData[3 * x + 1] = 0;
					iouData[3 * x + 2] = 0;
				}
				else
				{//事實上是雲 但預測不是雲 wrong, red
					iouData[3 * x] = 0;
					iouData[3 * x + 1] = 0;
					iouData[3 * x + 2] = 255;
					uunion += 1;
				}
			}
			else {
				if (iouData[3 * x] == 0 && iouData[3 * x + 1] == 0 && iouData[3 * x + 2] == 0)
				{//事實上不是雲 但預測是雲 wrong, blue
					iouData[3 * x] = 255;
					iouData[3 * x + 1] = 0;
					iouData[3 * x + 2] = 0;
					uunion += 1;
				}
				else
				{//事實上是雲 預測也是雲 correct, white
					iouData[3 * x] = 255;
					iouData[3 * x + 1] = 255;
					iouData[3 * x + 2] = 255;
					iintersection += 1;
					uunion += 1;
				}
			}
		}
	}

	imwrite(saveImg + fileName + ".jpg", iouImg);

	//float iou[3] = { 0 };
	iou[0] = iintersection;
	iou[1] = uunion;
	if (uunion == 0)
		iou[2] = 0;
	else
		iou[2] = (float)iintersection / (float)uunion;
	//return iou;
}

int main()
{
	ofstream IoUlog;
	IoUlog.open(saveImg + "log.csv");
	IoUlog << "image,intersection,union,IoU" << endl;
	for (int i = imageStart; i <= imageFinish; i++) {
		String loadFileName = to_string(i);
		//cout << "Read Image >> " << loadFileName << endl;
		Mat gt = imread(loadGT + loadFileName + loadFileType);
		Mat image = imread(loadBin + loadFileName + loadFileType);
		//float* iou = IoU(gt, image, loadFileName);
		IoU(gt, image, loadFileName);
		cout << loadFileName + loadFileType << "," << iou[0] << "," << iou[1] << "," << iou[2] << endl;
		IoUlog << loadFileName+loadFileType << "," << iou[0] << "," << iou[1] << "," << iou[2] << endl;
	}
	return 0;
}