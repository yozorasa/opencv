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

	iou[0] = iintersection;
	iou[1] = uunion;
	if (uunion == 0)
		iou[2] = 0;
	else
		iou[2] = (float)iintersection / (float)uunion;
}

int main()
{
	ofstream IoUlogW;
	IoUlogW.open(saveImg + "log.csv");
	IoUlogW << "image,intersection,union,IoU" << endl;
	for (int i = imageStart; i <= imageFinish; i++) {
		String loadFileName = to_string(i);
		//cout << "Read Image >> " << loadFileName << endl;
		Mat gt = imread(loadGT + loadFileName + loadFileType);
		Mat image = imread(loadBin + loadFileName + loadFileType);
		//float* iou = IoU(gt, image, loadFileName);
		IoU(gt, image, loadFileName);
		cout << loadFileName + loadFileType << "," << iou[0] << "," << iou[1] << "," << iou[2] << endl;
		IoUlogW << loadFileName+loadFileType << "," << iou[0] << "," << iou[1] << "," << iou[2] << endl;
	}
	ifstream IoUlogR;
	IoUlogR.open(saveImg + "log.csv");
	float IoUAvg = 0;
	char line[128];
	int count = 0;
	while (IoUlogR.getline(line, sizeof(line), '\n'))
	{
		cout << line << endl;
		String temp;
		temp = strtok(line, ",");
		temp = strtok(NULL, ",");
		temp = strtok(NULL, ",");
		if (temp != "0") {
			count++;
			temp = strtok(NULL, ",");
			const char *tempChar = temp.c_str();
			IoUAvg += atof(tempChar);
			//cout << temp << endl;
		}
	}
	IoUAvg /= count;
	cout << IoUAvg << endl;
	IoUlogR.close();
	IoUlogW << ",,," << endl;
	IoUlogW << "Image Have Cloud:," << count << ",IoU AVG," << IoUAvg << endl;
	IoUlogW.close();
	return 0;
}