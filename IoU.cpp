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
float cm[2][2] = { 0 };

void fillRect(Mat& image)
{
	Mat imageGray;
	cvtColor(image, imageGray, COLOR_BGR2GRAY);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(imageGray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);

	for (int i = 0; i < contours.size(); i++) {
		Rect contoursRect = boundingRect(contours[i]);
		rectangle(image, contoursRect, cv::Scalar(255, 255, 255), CV_FILLED);
	}
}


void IoU_CM(const Mat& groundTruthImg, const Mat& image, String fileName)
{
	int iintersection = 0;
	int uunion = 0;

	int TP = 0;
	int FN = 0;
	int FP = 0;
	int TN = 0;

	Mat iouImg = groundTruthImg.clone();
	iouImg.setTo(Scalar::all(0));
	int rows = iouImg.rows;
	int cols = iouImg.cols;
	int channels = iouImg.channels();

	for (int y = 0; y < rows; ++y)
	{
		const uchar *iData = image.ptr<uchar>(y);
		const uchar *gtData = groundTruthImg.ptr<uchar>(y);
		uchar *iouData = iouImg.ptr<uchar>(y);
		for (int x = 0; x < cols; ++x)
		{
			if (iData[3 * x] == 0 && iData[3 * x + 1] == 0 && iData[3 * x + 2] == 0)
			{
				if (gtData[3 * x] == 0 && gtData[3 * x + 1] == 0 && gtData[3 * x + 2] == 0)
				{//都不覺得是雲 correct, black
					iouData[3 * x] = 0;
					iouData[3 * x + 1] = 0;
					iouData[3 * x + 2] = 0;
				}
				else
				{//事實上是雲 但預測不是雲 wrong, red
					iouData[3 * x] = 0;
					iouData[3 * x + 1] = 0;
					iouData[3 * x + 2] = 255;
				}
			}
			else {
				if (gtData[3 * x] == 0 && gtData[3 * x + 1] == 0 && gtData[3 * x + 2] == 0)
				{//事實上不是雲 但預測是雲 wrong, blue
					iouData[3 * x] = 255;
					iouData[3 * x + 1] = 0;
					iouData[3 * x + 2] = 0;
				}
				else
				{//事實上是雲 預測也是雲 correct, white
					iouData[3 * x] = 255;
					iouData[3 * x + 1] = 255;
					iouData[3 * x + 2] = 255;
				}
			}
		}
	}
	for (int y = 0; y < rows; ++y)
	{
		uchar *iouData = iouImg.ptr<uchar>(y);
		for (int x = 0; x < cols; ++x)
		{
			if (iouData[3 * x] == 0 && iouData[3 * x + 1] == 0 && iouData[3 * x + 2] == 0)
			{//都不覺得是雲 correct, black
				TN += 1;
			}
			else if (iouData[3 * x] == 0 && iouData[3 * x + 1] == 0 && iouData[3 * x + 2] == 255)
			{//事實上是雲 但預測不是雲 wrong, red
				uunion += 1;
				FN += 1;	
			}
			else if (iouData[3 * x] == 255 && iouData[3 * x + 1] == 0 && iouData[3 * x + 2] == 0)
			{//事實上不是雲 但預測是雲 wrong, blue
				uunion += 1;
				FP += 1;
			}
			else
			{//事實上是雲 預測也是雲 correct, white
				iintersection += 1;
				uunion += 1;
				TP += 1;
			}
		}
	}
	//imshow("C", iouImg);
	//waitKey();
	imwrite(saveImg + fileName + ".jpg", iouImg);

	iou[0] = iintersection;
	iou[1] = uunion;
	if (uunion == 0)
		iou[2] = 0;
	else
		iou[2] = (float)iintersection / (float)uunion;

	int cmCount = TP + FN + FP + TN;
	cm[0][0] = (float)TP / (float)cmCount;
	cm[0][1] = (float)FN / (float)cmCount;
	cm[1][0] = (float)FP / (float)cmCount;
	cm[1][1] = (float)TN / (float)cmCount;
}

int main()
{
	ofstream IoUlogW;
	IoUlogW.open(saveImg + "log.csv");
	IoUlogW << "image,intersection,union,IoU,,TP,FN,FP,TN" << endl;
	for (int i = imageStart; i <= imageFinish; i++) {
		String loadFileName = to_string(i);
		//cout << "Read Image >> " << loadFileName << endl;
		Mat gt = imread(loadGT + loadFileName + loadFileType);
		Mat image = imread(loadBin + loadFileName + loadFileType);
		//float* iou = IoU(gt, image, loadFileName);
		fillRect(gt);
		fillRect(image);
		//imshow("A", gt);
		//imshow("B", image);
		//waitKey();
		imwrite(saveImg + loadFileName + "gt.jpg", gt);
		imwrite(saveImg + loadFileName + "yl.jpg", image);
		IoU_CM(gt, image, loadFileName);
		cout << loadFileName + loadFileType << "," << iou[0] << "," << iou[1] << "," << iou[2] << ",," << cm[0][0] << "," << cm[0][1] << "," << cm[1][0] << "," << cm[1][1] << endl;
		IoUlogW << loadFileName + loadFileType << "," << iou[0] << "," << iou[1] << "," << iou[2] << ",," << cm[0][0] << "," << cm[0][1] << "," << cm[1][0] << "," << cm[1][1] << endl;
	}

	ifstream IoUlogR;
	IoUlogR.open(saveImg + "log.csv");
	float IoUAvg = 0;
	char line[128];
	int IoUCount = -1;
	int cmCount = -1;
	float TP = 0;
	float FN = 0;
	float FP = 0;
	float TN = 0;
	while (IoUlogR.getline(line, sizeof(line), '\n'))
	{
		cout << line << endl;
		String temp;
		const char *tempChar;
		temp = strtok(line, ",");//image
		temp = strtok(NULL, ",");//intersection
		temp = strtok(NULL, ",");//union
		if (temp != "0") {
			IoUCount++;
			temp = strtok(NULL, ",");//IoU
			tempChar = temp.c_str();
			IoUAvg += atof(tempChar);
			//cout << temp << endl;
		}

		cmCount += 1;
		temp = strtok(NULL, ",");//TP
		tempChar = temp.c_str();
		TP += atof(tempChar);
		temp = strtok(NULL, ",");//FN
		tempChar = temp.c_str();
		FN += atof(tempChar);
		temp = strtok(NULL, ",");//FP
		tempChar = temp.c_str();
		FP += atof(tempChar);
		temp = strtok(NULL, ",");//TN
		tempChar = temp.c_str();
		TN += atof(tempChar);
	}
	IoUAvg /= IoUCount;
	TP /= cmCount;
	FN /= cmCount;
	FP /= cmCount;
	TN /= cmCount;
	cout << IoUAvg << endl;
	IoUlogR.close();
	IoUlogW << ",,,,,,,," << endl;
	IoUlogW << "Image Have Cloud:," << IoUCount << ",IoU AVG:," << IoUAvg << ",CM AVG:," << TP << "," << FN << "," << FP << "," << TN << endl;
	IoUlogW.close();
	return 0;
}