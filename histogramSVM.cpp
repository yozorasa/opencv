#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>
#include <fstream>
#include <cstdio>
#include <vector> 
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

using namespace cv;
using namespace std;
using namespace ml;

String record = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\lbp\\";
String loadLocationC = record + "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\lbp\\cloud\\";
String loadLocationO = record + "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\lbp\\other\\";
String fileType = ".jpg";

int cloudAmount = 497;
int otherAmount = 251;
float histTemp[256] = { 0 };
float histogram[748][256] = { 0 };
int tag[748] = { 0 };

int histogram(Mat lbp, Mat roi) {
	int hcount[256] = { 0 };
	float histogram[256] = { 0 };
	int rows = lbp.rows;
	int cols = lbp.cols;
	int pixelCount = 0;
	for (int r = 0; r < rows; ++r) {
		const uchar *lbpdata = lbp.ptr<uchar>(r);
		const uchar *roidata = roi.ptr<uchar>(r);
		for (int c = 0; c < cols; ++c)
		{
			if (!(roidata[3 * c] == 255 && roidata[3 * c + 1] == 0 && roidata[3 * c + 2] == 255)) {
				hcount[lbpdata[c]]++;
				pixelCount++;
			}
			else {
				//cout << "MAGENTA" << endl;
			}
		}
	}
	lbp.release();
	roi.release();
	for (int z = 0; z<256 && pixelCount != 0; z++) {
		//cout << "z = " << z << " hcount = ";
		//cout << hcount[z];
		histTemp[z] = (float)hcount[z] / pixelCount;
		//cout << " hist = " << histTemp[z] << " pixel = " << pixelCount << endl;
	}
	//cout << "Have " << pixelCount << "pixels." << endl;
	return 0;
}

int main()
{
	Mat lbp, roi;
	for (int i = 1; i<=cloudAmount; i++) {
		//file << cloudFileName[i] << "\n";
		cout << loadLocationC+to_string(i)+"_lbp"+fileType << endl;
		lbp = imread(loadLocationC+to_string(i)+"_lbp"+fileType, 0);
		roi = imread(loadLocationC+to_string(i)+"_roi"+fileType);
		//imshow("lbp", lbp);
		//imshow("roi", roi);
		for (int j = 0; j<256; i++) {
			histogram[i-1][j] = histTemp[j];
			cout << histogram[i-1][j] << endl;
		}
	}
	for (int i = 1; i<=otherAmount; i++) {
		//file << otherFileName[i] << "\n";
		cout << loadLocationO+to_string(i)+"_lbp"+fileType << endl;
		lbp = imread(loadLocationO+to_string(i)+"_lbp"+fileType, 0);
		roi = imread(loadLocationO+to_string(i)+"_roi"+fileType);
		//imshow("lbp", lbp);
		//imshow("roi", roi);
		for (int j = 0; j<256; i++) {
			histogram[i+cloudAmount-1][j] = histTemp[j];
			cout << histogram[i+cloudAmount-1][j] << endl;
		}
	}
	for (int i = 0; i<cloudAmount+otherAmount; i++) {
		if (i<cloudAmount)
			tag[i] = 1;
		else
			tag[i] = -1;
	}

	const int num_data = cloudAmount + otherAmount; //資料數
	const int num_column = 256; //欄位數

	Mat trainingDataMat(num_data, num_column, CV_32FC1, histogram);
	Mat labelsMat(num_data, 1, CV_32SC1, tag);

	SVM::ParamTypes params;
    SVM::KernelTypes kernel_type = SVM::LINEAR;
    Ptr<SVM> svm = SVM::create();
    svm->setKernel(kernel_type);

    svm->trainAuto(trainingData);
    svm->save(record+"SVM.xml");

	system("pause");
	return 0;
}