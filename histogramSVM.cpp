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

String record = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\lbpRename\\";
String loadLocationC = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\lbpRename\\cloud\\";
String loadLocationO = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\lbpRename\\other\\";
String fileType = ".jpg";
String svmFileName = "test.xml";

int cloudAmount = 5770 / 2;
int otherAmount = 1780 / 2;
float histTemp[256] = { 0 };
float histogramCal[5770 / 2 + 1780 / 2][256] = { 0 };
int tag[5770 / 2 + 1780 / 2] = { 0 };

int histogram(Mat lbp, Mat roi) {
	int hcount[256] = { 0 };
	int rows = lbp.rows;
	int cols = lbp.cols;
	int pixelCount = 0;
	for (int r = 0; r < rows; ++r) {
		const uchar *lbpdata = lbp.ptr<uchar>(r);
		const uchar *roidata = roi.ptr<uchar>(r);
		for (int c = 0; c < cols; ++c)
		{
			//cout << roidata[3 * c] << "" << roidata[3 * c + 1] << "" << roidata[3 * c + 2] << endl;
			if (!(roidata[3 * c] == 255 && roidata[3 * c + 1] == 0 && roidata[3 * c + 2] == 255)) {
				hcount[lbpdata[c]]++;
				pixelCount++;
			}
			else {
				//cout << "MAGENTA" << endl;
			}
		}
	}
	for (int z = 0; z<256; z++) {
		//cout << "z = " << z << " hcount = " << hcount[z];
		if (pixelCount == 0)
			histTemp[z] = 0;
		else
			histTemp[z] = (float)hcount[z] / pixelCount;
		//cout << " hist = " << histTemp[z] << " pixel = " << pixelCount << endl;
	}
	//cout << "Have " << pixelCount << "pixels." << endl;
	return 0;
}

void getSVMParams(SVM *svm)
{
	cout << "Kernel type     : " << svm->getKernelType() << endl;
	cout << "Type            : " << svm->getType() << endl;
	cout << "C               : " << svm->getC() << endl;
	cout << "Degree          : " << svm->getDegree() << endl;
	cout << "Nu              : " << svm->getNu() << endl;
	cout << "Gamma           : " << svm->getGamma() << endl;
}

void SVMtrain(Mat &trainMat, Mat trainLabels) {
	Ptr<SVM> svm = SVM::create();
	svm->setGamma(0.50625);
	svm->setC(100);
	svm->setKernel(SVM::LINEAR);
	svm->setType(SVM::C_SVC);
	Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);
	svm->train(td);
	//svm->trainAuto(td);
	svm->save(record + svmFileName);
	getSVMParams(svm);

	/*
	To acheive 100% rate.
	Descriptor Size : 576
	Kernel type     : 2
	Type            : 100
	C               : 2.5
	Degree          : 0
	Nu              : 0
	Gamma           : 0.03375
	the accuracy is :100

	*/

}

int main()
{
	Mat lbp, roi;
	for (int i = 1; i <= cloudAmount; i++) {
		//file << loadLocationC+to_string(i)+"_lbp"+fileType << "\n";
		cout << loadLocationC + to_string(i) + "_lbp" + fileType << endl;
		lbp = imread(loadLocationC + to_string(i) + "_lbp" + fileType, 0);
		roi = imread(loadLocationC + to_string(i) + "_roi" + fileType);
		//imshow("lbp", lbp);
		//imshow("roi", roi);
		histogram(lbp, roi);
		for (int j = 0; j<256; j++) {
			histogramCal[i - 1][j] = histTemp[j];
			//cout << histogramCal[i - 1][j] << endl;
		}
	}
	for (int i = 1; i <= otherAmount; i++) {
		//file << loadLocationO+to_string(i)+"_lbp"+fileType << "\n";
		cout << loadLocationO + to_string(i) + "_lbp" + fileType << endl;
		lbp = imread(loadLocationO + to_string(i) + "_lbp" + fileType, 0);
		roi = imread(loadLocationO + to_string(i) + "_roi" + fileType);
		//imshow("lbp", lbp);
		//imshow("roi", roi);
		histogram(lbp, roi);
		for (int j = 0; j<256; j++) {
			histogramCal[i + cloudAmount - 1][j] = histTemp[j];
			//cout << histogramCal[i + cloudAmount - 1][j] << endl;
		}
	}
	for (int i = 0; i<cloudAmount + otherAmount; i++) {
		if (i<cloudAmount)
			tag[i] = 1;
		else
			tag[i] = -1;
	}

	const int num_data = cloudAmount + otherAmount; //資料數
	const int num_column = 256; //欄位數

	Mat trainingDataMat(num_data, num_column, CV_32FC1, histogramCal);
	Mat labelsMat(num_data, 1, CV_32SC1, tag);

	//SVMtrain(hogTrain_data, labelsMat);

	Ptr<TrainData> trainingData = TrainData::create(trainingDataMat, ROW_SAMPLE, labelsMat);

	SVM::ParamTypes params;
	SVM::Types svm_type = SVM::C_SVC;
	SVM::KernelTypes kernel_type = SVM::LINEAR;
	Ptr<SVM> svm = SVM::create();
	svm->setKernel(kernel_type);
	svm->setType(svm_type);
	svm->trainAuto(trainingData);
	svm->save(record + svmFileName);

	for (int i = 1; i <= cloudAmount; i++) {
		lbp = imread(loadLocationC + to_string(i) + "_lbp" + fileType, 0);
		roi = imread(loadLocationC + to_string(i) + "_roi" + fileType);
		histogram(lbp, roi);
		float testData[256] = { 0 };
		for (int i = 0; i < 256; i++) {
			testData[i] = histTemp[i];
			//cout << testData[i] << endl;
		}
		Mat src(1, 256, CV_32FC1, testData);
		int response = svm->predict(src);
		cout << "Cloud <<< flag = " << response << endl;
	}
	for (int i = 1; i <= otherAmount; i++) {
		lbp = imread(loadLocationO + to_string(i) + "_lbp" + fileType, 0);
		roi = imread(loadLocationO + to_string(i) + "_roi" + fileType);
		histogram(lbp, roi);
		float testData[256] = { 0 };
		for (int i = 0; i < 256; i++) {
			testData[i] = histTemp[i];
			//cout << testData[i] << endl;
		}
		Mat src(1, 256, CV_32FC1, testData);
		int response = svm->predict(src);
		cout << "Other <<< flag = " << response << endl;
	}

	system("pause");
	return 0;
}