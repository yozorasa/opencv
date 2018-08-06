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
#include <opencv2/objdetect.hpp>

using namespace cv;
using namespace std;
using namespace ml;

String record = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\lbpRename\\";
String loadLocationC = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\lbpRename\\cloud\\";
String loadLocationO = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\lbpRename\\other\\";
String fileType = ".jpg";
String svmFileName = "test.xml";

int cloudAmount = 498;
int otherAmount = 251;
float histTemp[256] = { 0 };
float histogramCal[498 + 251][256] = { 0 };
int tag[498 + 251] = { 0 };


HOGDescriptor *hog = new HOGDescriptor(Size(64, 64), Size(8, 8), Size(4, 4), Size(4, 4), 9, 1);

vector< Mat >  hogDatas;

void convert_to_ml(Mat& trainData)
{
	//--Convert data
	const int rows = (int)hogDatas.size();
	const int cols = (int)std::max(hogDatas[0].cols, hogDatas[0].rows);
	Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
	trainData = Mat(rows, cols, CV_32FC1);

	for (size_t i = 0; i < hogDatas.size(); ++i)
	{
		CV_Assert(hogDatas[i].cols == 1 || hogDatas[i].rows == 1);

		if (hogDatas[i].cols == 1)
		{
			transpose(hogDatas[i], tmp);
			tmp.copyTo(trainData.row((int)i));
		}
		else if (hogDatas[i].rows == 1)
		{
			hogDatas[i].copyTo(trainData.row((int)i));
		}
	}
}

void convert_to_ml2(Mat &trainData, Mat hogDatas)
{
	//--Convert data
	// const int rows = (int)hogDatas.size();
	const int cols = (int)std::max(hogDatas.cols, hogDatas.rows);
	Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
	trainData = Mat(1, cols, CV_32FC1);

	// for (size_t i = 0; i < hogDatas.size(); ++i)
	// {
	CV_Assert(hogDatas.cols == 1 || hogDatas.rows == 1);

	if (hogDatas.cols == 1)
	{
		transpose(hogDatas, tmp);
		tmp.copyTo(trainData.row(0));
	}
	else if (hogDatas.rows == 1)
	{
		hogDatas.copyTo(trainData.row(0));
	}
	// }
}

vector<float> hogCompute(Mat lbp) {
	vector<float>  hogDescriptors;
	resize(lbp, lbp, Size(64, 64), 0, 0, CV_INTER_AREA);
	hog->compute(lbp, hogDescriptors, Size(1, 1), Size(0, 0));
	hogDatas.push_back(Mat(hogDescriptors).clone());
	return hogDescriptors;
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
	svm->setType(SVM::ONE_CLASS);
	Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);
	svm->train(td);
	//svm->trainAuto(td);
	svm->save("test.xml");
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
		hogCompute(lbp);
	}
	for (int i = 1; i <= otherAmount; i++) {
		//file << loadLocationO+to_string(i)+"_lbp"+fileType << "\n";
		cout << loadLocationO + to_string(i) + "_lbp" + fileType << endl;
		lbp = imread(loadLocationO + to_string(i) + "_lbp" + fileType, 0);
		hogCompute(lbp);
	}
	for (int i = 0; i<cloudAmount + otherAmount; i++) {
		if (i<cloudAmount)
			tag[i] = 1;
		else
			tag[i] = -1;
	}
	Mat hogTrain_data;
	convert_to_ml(hogTrain_data);

	const int num_data = cloudAmount + otherAmount; //資料數

	Mat labelsMat(num_data, 1, CV_32SC1, tag);

	//SVMtrain(hogTrain_data, labelsMat);

	Ptr<TrainData> trainingData = TrainData::create(hogTrain_data, ROW_SAMPLE, labelsMat);

	SVM::ParamTypes params;
	SVM::Types svm_type = SVM::C_SVC;
	SVM::KernelTypes kernel_type = SVM::LINEAR;
	Ptr<SVM> svm = SVM::create();
	svm->setKernel(kernel_type);
	svm->setType(svm_type);
	svm->trainAuto(trainingData);
	svm->save(record + "test.xml");

	for (int i = 1; i <= cloudAmount; i++) {
		lbp = imread(loadLocationC + to_string(i) + "_lbp" + fileType, 0);
		vector<float> src = hogCompute(lbp);
		Mat hogTrain_data;
		convert_to_ml2(hogTrain_data, Mat(src));
		int response = svm->predict(hogTrain_data);
		cout << "Cloud <<< flag = " << response << endl;
	}
	for (int i = 1; i <= otherAmount; i++) {
		lbp = imread(loadLocationO + to_string(i) + "_lbp" + fileType, 0);
		vector<float> src = hogCompute(lbp);
		Mat hogTrain_data;
		convert_to_ml2(hogTrain_data, Mat(src));
		int response = svm->predict(hogTrain_data);
		cout << "Other <<< flag = " << response << endl;
	}
	waitKey(0);

	system("pause");
	return 0;
}