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

int cloudAmount = 498 ;
int otherAmount = 251 ;
int tag[498 + 251] = { 0 };

HOGDescriptor *hog = new HOGDescriptor(Size(64,64), Size(16,16), Size(8,8), Size(8,8), 9, 1 );
vector< Mat >  hogDatas;

void convert_to_ml( Mat& trainData )
{
    //--Convert data
    const int rows = (int)hogDatas.size();
    const int cols = (int)std::max( hogDatas[0].cols, hogDatas[0].rows );
    Mat tmp( 1, cols, CV_32FC1 ); //< used for transposition if needed
    trainData = Mat( rows, cols, CV_32FC1 );

    for( size_t i = 0 ; i < hogDatas.size(); ++i )
    {
        CV_Assert( hogDatas[i].cols == 1 || hogDatas[i].rows == 1 );

        if( hogDatas[i].cols == 1 )
        {
            transpose( hogDatas[i], tmp );
            tmp.copyTo( trainData.row( (int)i ) );
        }
        else if( hogDatas[i].rows == 1 )
        {
            hogDatas[i].copyTo( trainData.row( (int)i ) );
        }
    }
}

void hogCompute(Mat lbp){
	vector<float>  hogDescriptors;
	resize(lbp, lbp, Size(64,64), 0, 0, CV_INTER_AREA);
	hog->compute(lbp, hogDescriptors,Size(1,1), Size(0,0)); 
	hogDatas.push_back( Mat( hogDescriptors ).clone() );
}

int main()
{
	Mat lbp, roi;
	for (int i = 1; i <= cloudAmount; i++) {
		cout << loadLocationC + to_string(i) + "_lbp" + fileType << endl;
		lbp = imread(loadLocationC + to_string(i) + "_lbp" + fileType, 0);
		hogCompute(lbp);
	}
	for (int i = 1; i <= otherAmount; i++) {
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
	convert_to_ml( hogTrain_data );

	const int num_data = cloudAmount + otherAmount; //資料數
	//const int num_column = 256; //欄位數

	//Mat trainingDataMat(num_data, num_column, CV_32FC1, histogramCal);
	Mat labelsMat(num_data, 1, CV_32SC1, tag);
	Ptr<TrainData> trainingData = TrainData::create(hogTrain_data, ROW_SAMPLE, labelsMat);
	
	SVM::ParamTypes params;
	SVM::KernelTypes kernel_type = SVM::LINEAR;
	Ptr<SVM> svm = SVM::create();
	svm->setKernel(kernel_type);

	svm->trainAuto(trainingData);
	svm->save(record + "SVM_hog.xml");

	system("pause");
	return 0;
}