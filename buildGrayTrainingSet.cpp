#include <opencv2/opencv.hpp>  
using namespace cv;
#include<iostream>
#include<fstream>
using namespace std;
#include <stdlib.h>
#include <string>

int imageStart = 89;
int imageFinish = 452;
int clusterNumber = 4;
String loadLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\cloud\\img 1-452 (taiwan)\\";
String loadFileType = ".jpg";
String saveLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\cloud\\img 1-452 (taiwan)\\kmeansGray\\";

int bgrAbsThreshold = 15;
int bgrAvgThreshold = 160;
int greyThreshold = 160;
int rectSizeThreshold = 8000;

int cloudKey = 120;		//key 'x'
int somethingKey = 122;	//key 'z' 
int key = 0;

//int cloudCount = 582;
//int otherCount = 1556;

float widthUnit;
float heightUnit;

Mat kmeans(const Mat& image, int clusterCounts) {
	cout << "IN KMEANS" << endl;
	Mat labels;//聚類後的標籤
			   //assert(image.channels() != 1);
	int rows = image.rows;
	int cols = image.cols;
	int channels = image.channels();

	//保存聚類後的圖
	Mat clusteredMat(rows, cols, CV_8UC3);
	clusteredMat.setTo(Scalar::all(0));

	//pixels儲存各個像素
	Mat pixels(rows*cols, 1, CV_32FC3); //Color Mode
	pixels.setTo(Scalar::all(0));

	Mat centers(clusterCounts, 1, pixels.type());//儲存中心

												 //將image各個像素塞到pixels的一維陣列
	for (int i = 0; i < rows; ++i)
	{
		const uchar *idata = image.ptr<uchar>(i);
		float *pdata = pixels.ptr<float>(0);

		for (int j = 0; j < cols*channels; ++j)
		{
			pdata[i*cols*channels + j] = saturate_cast<float>(idata[j]);
			//pdata[i*cols + j] = idata[j];
		}
	}

	//分群
	cout << "START RUN KMEANS" << endl;
	kmeans(pixels, clusterCounts, labels, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 0), 5, KMEANS_PP_CENTERS, centers);

	cout << "FINISH RUN KMEANS" << endl;//cout << "Center of Color = " << endl;
										//cout << centers.size() << endl;
										//cout << centers << endl << endl;

										//判斷各群是否為雲
	int cloudFlag[10];

	//Color Mode
	Scalar clusterColor[10];
	double bgrAvg[10];
	int bgrDisFlag[10];
	for (int i = 0; i < clusterCounts; i++)
	{
		float *clusterCenter = centers.ptr<float>(i);
		bgrAvg[i] = 0;
		bgrDisFlag[i] = 0;
		cloudFlag[i] = 0;
		clusterColor[i] = Scalar(clusterCenter[0], clusterCenter[1], clusterCenter[2]);//聚類中心BGR值
																					   //計算BGR平均(色彩亮度)
		for (int j = 0; j < 3; j++)
		{
			bgrAvg[i] += clusterCenter[j];
		}
		bgrAvg[i] /= 3;

		//計算B G R相差多少(色彩飽和度)
		for (int j = 0; j < 3; j++)
		{
			if (abs(bgrAvg[i] - clusterCenter[j]) > bgrAbsThreshold)
			{
				//cout << "Dis = " << abs(bgrAvg[i] - clusterCenter[j]) << endl;
				bgrDisFlag[i] = 1;
			}
		}

		//亮度高 飽和度低 判斷為雲
		if (bgrDisFlag[i] == 0 && bgrAvg[i] >= bgrAvgThreshold)
		{
			cloudFlag[i] = 1;
		}
		cout << "bgrAvg = " << bgrAvg[i] << endl;
		cout << "bgrDisFlag = " << bgrDisFlag[i] << endl;
		cout << "cloudFlag = " << cloudFlag[i] << endl;
	}

	/*/Grey Mode
	int clusterColor[10];
	for (int i = 0; i < clusterCounts; i++)
	{
	clusterColor[i] = centers.at<float>(i);
	cloudFlag[i] = 0;
	if (centers.at<float>(i) >= greyThreshold)
	{
	cloudFlag[i] = 1;
	}
	//cout << "cloudFlag = " << cloudFlag[i] << endl;
	}
	*/

	//二值化 雲(前景) 非雲(背景)
	clusteredMat = image.clone();
	Scalar paintOn;
	//Color Mode
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols*channels; j += channels)
		{
			if (cloudFlag[labels.at<int>(i*cols + (j / channels))] == 1)
			{
				paintOn = Scalar(255, 255, 255);
			}
			else
			{
				paintOn = Scalar(0, 0, 0);
				circle(clusteredMat, Point(j / channels, i), 1, paintOn); //非雲(背景)為黑
			}
			//circle(clusteredMat, Point(j / channels, i), 1, paintOn); //非雲(背景)為黑 雲(前景)為白
		}
	}

	/*/Gray Mode
	cout << "PAINT" << endl;
	for (int i = 0; i < rows; ++i)
	{
	for (int j = 0; j < cols; ++j)
	{
	if (cloudFlag[labels.at<int>(i*cols + j)] == 1)
	{
	paintOn = Scalar(255, 255, 255);
	}
	else
	{
	paintOn = Scalar(0, 0, 0);
	circle(clusteredMat, Point(j, i), 1, paintOn);
	}
	//circle(clusteredMat, Point(j, i), 1, colorTab[labels.at<int>(i*cols + j)]);        //標記像素點的類別，顏色區分
	}
	}
	*/
	cout << "OUT KMEANS" << endl;
	return clusteredMat;
}

Mat cutROI(const Mat& cImg, const Mat& image, const Mat& clusteredMat, String fileName)
{
	cout << "IN ROI" << endl;

	ofstream fout(saveLocation + "yolo\\" + fileName + ".txt");

	int rows = image.rows;
	int cols = image.cols;
	float widthUnit = 1.0 / cols;
	float heightUnit = 1.0 / rows;

	rectSizeThreshold = rows * cols / 1000;
	int channels = image.channels();
	//cout << "rectSizeThresholdl = " << rectSizeThreshold << endl;
	Mat rectClusteredMat = clusteredMat.clone();
	Mat clusterMatGray;
	cvtColor(clusteredMat, clusterMatGray, COLOR_BGR2GRAY);
	//GaussianBlur(clusteredMat, edgeColor, Size(3, 3), 0);
	//Canny(clusteredMat, edgeColor, 50, 150, 3);
	dilate(clusterMatGray, clusterMatGray, Mat());
	dilate(clusterMatGray, clusterMatGray, Mat());
	erode(clusterMatGray, clusterMatGray, Mat());
	erode(clusterMatGray, clusterMatGray, Mat());
	imshow("clusterMatGray", clusterMatGray);
	Mat contoursImg = image.clone();
	Mat allROI(image.size(), CV_8UC3, Scalar(0, 0, 0));
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	RNG rng(12345);
	Mat cmContour = clusteredMat.clone();

	findContours(clusterMatGray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	for (int i = 0; i<contours.size(); i++) {
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), 255);
		drawContours(contoursImg, contours, i, color, 2, 8, hierarchy);
		drawContours(cmContour, contours, i, color, 2, 8, hierarchy);
		Rect contoursRect = boundingRect(contours[i]);
		//cout << "Contours " << i << " = " << contours[i] << endl;
		if (contoursRect.width*contoursRect.height > rectSizeThreshold)
		{
			//cout << "RECT " << i << " = " << contoursRect.width*contoursRect.height << endl;
			rectangle(rectClusteredMat, contoursRect, cv::Scalar(0, 0, 255), 2);
			Mat rectROI(contoursRect.size(), CV_8UC3, Scalar(0, 0, 0));
			Mat crectROI(contoursRect.size(), CV_8UC3, Scalar(0, 0, 0));
			for (int y = 0; y < contoursRect.height; ++y)
			{
				const uchar *idata = clusteredMat.ptr<uchar>(y + contoursRect.y);
				const uchar *cdata = cImg.ptr<uchar>(y + contoursRect.y);
				const uchar *gdata = image.ptr<uchar>(y + contoursRect.y);
				uchar *adata = allROI.ptr<uchar>(y + contoursRect.y);
				uchar *pdata = rectROI.ptr<uchar>(y);
				uchar *rcdata = crectROI.ptr<uchar>(y);

				for (int x = 0; x < contoursRect.width; ++x)
				{
					int xx = contoursRect.x;
					pdata[3 * x] = gdata[3 * (x + xx)];
					pdata[3 * x + 1] = gdata[3 * (x + xx) + 1];
					pdata[3 * x + 2] = gdata[3 * (x + xx) + 2];

					rcdata[3 * x] = cdata[3 * (x + xx)];
					rcdata[3 * x + 1] = cdata[3 * (x + xx) + 1];
					rcdata[3 * x + 2] = cdata[3 * (x + xx) + 2];

					adata[3 * (x + xx)] = idata[3 * (x + xx)];
					adata[3 * (x + xx) + 1] = idata[3 * (x + xx) + 1];
					adata[3 * (x + xx) + 2] = idata[3 * (x + xx) + 2];
				}
			}
			//namedWindow("RectImage" + i);
			imshow("cloud or not", crectROI);
			//wait key

			cout << "Enter: ";
			key = waitKey(0);
			cout << key;
			while (key != cloudKey && key != somethingKey) {
				cout << endl << "Enter: ";
				key = waitKey(0);
				cout << key;
			}

			if (key == cloudKey) {
				cout << ">> cloud" << endl;
				//write the position infomation in file

				//BBox Label Tool
				//fout << "0 " << contoursRect.x << " " << contoursRect.y;
				//fout << " " << contoursRect.x+contoursRect.width << " " << contoursRect.y+contoursRect.height << endl;

				//YOLOv2 format
				float centerX = contoursRect.x + contoursRect.width / 2.0;
				float centerY = contoursRect.y + contoursRect.height / 2.0;

				float x = centerX * widthUnit;
				float y = centerY * heightUnit;
				float w = contoursRect.width * widthUnit;
				float h = contoursRect.height * heightUnit;

				fout << "0 " << x << " " << y << " " << w << " " << h << endl;
				cout << ">> Cloud\\" + fileName + "_" + to_string(i) + ".jpg" << endl;
				
				imwrite(saveLocation + "cutCloud\\" + fileName + "_" + to_string(i) + ".jpg", rectROI);
				//imwrite(saveLocation + "cutCloud\\" + to_string(cloudCount) + ".jpg", rectROI);
				//cloudCount++;

			}
			else {
				cout << ">> Other\\" + fileName + "_" + to_string(i) + ".jpg" << endl;
				imwrite(saveLocation + "cutOther\\" + fileName + "_" + to_string(i) + ".jpg", rectROI);
				//imwrite(saveLocation + "cutOther\\" + to_string(otherCount) + ".jpg", rectROI);
				//otherCount++;
			}
		}

	}
	//imshow("rectClusteredMat", rectClusteredMat);
	//imshow("cmContour", cmContour);
	//imshow("contoursImg", contoursImg);

	//imwrite(saveLocation + fileName + "Contours" + ".jpg", contoursImg);

	fout.close();
	cout << "OUT ROI" << endl;
	return allROI;
}


int main()
{
	for (int i = imageStart; i <= imageFinish; i++) {
		String clusterNumber_str = to_string(clusterNumber);
		String loadFileName = to_string(i);
		String saveFileGray = saveLocation + "yolo\\" + loadFileName + loadFileType;
		String saveFileKmeans = saveLocation + loadFileName + "_K" + loadFileType;
		String saveFileROI = saveLocation + loadFileName + "_ROI" + loadFileType;

		cout << "Read Image >> " << loadFileName + loadFileType << endl;
		Mat colorImage = imread(loadLocation + loadFileName + loadFileType);
		Mat testImage = imread(loadLocation + loadFileName + loadFileType, 0);
		if (testImage.empty())
		{
			return -1;
		}

		Mat image;
		cvtColor(testImage, image, COLOR_GRAY2BGR);
		imwrite(saveFileGray, image);

		//Mat kmeansResult = kmeans(image, clusterNumber);
		//imwrite(saveFileKmeans, kmeansResult);

		Mat kmeansResult = imread(saveFileKmeans);
		Mat allROI = cutROI(colorImage, image, kmeansResult, loadFileName);
		//imwrite(saveFileROI, allROI);

		//waitKey(0);
	}
	return 0;
}