#include <opencv2/opencv.hpp>  
using namespace cv;
#include<iostream>
using namespace std;
#include <stdlib.h>
#include <string> 

Scalar colorTab[] =     //10個顏色  
{
	Scalar(0, 0, 255),
	Scalar(0, 255, 0),
	Scalar(255, 100, 100),
	Scalar(255, 0, 255),
	Scalar(0, 255, 255),
	Scalar(255, 0, 0),
	Scalar(255, 255, 0),
	Scalar(255, 0, 100),
	Scalar(100, 100, 100),
	Scalar(50, 125, 125)
};

class ClusterPixels
{
private:
	Mat image;          //待聚類圖象  
	Mat labels;         //聚類後的標籤  
	int clusterCounts;  //分類數,不得大於10，只是顏色定義只有10類，並不是算法限制
	Mat edgeColor;
	Mat edgeGray;

public:
	ClusterPixels() :clusterCounts(0) {}
	ClusterPixels(const Mat& src, int clusters = 5) :clusterCounts(clusters) { image = src.clone(); }

	void setImage(const Mat& src) { image = src.clone(); };
	void setClusters(int clusters) { clusterCounts = clusters; }

	Mat getLabels() { return labels; };      //返回聚類後的標籤  

	Mat clusterGrayImageByKmeans()
	{
		Mat originImage = image.clone();
		//轉換成灰度圖  
		if (image.channels() != 1)
			cvtColor(image, image, COLOR_BGR2GRAY);

		int rows = image.rows;
		int cols = image.cols;

		//保存聚類後的圖片  
		Mat clusteredMat(rows, cols, CV_8UC3);
		clusteredMat.setTo(Scalar::all(0));

		Mat pixels(rows*cols, 1, CV_32FC1); //pixels用於保存所有的灰度像素
		Mat centers(clusterCounts, 1, pixels.type());

		for (int i = 0; i < rows; ++i)
		{
			const uchar *idata = image.ptr<uchar>(i);
			float *pdata = pixels.ptr<float>(0);
			for (int j = 0; j < cols; ++j)
			{
				pdata[i*cols + j] = idata[j];
			}
		}

		kmeans(pixels, clusterCounts, labels, TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 0), 5, KMEANS_PP_CENTERS, centers);
		cout << "Center of Gray = " << endl;
		cout << centers.size() << endl;
		cout << centers << endl << endl << endl;
		cout << "test = " << centers.at<float>(0) << endl;
		int maxLabel = 0;
		int cloudFlag[10];
		for (int i = 0; i < clusterCounts; i++)
		{
			cloudFlag[i] = 0;
			if (centers.at<float>(i) >= 170)
			{
				cloudFlag[i] = 1;
			}
			cout << "cloudFlag = " << cloudFlag[i] << endl;
		}

		Scalar paintOn;
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
				}
				circle(clusteredMat, Point(j, i), 1, paintOn);
				//circle(clusteredMat, Point(j, i), 1, colorTab[labels.at<int>(i*cols + j)]);        //標記像素點的類別，顏色區分
			}
		}

		cvtColor(clusteredMat, edgeGray, COLOR_BGR2GRAY);
		//GaussianBlur(clusteredMat, edgeGray, Size(3, 3), 0);
		//Canny(clusteredMat, edgeGray, 50, 150, 3);
		//imshow("CannyGray", edgeGray);
		Mat contoursImg = originImage.clone();
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		RNG rng(12345);
		findContours(edgeGray, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
		for (int i = 0; i<contours.size(); i++) {
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), 255);
			drawContours(contoursImg, contours, i, color, 2, 8, hierarchy);
		}
		imshow("contoursImgGray", contoursImg);

		return clusteredMat;
	}

	Mat clusterColorImageByKmeans()
	{
		assert(image.channels() != 1);

		int rows = image.rows;
		int cols = image.cols;
		int channels = image.channels();

		//保存聚類後的圖片  
		Mat clusteredMat(rows, cols, CV_8UC3);
		clusteredMat.setTo(Scalar::all(0));

		Mat pixels(rows*cols, 1, CV_32FC3); //pixels用於保存所有的灰度像素  
		pixels.setTo(Scalar::all(0));
		Mat centers(clusterCounts, 1, pixels.type());

		for (int i = 0; i < rows; ++i)
		{
			const uchar *idata = image.ptr<uchar>(i);
			float *pdata = pixels.ptr<float>(0);

			for (int j = 0; j < cols*channels; ++j)
			{
				pdata[i*cols*channels + j] = saturate_cast<float>(idata[j]);
			}
		}

		kmeans(pixels, clusterCounts, labels, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 0), 5, KMEANS_PP_CENTERS, centers);
		//kmeans(pixels, clusterCounts, labels, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 0), 5, KMEANS_RANDOM_CENTERS);
		cout << "Center of Color = " << endl;
		cout << centers.size() << endl;
		cout << centers << endl << endl;
		double bgrAvg[10];
		int bgrDisFlag[10];
		int cloudFlag[10];
		for (int i = 0; i < clusterCounts; i++)
		{
			float *clusterCenter = centers.ptr<float>(i);
			bgrAvg[i] = 0;
			bgrDisFlag[i] = 0;
			cloudFlag[i] = 0;
			for (int j = 0; j < 3; j++)
			{
				bgrAvg[i] += clusterCenter[j];
			}
			bgrAvg[i] /= 3;
			for (int j = 0; j < 3; j++)
			{
				if (abs(bgrAvg[i] - clusterCenter[j]) > 10)
				{
					//cout << "Dis = " << abs(bgrAvg[i] - clusterCenter[j]) << endl;
					bgrDisFlag[i] = 1;
				}
			}
			if (bgrDisFlag[i] == 0 && bgrAvg[i] >= 170)
			{
				cloudFlag[i] = 1;
			}
			cout << "bgrAvg = " << bgrAvg[i] << endl;
			cout << "bgrDisFlag = " << bgrDisFlag[i] << endl;
			cout << "cloudFlag = " << cloudFlag[i] << endl;
		}
		Scalar paintOn;
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
				}
				circle(clusteredMat, Point(j / channels, i), 1, paintOn);
				//circle(clusteredMat, Point(j / channels, i), 1, colorTab[labels.at<int>(i*cols + (j / channels))]);        //標記像素點的類別，顏色區分
			}
		}

		cvtColor(clusteredMat, edgeColor, COLOR_BGR2GRAY);
		//GaussianBlur(clusteredMat, edgeColor, Size(3, 3), 0);
		//Canny(clusteredMat, edgeColor, 50, 150, 3);
		//imshow("CannyColor", edgeColor);
		Mat contoursImg = image.clone();
		Mat rectImg = image.clone();
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		RNG rng(12345);
		findContours(edgeColor, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
		for (int i = 0; i<contours.size(); i++) {
			Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), 255);
			drawContours(contoursImg, contours, i, color, 2, 8, hierarchy);

			Rect contoursRect = boundingRect(contours[i]);
			if (contoursRect.width*contoursRect.height>rows*cols / 50)
				cv::rectangle(rectImg, contoursRect, cv::Scalar(0, 0, 255), 2);

		}
		imshow("contoursImgColor", contoursImg);
		imshow("rectImgColor", rectImg);

		return clusteredMat;
	}
};

int main()
{
	int clusterNumber = 4;
	String clusterNumber_str = to_string(clusterNumber);
	String loadLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\kmeansResult\\";
	String loadFileName = "0000c";
	String loadFileType = ".png";
	String saveLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\kmeansResult\\";
	String saveFileNameC = saveLocation + loadFileName + "k" + clusterNumber_str + "c_bw" + loadFileType;
	String saveFileNameG = saveLocation + loadFileName + "k" + clusterNumber_str + "g_bw" + loadFileType;
	Mat testImage = imread(loadLocation + loadFileName + loadFileType);
	Mat grayImage;
	cvtColor(testImage, grayImage, COLOR_BGR2GRAY);
	imwrite(saveLocation + loadFileName + loadFileType, testImage);
	imwrite(saveLocation + loadFileName + "g" + loadFileType, grayImage);
	//imshow("gray", grayImage);
	imshow("origin", testImage);
	if (testImage.empty())
	{
		return -1;
	}
	ClusterPixels clusterPix(testImage, clusterNumber);

	Mat colorResults = clusterPix.clusterColorImageByKmeans();
	Mat grayResult;// = clusterPix.clusterGrayImageByKmeans();

	if (!colorResults.empty())
	{
		//hconcat(testImage, colorResults, colorResults);
		imshow("clusterImage", colorResults);
		imwrite(saveFileNameC, colorResults);
	}

	if (!grayResult.empty())
	{
		//hconcat(testImage, grayResult, grayResult);
		imshow("grayCluster", grayResult);
		imwrite(saveFileNameG, grayResult);
	}

	if (waitKey() == 27)
		return 0;
}
