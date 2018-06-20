#include <opencv2/opencv.hpp>
using namespace cv;
#include<iostream>
using namespace std;
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

public:
	ClusterPixels() :clusterCounts(0) {}
	ClusterPixels(const Mat& src, int clusters = 5) :clusterCounts(clusters) { image = src.clone(); }

	void setImage(const Mat& src) { image = src.clone(); };
	void setClusters(int clusters) { clusterCounts = clusters; }

	Mat getLabels() { return labels; };      //返回聚類後的標籤  

	Mat clusterGrayImageByKmeans()
	{
		//轉換成灰度圖  
		if (image.channels() != 1)
			cvtColor(image, image, COLOR_BGR2GRAY);

		int rows = image.rows;
		int cols = image.cols;

		//保存聚類後的圖片  
		Mat clusteredMat(rows, cols, CV_8UC3);
		clusteredMat.setTo(Scalar::all(0));

		Mat pixels(rows*cols, 1, CV_32FC1); //pixels用於保存所有的灰度像素

		for (int i = 0; i < rows; ++i)
		{
			const uchar *idata = image.ptr<uchar>(i);
			float *pdata = pixels.ptr<float>(0);
			for (int j = 0; j < cols; ++j)
			{
				pdata[i*cols + j] = idata[j];
			}
		}

		kmeans(pixels, clusterCounts, labels, TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 0), 5, KMEANS_PP_CENTERS);

		for (int i = 0; i < rows; ++i)
		{
			for (int j = 0; j < cols; ++j)
			{
				circle(clusteredMat, Point(j, i), 1, colorTab[labels.at<int>(i*cols + j)]);        //標記像素點的類別，顏色區分  
			}
		}

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

		for (int i = 0; i < rows; ++i)
		{
			const uchar *idata = image.ptr<uchar>(i);
			float *pdata = pixels.ptr<float>(0);

			for (int j = 0; j < cols*channels; ++j)
			{
				pdata[i*cols*channels + j] = saturate_cast<float>(idata[j]);
			}
		}

		kmeans(pixels, clusterCounts, labels, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 0), 5, KMEANS_PP_CENTERS);
		//kmeans(pixels, clusterCounts, labels, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 0), 5, KMEANS_RANDOM_CENTERS);

		for (int i = 0; i < rows; ++i)
		{
			for (int j = 0; j < cols*channels; j += channels)
			{
				circle(clusteredMat, Point(j / channels, i), 1, colorTab[labels.at<int>(i*cols + (j / channels))]);        //標記像素點的類別，顏色區分  
			}
		}

		return clusteredMat;
	}
};

int main()
{
	int clusterNumber = 3;
	String clusterNumber_str = to_string(clusterNumber);
	String loadLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\kmeansResult\\";
	String loadFileName = "0000c";
	String loadFileType = ".png";
	String saveLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\kmeansResult\\";
	String saveFileNameC = saveLocation + loadFileName + "k"+ clusterNumber +"c" + loadFileType;
	String saveFileNameG = saveLocation + loadFileName + "k"+ clusterNumber +"g" + loadFileType;
	Mat testImage = imread(loadLocation + loadFileName + loadFileType);
	Mat grayImage;
	cvtColor(testImage, grayImage, COLOR_BGR2GRAY);
	imwrite(saveLocation + loadFileName + loadFileType, testImage);
	imwrite(saveLocation + loadFileName + "g" + loadFileType, grayImage);
	//imshow("gray", grayImage);
	if (testImage.empty())
	{
		return -1;
	}
	ClusterPixels clusterPix(testImage, clusterNumber);

	Mat colorResults = clusterPix.clusterColorImageByKmeans();
	Mat grayResult = clusterPix.clusterGrayImageByKmeans();

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
