#include "opencv2/opencv.hpp"  
using namespace cv;
using namespace std;
#include <string>

String loadLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\test2\\cutOrder\\";
String loadFileType = ".jpg";
String saveLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\test2\\cutOrder\\lbp\\";
String nameBeforeNumber = "cut";
int startNum = 1;
int finishNum = 2983;

Mat LBP(String fileName)
{
	Mat src_image;
	src_image = imread(loadLocation + fileName + loadFileType);
	bool affiche = true;
	cv::Mat Image(src_image.rows, src_image.cols, CV_8UC1);  //建立一個與src_image等高等寬的單通道圖像Image
	cv::Mat lbp(src_image.rows, src_image.cols, CV_8UC1);    //建立一個與src_image等高等寬的單通道圖像lbp

	if (src_image.channels() == 3)
		cvtColor(src_image, Image, CV_BGR2GRAY);             //LBP只能處理灰度圖像，這裏如果傳過來的是彩色照片，要轉化為灰度圖

	unsigned center = 0;                                     //提取需要計算LBP值得中心點的灰度值
	unsigned center_lbp = 0;                                 //計算center處的LBP值
															 //計算LBP圖像
	for (int row = 1; row < Image.rows - 1; row++)
	{
		for (int col = 1; col < Image.cols - 1; col++)
		{
			center = Image.at<uchar>(row, col);
			center_lbp = 0;

			if (center <= Image.at<uchar>(row - 1, col - 1))
				center_lbp += 1;

			if (center <= Image.at<uchar>(row - 1, col))
				center_lbp += 2;

			if (center <= Image.at<uchar>(row - 1, col + 1))
				center_lbp += 4;

			if (center <= Image.at<uchar>(row, col - 1))
				center_lbp += 8;

			if (center <= Image.at<uchar>(row, col + 1))
				center_lbp += 16;

			if (center <= Image.at<uchar>(row + 1, col - 1))
				center_lbp += 32;

			if (center <= Image.at<uchar>(row + 1, col))
				center_lbp += 64;

			if (center <= Image.at<uchar>(row + 1, col + 1))
				center_lbp += 128;
			lbp.at<uchar>(row, col) = center_lbp;         //把center處計算好的LBP值存放在lbp圖像的相應位置
		}
	}
	if (affiche == true)
	{
		//cv::imshow("image LBP", lbp);
		imwrite(saveLocation + fileName + "lbp" + loadFileType, lbp);
		//waitKey(10);
		//cv::imshow("grayscale", Image);
		//waitKey(10);
	}

	/*else
	{
	cv::destroyWindow("image LBP");
	cv::destroyWindow("grayscale");
	}*/

	return lbp;
}
int main()
{
	String fileName = nameBeforeNumber + to_string(0);
	for (int i = startNum; i <= finishNum; i++)
	{
		cout << i;
		fileName = nameBeforeNumber + to_string(i);
		LBP(fileName);
		cout << " OK!" << endl;
	}
	return 0;
}