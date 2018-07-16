#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>
using namespace cv;
using namespace std;

String loadLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\waitForTag\\";
String fileType = ".jpg";
String saveLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\tagOn\\";
//String loadName = "";
//String saveName = "";
int imageStart = 1;
int imageFinish = 20;
float blackThreshold = 0.85;
float whiteThreshold = 0.9;

int bwJudge(Mat image){
	if (!image.data)
		return -1;

	int row = image.rows;
	int col = image.cols;
	int size = row * col;
	int cnt = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j<col; j++) {
			if (image.channels() == 3) {
				if ((int)(image.at<Vec3b>(i, j)[0]) == 0 && (int)(image.at<Vec3b>(i, j)[1]) == 0 && (int)(image.at<Vec3b>(i, j)[2]) == 0) {
					cnt++;
				}
			}
			else if (image.channels() == 1) {
				if ((int)(image.at<uchar>(i, j)) != 0) {
					cnt++;
				}
			}
		}
	}
	cout << "  cnt = " << cnt;
	if (cnt >= blackThreshold * size) {
		cout << " --> Black" << endl;
		return 1;
	}
	else if ((size - cnt) >= whiteThreshold * size) {
		cout << " --> White" << endl;
		return 2;
	}
	else {
		cout << " --> Judge Please" << endl;
		return 0;
	}
}

int main()
{
	for (int z = imageStart; z <= imageFinish; z++) {
		cout << "image " + to_string(z);
		Mat image = imread(loadLocation + to_string(z) + fileType);

		int flag = bwJudge(image);
		cout << "flag = " << flag << endl;
		
		imshow("img", image);
		waitKey(0);
	}
	return 0;
}