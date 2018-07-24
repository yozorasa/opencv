#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>
using namespace cv;
using namespace std;

String loadLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\waitForTag\\";
String fileType = ".jpg";
String saveLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\tagOn2\\";
//String loadName = "";
//String saveName = "";
int imageStart = 1;
int imageFinish = 200000;
int need = 10000;
float blackThreshold = 0.85;
float whiteThreshold = 0.9;
float turnThreshold = 0.1;

int bwJudge(Mat image) {
	if (!image.data)
		return -1;

	int row = image.rows;
	int col = image.cols;
	int size = row * col;
	int cnt = 0, left = 0, center = 0, right = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j<col; j++) {
			if (image.channels() == 3) {
				if ((int)(image.at<Vec3b>(i, j)[0]) == 255 && (int)(image.at<Vec3b>(i, j)[1]) == 255 && (int)(image.at<Vec3b>(i, j)[2]) == 255) {
					cnt++;
					if (j<col / 3)
						left++;
					else if (j >= col * 2 / 3)
						right++;
					else
						center++;
				}
			}
			else if (image.channels() == 1) {
				if ((int)(image.at<uchar>(i, j)) == 255) {
					cnt++;
					if (j<col / 3)
						left++;
					else if (j >= col * 2 / 3)
						right++;
					else
						center++;
				}
			}
		}
	}

	float leftPercentage = (float)left / size;
	float centerPercentage = (float)center / size;
	float rightPercentage = (float)right / size;
	float cloudPercentage = (float)cnt / size;

	cout << "  cnt = " << cloudPercentage << "  left = " << leftPercentage << "  center = " << centerPercentage << "  right = " << rightPercentage;

	if (cnt >= blackThreshold * size) {
		cout << " --> Black" << endl;
		return -1;
	}
	else if ((size - cnt) >= whiteThreshold * size) {
		cout << " --> White" << endl;
		return -2;
	}
	else {
		cout << " --> Judge Please" << endl;
		if (left >= right && abs(leftPercentage - centerPercentage) > turnThreshold) {
			cout << " --> TURN LEFT";
			return 1;
		}
		else if (left<right && abs(rightPercentage - centerPercentage) > turnThreshold) {
			cout << " --> TURN RIGHT";
			return 2;
		}
		else {
			return 0;
		}
	}
	return 0;
}

int main()
{
	for (int z = imageStart, int use = 1; z <= imageFinish && use <= need; z++) {
		cout << "image " + to_string(z);
		Mat image = imread(loadLocation + to_string(z) + fileType);

		int flag = bwJudge(image);
		cout << "flag = " << flag << endl;
		if (flag >= 0)
			imwrite(saveLocation + +"k_" + to_string(use++) + "_" + to_string(flag) + fileType, image);
		//imshow("img", image);
		waitKey(0);
	}
	return 0;
}