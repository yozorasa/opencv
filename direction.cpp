#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>
using namespace cv;
using namespace std;

String loadLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\cloud\\img 1075-1511 (russiaMoskva)\\kmeansBinary\\convexHull\\size256x128\\";
String loadLocation2 = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\cloud\\img 1075-1511 (russiaMoskva)\\kmeansBinary\\convexHull\\size256x128(origin)\\";
String fileType = ".jpg";
String saveLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\size256x128\\newJudge\\convexHull\\";
String saveLocation2 = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\size256x128\\newJudge\\origin\\";
//String loadName = "";
//String saveName = "";
int imageStart = 1;
int imageFinish = 60000;
int need = 10000;
float blackThreshold = 0.95;
float whiteThreshold = 0.95;
float turnThreshold = 0.0005;
int row1Weight = 3;
int row2Weight = 7;
int sizeWeight;

int bwJudge(Mat image) {
	if (!image.data)
		return -1;

	int row = image.rows;
	int col = image.cols;
	int size = row * col;
	sizeWeight = size / 2 * row1Weight + size / 2 * row2Weight;
	int cnt = 0, left = 0, center = 0, right = 0;
	for (int i = 0; i < row; i++) {
		for (int j = 0; j<col; j++) {
			if ((int)(image.at<Vec3b>(i, j)[0]) == 0 && (int)(image.at<Vec3b>(i, j)[1]) == 0 && (int)(image.at<Vec3b>(i, j)[2]) == 0) {
				if (i<row / 2)
					cnt += row1Weight;
				else
					cnt += row2Weight;

				if (j < col * 2 / 4) {
					if (i<row / 2)
						left += row1Weight;
					else
						left += row2Weight;
				}
				if (j >= col * 1 / 4 && j < col * 3 / 4) {
					if (i<row / 2)
						center += row1Weight;
					else
						center += row2Weight;
				}
				if (j >= col * 2 / 4) {
					if (i<row / 2)
						right += row1Weight;
					else
						right += row2Weight;
				}
			}
		}
	}

	float leftPercentage = (float)left / (sizeWeight);
	float centerPercentage = (float)center / (sizeWeight);
	float rightPercentage = (float)right / (sizeWeight);
	float blackPercentage = (float)cnt / (sizeWeight);

	cout << "  cnt = " << blackPercentage << "  left = " << leftPercentage << "  center = " << centerPercentage << "  right = " << rightPercentage;

	if (cnt > blackThreshold * sizeWeight) {
		cout << " --> Black  ";
		return -1;
	}
	else if ((sizeWeight - cnt) > whiteThreshold * sizeWeight) {
		cout << " --> White  ";
		return -2;
	}
	else {
		cout << " --> Judge Please" << endl;
		cout << " left " << left << " right " << right << " size " << size << "sizeWeight " << sizeWeight << endl;
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
	//ONLY SAVE USEFUL IMAGE
	for (int z = imageStart, use = 1; z <= imageFinish && use <= need; z++) {
		cout << "image " + to_string(z) + " SAVE: " + to_string(use);
		Mat image = imread(loadLocation + to_string(z) + fileType);

		int flag = bwJudge(image);
		cout << "flag = " << flag << endl;
		if (flag >= 0) {
			Mat image2 = imread(loadLocation2 + to_string(z) + fileType);
			imwrite(saveLocation + to_string(use) + "_" + to_string(flag) + fileType, image);
			imwrite(saveLocation2 + to_string(use) + "_" + to_string(flag) + fileType, image2);
			use++;
		}
		//imshow("img", image);
		waitKey(0);
	}
	/*
	for (int z = imageStart; z <= imageFinish; z++) {
		cout << "image " + to_string(z);
		Mat image = imread(loadLocation + to_string(z) + fileType);
		Mat image2 = imread(loadLocation2 + to_string(z) + fileType);
		int flag = bwJudge(image);
		cout << "flag = " << flag << endl;
		//imshow("img", image);
		imwrite(saveLocation + to_string(z) + "_" + to_string(flag) + fileType, image);
		imwrite(saveLocation2 + to_string(z) + "_" + to_string(flag) + fileType, image2);
		waitKey(0);
	}
	*/

	return 0;
}