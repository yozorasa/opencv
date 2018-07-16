#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>

using namespace cv;
using namespace std;

String loadLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\cloud\\img 1075-1511 (russiaMoskva)\\kmeansBinary\\convexHull\\";
String fileType = ".jpg";
String saveLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\cloud\\img 1075-1511 (russiaMoskva)\\kmeansBinary\\convexHull\\size128\\";
String loadName = "cH";
String saveName = "p";
int imageStart = 1075;
int imageFinish = 1511;
int ROISize = 128;

int main()
{
	Mat img, subImg;
	int x, y, col, row, key;
	
	for(int z=imageStart;z<=imageFinish;z++)
	{
		cout << z;
		img = cv::imread(loadLocation + to_string(z) + loadName + fileType);
		if (!img.data) // Check for invalid input
		{
			cout << "Could not open or find the img" << std::endl;
			return -1;
		}
		x = 0;
		y = 0;
		col = img.cols / ROISize;
		col = (img.cols % ROISize) > (ROISize / 2) ? (col + 1) : col;
		row = img.rows / ROISize;
		row = (img.rows % ROISize) > (ROISize / 2) ? (row + 1) : row;
		int piece = 1;
		for (int i = 0; i < row; i++)
		{
			y = i * ROISize;
			if (i == row - 1)
				y = img.rows - ROISize;
			for (int j = 0; j < col; j++)
			{
				x = j * ROISize;
				if (j == col - 1)
					x = img.cols - ROISize;
				subImg = img(Rect(x, y, ROISize, ROISize));
				imwrite(saveLocation+to_string(z)+saveName+to_string(piece)+fileType, subImg);
				piece++;

			}
		}
		cout << " OK!" << endl;
	}

	return 0;
}