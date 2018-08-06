#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>
using namespace cv;
using namespace std;

String loadLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\cloud\\img 1075-1511 (russiaMoskva)\\kmeansBinary\\convexHull\\";
String fileType = ".jpg";
String saveLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\cloud\\img 1075-1511 (russiaMoskva)\\kmeansBinary\\convexHull\\size256x128\\";
String loadName = "cH";
String saveName = "p";
int imageStart = 1075;
int imageFinish = 1511;
int rowROISize = 128;
int colROISize = 256;

int main()
{
	Mat img, subImg;
	int x, y, col, row, key;
	int count = 1;
	for (int z = imageStart; z <= imageFinish; z++)
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
		col = img.cols / colROISize;
		col = (img.cols % colROISize) > (colROISize / 2) ? (col + 1) : col;
		row = img.rows / rowROISize;
		row = (img.rows % rowROISize) > (rowROISize / 2) ? (row + 1) : row;
		int piece = 1;
		for (int i = 0; i < row; i++)
		{
			y = i * rowROISize;
			if (i == row - 1)
				y = img.rows - rowROISize;
			for (int j = 0; j < col; j++)
			{
				x = j * colROISize;
				if (j == col - 1)
					x = img.cols - colROISize;
				subImg = img(Rect(x, y, colROISize, rowROISize));
				//imwrite(saveLocation+to_string(z)+saveName+to_string(piece)+fileType, subImg);
				//cout << saveLocation + to_string(count) + fileType << endl;
				imwrite(saveLocation + to_string(count++) + fileType, subImg);
				if (count > 60000)
					return 0;
				piece++;

			}
		}
		cout << " OK!" << endl;
	}

	return 0;
}