#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>
using namespace cv;
using namespace std;

String loadLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\test2\\cutOrder\\";
String loadLocation2 = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\test2\\cutOrder\\lbp\\";
String fileType = ".jpg";
String saveLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\test2\\cutOrder\\cutTag\\";
String saveLocation2 = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\test2\\cutOrder\\lbpTag\\";
String loadFirstName = "cut";
String loadLastName = "lbp";
String saveName = "k_";
int imageStart = 1;
int imageFinish = 500;

int main()
{
	
	int cloudKey = 122;
	int somethingKey = 120;
	int key = 0;
	for (int z = imageStart; z <= imageFinish; z++) {
		cout << "image " + to_string(z);
		Mat image = imread(loadLocation + loadFirstName + to_string(z) + fileType);
		Mat lbp = imread(loadLocation2 + loadFirstName + to_string(z) + loadLastName + fileType);
		imshow("img", image);
		//imshow("lbp", lbp);
		cout << " Enter: ";
		key = waitKey(0);
		cout << key;
		while (key != cloudKey && key != somethingKey) {
			cout << endl << "Enter: ";
			key = waitKey(0);
			cout << key;
		}
		if (key == cloudKey) {
			cout << ">>>>>>>>>>>>>>>>> cloud" << endl;
			imwrite(saveLocation + loadFirstName + "_" + to_string(z) + "_1" + fileType, image);
			imwrite(saveLocation2 + saveName + "_" + to_string(z) + "_1" + fileType, lbp);
		}
		else {
			cout << ">>>>>>>>>>>>>>>>> something" << endl;
			imwrite(saveLocation + loadFirstName + "_" + to_string(z) + "_0" + fileType, image);
			imwrite(saveLocation2 + saveName + "_" + to_string(z) + "_0" + fileType, lbp);
		}
	}
	return 0;
}