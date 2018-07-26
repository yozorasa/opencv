#include "opencv2/opencv.hpp"
#include <string>
#include <iostream>
using namespace cv;
using namespace std;

String loadLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\test3\\cutOrder\\";
String loadLocation2 = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\test3\\lbp\\";
String fileType = ".jpg";
String saveLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\lbpRename\\cloud\\";
String saveLocation2 = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\lbpRename\\other\\";
String loadFirstName = "cut";
String loadLastName = "lbp";
//String saveName = "k_";
int imageStart = 1;
int imageFinish = 3025;
int saveCloudNumberFrom = 498-1;
int saveOtherNumberFrom = 252-1;

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
			imwrite(saveLocation + to_string(z+saveCloudNumberFrom) + "_roi" + fileType, image);
			imwrite(saveLocation + to_string(z+saveCloudNumberFrom) + "_lbp" + fileType, lbp);
		}
		else {
			cout << ">>>>>>>>>>>>>>>>> something" << endl;
			imwrite(saveLocation2 + to_string(z+saveOtherNumberFrom) + "_roi" + fileType, image);
			imwrite(saveLocation2 + to_string(z+saveOtherNumberFrom) + "_lbp" + fileType, lbp);
		}
	}
	return 0;
}