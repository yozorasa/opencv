#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;
using namespace ml;

String record = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\lbpRename\\";
String loadLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\lbpRename\\cloud\\";
String loadLocationNot = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\lbpRename\\other\\";
String fileType = ".jpg";
int cloudAmount = 498;
int otherAmount = 251;
float histTemp[256] = { 0 };

int histogram(Mat lbp, Mat roi) {
    int hcount[256] = { 0 };
    float histogram[256] = { 0 };
    int rows = lbp.rows;
    int cols = lbp.cols;
    int pixelCount = 0;
    for (int r = 0; r < rows; ++r) {
        const uchar *lbpdata = lbp.ptr<uchar>(r);
        const uchar *roidata = roi.ptr<uchar>(r);
        for (int c = 0; c < cols; ++c)
        {
            if (!(roidata[3 * c] == 255 && roidata[3 * c + 1] == 0 && roidata[3 * c + 2] == 255)) {
                hcount[lbpdata[c]]++;
                pixelCount++;
            }
            else {
                //cout << "MAGENTA" << endl;
            }
        }
    }
    lbp.release();
    roi.release();
    for (int z = 0; z<256 && pixelCount != 0; z++) {
        //cout << "z = " << z << " hcount = ";
        //cout << hcount[z];
        histTemp[z] = (float)hcount[z] / pixelCount;
        //cout << " hist = " << histTemp[z] << " pixel = " << pixelCount << endl;
    }
    //cout << "Have " << pixelCount << "pixels." << endl;
    return 0;
}

bool classification(Mat src) {
    Ptr<SVM> svm = SVM::create();
    svm = SVM::load(record + "SVMresult.xml");
    int response = svm->predict(src);
    return response;
}


int main() {
    Mat lbp, roi;
    for (int i = 1; i <= otherAmount; i++){
        lbp = imread(loadLocationNot + to_string(i) + "_lbp" + fileType, 0);
        roi = imread(loadLocationNot + to_string(i) + "_roi" + fileType);
        //imshow("lbp", lbp);
        //imshow("roi", roi);
        histogram(lbp, roi);
        float testData[256] = { 0 };
        for (int i = 0; i < 256; i++) {
            testData[i] = histTemp[i];
            //cout << testData[i] << endl;
        }
        Mat src(1, 256, CV_32FC1, testData);
        bool flag = classification(src);
        cout << "flag = " << flag << endl;
    }
    //waitKey(0);
    //system("pause");
    return 0;
}