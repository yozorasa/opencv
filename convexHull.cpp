#include <opencv2/opencv.hpp>  
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
 
using namespace cv;
using namespace std;
 
int imageStart = 1;
int imageFinish = 452;
String loadLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\cloud\\img 1-452 (taiwan)\\kmeansBinary\\";
String loadFileType = ".jpg";
String saveLocation = "C:\\Users\\yozorasa\\Documents\\GraduateSchool\\space\\cloud\\img 1-452 (taiwan)\\kmeansBinary\\convexHull\\";
String loadName = "jB";
String saveName = "cH";

int main()
{
	
	for(int i=imageStart;i<=imageFinish;i++){
		cout << i;
		Mat src; 
		Mat src_gray;
		src = imread(loadLocation+to_string(i)+loadName+loadFileType);
		//resize(src, src, Size(640,480), 0, 0, INTER_CUBIC);
		cvtColor( src, src_gray, CV_BGR2GRAY );
		blur( src_gray, src_gray, Size(3,3) ); 
		//namedWindow( "Source", CV_WINDOW_AUTOSIZE );
		//imshow( "Source", src );



		// Convex Hull implementation
		Mat src_copy = src.clone();
		Mat threshold_output;
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		// Find contours
		threshold( src_gray, threshold_output, 200, 255, THRESH_BINARY );
		findContours( threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

		// Find the convex hull object for each contour
		vector<vector<Point> >hull( contours.size() );
		for( int i = 0; i < contours.size(); i++ )
		{  convexHull( Mat(contours[i]), hull[i], false ); }

		// Draw contours + hull results
		RNG rng;
		Mat drawing = Mat::zeros( threshold_output.size(), CV_8UC3 );
		for( int i = 0; i< contours.size(); i++ )
		{
			//Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
			Scalar white = Scalar(255, 255, 255);
			//drawContours( drawing, contours, i, white, 1, 8, vector<Vec4i>(), 0, Point() );
			//drawContours( drawing, hull, i, white, 1, 8, vector<Vec4i>(), 0, Point() );
			drawContours( drawing, hull, i, white, CV_FILLED, 8, vector<Vec4i>(), 0, Point() );
		}

		// Show in a window
		//namedWindow( "Hull demo", CV_WINDOW_AUTOSIZE );
		//imshow( "Hull demo", drawing );
		imwrite(saveLocation+to_string(i) + saveName+loadFileType, drawing);
		cout << " OK!" << endl;
	}
	waitKey(0);
	return(0);
}