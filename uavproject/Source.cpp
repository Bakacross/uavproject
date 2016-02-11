#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2\video\tracking.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace cv;
using namespace std;

static const double pi = 3.14159265358979323846;
int maxCorners = 50;
int thresholdCorners = 700;
double qualityLevel = 0.01;
double minDistance = 10;
int blockSize = 3;
bool useHarrisDetector = false;
double k_harris = 0.04;
double alpha = 1;

inline static double square(int a)

{

	return a * a;

}

void refresh_features(Mat gray, vector<Point2f> &flow_c);

int main(int argc, char* argv[])
{
	Mat image1, image2, image_gray1, image_gray2, image2_average;
	vector< Point2f > corners, flow_corners, new_corners;
	int initial_nb;
	Mat mask(image_gray1.size(), CV_8UC1, Scalar(255));
	vector<double> hypotenuse, hypotenuseLast;
	bool first = 1;
	vector<double> angle;
	vector<uchar> status;
	vector<float> err;
	double mean_hypotenuse;
	double color;
	Size optical_flow_window = cvSize(3, 3);
	TermCriteria optical_flow_termination_criteria = cvTermCriteria(CV_TERMCRIT_ITER | CV_TERMCRIT_EPS, 50, 0.01);
	int line_thickness = 1;
	CvPoint p, q;
	CvScalar line_color;
	line_color = CV_RGB(255, 0, 0);
	RNG rng(12345);

	//VideoCapture cap(0); //capture the video from webcam
	VideoCapture cap("dronecourse360p.mp4");
	int fps = cap.get(CV_CAP_PROP_FPS);
	cap.set(CV_CAP_PROP_POS_MSEC, 5000);

	if (!cap.isOpened())  // if not success, exit program
	{
		cout << "Cannot open the webcam" << endl;
		return -1;
	}

	namedWindow("Flow", CV_WINDOW_AUTOSIZE);
	namedWindow("Flow average", CV_WINDOW_AUTOSIZE);
	//namedWindow("Depth Estimation", CV_WINDOW_AUTOSIZE);
	//namedWindow("Mask", CV_WINDOW_AUTOSIZE);

	cap.read(image1); // read a new frame from video
	cvtColor(image1, image_gray1, CV_BGR2GRAY);
	goodFeaturesToTrack(image_gray1, corners, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k_harris);
	initial_nb = corners.size();

	while (1) {

		cap.read(image2); // read a new frame from video
		image2.copyTo(image2_average);
		Mat bg(image2.size(), CV_8UC3, Scalar(255, 255, 255));
		cvtColor(image2, image_gray2, CV_BGR2GRAY);
		calcOpticalFlowPyrLK(image_gray1, image_gray2, corners, flow_corners, status, err, optical_flow_window, 5, optical_flow_termination_criteria, 0, 0.001);
		size_t i, k;
		if (!flow_corners.empty()) {
			hypotenuse.clear();
			angle.clear();
			mean_hypotenuse = 0;
			for (i = k = 0; i < flow_corners.size(); i++)

			{
				// if LK failed don't draw and delete it from vector corners
				if (!status[i] || err[i] > 50) {
					continue;
				}

				angle.push_back(atan2(corners[i].y - flow_corners[i].y, corners[i].x - flow_corners[i].x));
				hypotenuse.push_back(sqrt(square(flow_corners[i].x - corners[i].x) + square(flow_corners[i].y - corners[i].y)));
				mean_hypotenuse += sqrt(square(flow_corners[i].x - corners[i].x) + square(flow_corners[i].y - corners[i].y));
				corners[k] = corners[i];
				flow_corners[k++] = flow_corners[i];


			}
			flow_corners.resize(k);
			corners.resize(k);
			mean_hypotenuse /= hypotenuse.size();
			
		}


		if (!hypotenuse.empty()) {
			for (i = 1; i < hypotenuse.size(); i++) { //begin at i = 1 !

				//color = hypotenuse[i] * 255 / (hypotenuse[i] + 3);
				//circle(image2, flow_corners[i], 3, Scalar(0, 255 - color, color), -1, 8);

				/* DRAW ARROW */

				p.x = (int)corners[i].x;
				p.y = (int)corners[i].y;
				q.x = (int)flow_corners[i].x;
				q.y = (int)flow_corners[i].y;

				q.x = (int)(p.x - 3 * hypotenuse[i] * cos(angle[i]));
				q.y = (int)(p.y - 3 * hypotenuse[i] * sin(angle[i]));

				line(image2, p, q, line_color, line_thickness, CV_AA, 0);

				p.x = (int)(q.x + 3 * cos(angle[i] + pi / 4));
				p.y = (int)(q.y + 3 * sin(angle[i] + pi / 4));

				line(image2, p, q, line_color, line_thickness, CV_AA, 0);

				p.x = (int)(q.x + 3 * cos(angle[i] - pi / 4));
				p.y = (int)(q.y + 3 * sin(angle[i] - pi / 4));

				line(image2, p, q, line_color, line_thickness, CV_AA, 0);

				//

				if (first) {
					first = 0;
					hypotenuseLast = hypotenuse;
				}
				else {

					p.x = (int)corners[i].x;
					p.y = (int)corners[i].y;
					q.x = (int)flow_corners[i].x;
					q.y = (int)flow_corners[i].y;

					q.x = (int)(p.x - 3 * (alpha*hypotenuse[i] + (1 - alpha)*hypotenuseLast[i]) * cos(angle[i]));
					q.y = (int)(p.y - 3 * (alpha*hypotenuse[i] + (1 - alpha)*hypotenuseLast[i]) * sin(angle[i]));

					line(image2_average, p, q, line_color, line_thickness, CV_AA, 0);

					p.x = (int)(q.x + 3 * cos(angle[i] + pi / 4));
					p.y = (int)(q.y + 3 * sin(angle[i] + pi / 4));

					line(image2_average, p, q, line_color, line_thickness, CV_AA, 0);

					p.x = (int)(q.x + 3 * cos(angle[i] - pi / 4));
					p.y = (int)(q.y + 3 * sin(angle[i] - pi / 4));

					line(image2_average, p, q, line_color, line_thickness, CV_AA, 0);
					if (hypotenuse[i] - hypotenuseLast[i] != 0) { cout << hypotenuse[i] - hypotenuseLast[i] << endl; }
					hypotenuseLast = hypotenuse;
				}	
			}
		}

		imshow("Flow", image2);
		imshow("Flow average", image2_average);
		//imshow("Depth Estimation", bg);

		if (flow_corners.size() < thresholdCorners)
		{
			refresh_features(image_gray2, flow_corners);
			initial_nb = flow_corners.size();
		}

		swap(image_gray1, image_gray2);
		swap(corners, flow_corners);
		waitKey(1000 / fps);
		//waitKey(0);
		if (waitKey(10) == 27) //wait for 'esc' key press for 30 ms. If 'esc' key is pressed, break loop
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
		if (waitKey(10) == 'n') {
			cap.set(CV_CAP_PROP_POS_MSEC, cap.get(CV_CAP_PROP_POS_MSEC) + 5000);
		}
	}
}

void refresh_features(Mat gray, vector<Point2f> &flow_c)
{
	//Mat mask(image_gray.size(), CV_8UC1, Scalar(255));
	int size_mask = 40;
	vector<Point2f> new_corners;
	Mat mask(gray.size(), CV_8UC1, Scalar(255));
	if (!flow_c.empty()) {
		int x_mask, y_mask, w_mask, h_mask;
		size_t j;
		for (j = 0; j < flow_c.size(); j++) {

			if (flow_c[j].x < 0) {
				flow_c[j].x = 0;
			}
			else if (flow_c[j].x > gray.size().width) {
				flow_c[j].x = gray.size().width;
			}
			if (flow_c[j].y < 0) {
				flow_c[j].y = 0;
			}
			else if (flow_c[j].y > gray.size().height) {
				flow_c[j].y = gray.size().height;
			}

			x_mask = flow_c[j].x - size_mask / 2;
			y_mask = flow_c[j].y - size_mask / 2;
			w_mask = size_mask;
			h_mask = size_mask;

			if (x_mask < 0) {
				x_mask = 0;
				w_mask = flow_c[j].x + size_mask / 2;
			}
			else if (x_mask + w_mask > gray.size().width) {
				w_mask = gray.size().width - x_mask;
			}

			if (y_mask < 0) {
				y_mask = 0;
				h_mask = flow_c[j].y + size_mask / 2;
			}
			else if (y_mask + h_mask > gray.size().height) {
				h_mask = gray.size().height - y_mask;
			}
			mask(Rect(x_mask, y_mask, w_mask, h_mask)).setTo(Scalar(0));
		}
	}
	goodFeaturesToTrack(gray, new_corners, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k_harris);
	flow_c.insert(flow_c.end(), new_corners.begin(), new_corners.end());

	//imshow("Mask", mask);
}