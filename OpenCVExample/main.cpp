/*
 * This code is provided as part of "A Practical Introduction to Computer Vision with OpenCV"
 * by Kenneth Dawson-Howe � Wiley & Sons Inc. 2014.  All rights reserved.
 */
#include "Utilities.h"
#include <iostream>
#include <fstream>
#include <list>

#define REGENTHOUSE_IMAGE_INDEX 0
#define CAMPANILE1_IMAGE_INDEX 1
#define CAMPANILE2_IMAGE_INDEX 2
#define CREST_IMAGE_INDEX 3
#define OLDLIBRARY_IMAGE_INDEX 4
#define WINDOW1_IMAGE_INDEX 5
#define WINDOW2_IMAGE_INDEX 6
#define WINDOW1_LOCATIONS_IMAGE_INDEX 7
#define WINDOW2_LOCATIONS_IMAGE_INDEX 8
#define BIKES_IMAGE_INDEX 9
#define PEOPLE2_IMAGE_INDEX 10
#define ASTRONAUT_IMAGE_INDEX 11
#define PEOPLE1_IMAGE_INDEX 12
#define PEOPLE1_SKIN_MASK_IMAGE_INDEX 13
#define SKIN_IMAGE_INDEX 14
#define CHURCH_IMAGE_INDEX 15
#define FRUIT_IMAGE_INDEX 16
#define COATS_IMAGE_INDEX 17
#define STATIONARY_IMAGE_INDEX 18
#define PETS124_IMAGE_INDEX 19
#define PETS129_IMAGE_INDEX 20
#define PCB_IMAGE_INDEX 21
#define LICENSE_PLATE_IMAGE_INDEX 22
#define BICYCLE_BACKGROUND_IMAGE_INDEX 23
#define BICYCLE_MODEL_IMAGE_INDEX 24
#define NUMBERS_IMAGE_INDEX 25
#define GOOD_ORINGS_IMAGE_INDEX 26
#define BAD_ORINGS_IMAGE_INDEX 27
#define UNKNOWN_ORINGS_IMAGE_INDEX 28

#define SURVEILLANCE_VIDEO_INDEX 0
#define BICYCLES_VIDEO_INDEX 1
#define ABANDONMENT_VIDEO_INDEX 2
#define DRAUGHTS_VIDEO_INDEX 3

#define HAAR_FACE_CASCADE_INDEX 0

#include <chrono>
#include <thread>

int liveVideo()
{
	// Create a VideoCapture object and use camera to capture the video
		VideoCapture cap(1);
		// Check if camera opened successfully
		if (!cap.isOpened()) {
				cout << "Error opening video stream" << endl;
				return -1;
		}

		cap.set(cv::CAP_PROP_FRAME_WIDTH, 1024); //3264);
		cap.set(cv::CAP_PROP_FRAME_HEIGHT, 768); //2448);
		// Default resolutions of the frame are obtained.The default resolutions are system dependent.
		int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
		int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

		// Define the codec and create VideoWriter object.The output is stored in 'outcpp.avi' file.
		VideoWriter video("outcpp.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 20, Size(frame_width, frame_height));
		chrono::system_clock::time_point capture_time = chrono::system_clock::now();
		while (1) {
			this_thread::sleep_until(capture_time);
			capture_time = capture_time + 1s;
			Mat frame;
			// Capture frame-by-frame

			cap >> frame;
			// If the frame is empty, break immediately
			if (frame.empty())
				break;
			// Write the frame into the file 'outcpp.avi'
			video.write(frame);
			// Display the resulting frame   
			imshow("Frame", frame);
			// Press  ESC on keyboard to  exit
			char c = (char)waitKey(1);
			if (c == 27)
				break;
		}

	// When everything done, release the video capture and write object
	cap.release();
	video.release();
	//destroyAllWindows();
	return 0;
}

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <iostream>
using namespace cv;
using namespace std;

/*
void main() {
	
	string path = "Media/CarSpeedTest1EmptyFrame.jpg";

	Mat img = imread(path);
	Mat grayscale_img;
	Mat binary_img;

	
	namedWindow("Show Binary");
	cvtColor(img, grayscale_img, COLOR_BGR2GRAY);
	imshow("Image", grayscale_img);
	threshold(grayscale_img, binary_img, 100, 255, THRESH_BINARY);
	imshow("Show Binary", binary_img);


	string filename("Media/StayingInLane_MPEG4 (1).avi");
	VideoCapture video; 
	video.open(filename);
	if (!video.isOpened())
	{
		cout << "Cannot open video file: " << filename << endl;
			//		return -1;
	}

	namedWindow("MyVideo");
	namedWindow("MyBinaryVideo");

	while (1)
	{
		Mat frame;

		bool bSuccess = video.read(frame); // read a new frame from video

		if (!bSuccess)
		{
			cout << "Cannot read a frame from video stream" << endl;
			break;
		}

		Mat grayscale;
		cvtColor(frame, grayscale, COLOR_BGR2GRAY);

		imshow("MyVideo", grayscale);
		threshold(grayscale, binary_img, 90, 255, THRESH_BINARY);
		imshow("MyBinaryVideo", binary_img);

		if (waitKey(30) == 27)
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}

	waitKey(0);


}*/


////// ****************************
////// KENS CODE HERE!!!!!!!
////// *************************


int main(int argc, const char** argv)
{
//	liveVideo();
	char* file_location = "Media/";
	char* image_files[] = {

		"TrinityRegentHouse.jpg", //0
		"TrinityCampanile1.jpg",
		"TrinityCampanile3.jpg",
		"TrinityCrest.jpg",
		"TrinityOldLibrary.jpg",
	    "TrinityWindow1.jpg", //5
	    "TrinityWindow2.jpg",
	    "TrinityWindow1Locations.png",
	    "TrinityWindow2Locations.png",
		"TrinityBikes1.jpg",
		"People2.jpg", //10
		"Astronaut2.jpg",
		"People1.jpg",
		"People1SkinMask.jpg",
		"SkinSamples.jpg",\
		"Church.jpg", //15
		"FruitStall.jpg",
		"CoatHanger.jpg" ,
		"Stationery.jpg",
		"PETS2000Frame0124.jpg",
		"PETS2000Frame0129.jpg", //20
		"PCBImage.jpg" ,
	    "LicensePlate1.jpg",
		"BicycleBackgroundImage.jpg",
		"BicycleModel2.jpg",
		"Numbers.jpg", //25
		"GoodORings.jpg",
		"BadORings.jpg",
		"UnknownORings.jpg"		
    };

	// Load images
	int number_of_images = sizeof(image_files)/sizeof(image_files[0]);
	Mat* image = new Mat[number_of_images];
	for (int file_no=0; (file_no < number_of_images); file_no++)
	{
		string filename(file_location);
		filename.append(image_files[file_no]);
		image[file_no] = imread(filename, -1);
		if (image[file_no].empty())
		{
			cout << "Could not open " << image[file_no] << endl;
			return -1;
		}
	}



	// Needed for mean shift in histogram demos
	Rect Surveillance_car_position_frame_124(251,164,64,32);
	Rect Bicycles_position_frame_180(242,26,37,60);
	Rect Person_position_frame_100(507,110,67,90);
	// Load video(s)
	char* video_files[] = { 
		"PETS2000_mjpeg.avi",
		"Bicycles_mjpeg.avi",
		"ObjectAbandonmentAndRemoval1_mjpeg.avi",
		"DraughtsGame1.MOV" };
	int number_of_videos = sizeof(video_files)/sizeof(video_files[0]);
	VideoCapture* video = new VideoCapture[number_of_videos];
	for (int video_file_no=0; (video_file_no < number_of_videos); video_file_no++)
	{
		string filename(file_location);
		filename.append(video_files[video_file_no]);
		video[video_file_no].open(filename);
		if( !video[video_file_no].isOpened() )
		{
			cout << "Cannot open video file: " << filename << endl;
//			return -1;
		}
	}

	// Load Haar Cascade(s)
	vector<CascadeClassifier> cascades;
	char* cascade_files[] = { 
		"haarcascades/haarcascade_frontalface_alt.xml" };
	int number_of_cascades = sizeof(cascade_files)/sizeof(cascade_files[0]);
	for (int cascade_file_no=0; (cascade_file_no < number_of_cascades); cascade_file_no++)
	{
		CascadeClassifier cascade;
		string filename(file_location);
		filename.append(cascade_files[cascade_file_no]);
		if( !cascade.load( filename ) )
		{
			cout << "Cannot load cascade file: " << filename << endl;
			return -1;
		}
		else cascades.push_back(cascade);
	}

	int line_step = 13;
	Point location( 7, 13 );
	Scalar colour( 0, 0, 255);
	Mat default_image = ComputeDefaultImage( image[CAMPANILE1_IMAGE_INDEX] );
	putText( default_image, "OpenCV demonstration system from:", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += line_step*3/2;
	putText( default_image, "    A PRACTICAL INTRODUCTION TO COMPUTER VISION WITH OPENCV", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += line_step;
	putText( default_image, "     by Kenneth Dawson-Howe (C) John Wiley & Sons, Inc. 2014", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += line_step*5/2;
	putText( default_image, "Menu choices:", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += line_step*3/2;
	putText( default_image, "1. Images (Sampling+Quantisation, Colour Models, Noise+Smoothing)", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += line_step;
	putText( default_image, "2. Histograms (Histograms, Equalisation, Selection, Back Proj)", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += line_step;
	putText(default_image, "3. Binary Vision (Thresholding, Morphology)", location, FONT_HERSHEY_SIMPLEX, 0.4, colour);
	location.y += line_step;
	putText(default_image, "4. Region Segmentation (Connected Components, k-means, Mean Shift)", location, FONT_HERSHEY_SIMPLEX, 0.4, colour);
	location.y += line_step;
	putText( default_image, "5. Geometric models (Transformation, Interpolation)", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += line_step;
	putText( default_image, "6. Edges (Roberts, Sobel, Laplacian, Colour, Sharpening, Line, Hough)", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += line_step;
	putText( default_image, "7. Features (Features and Feature Matching)", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += line_step;
	putText( default_image, "8. Recognition (Statistics, Templates, Chamfer, Haar and HoG)", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += line_step;
	putText( default_image, "9. Video Processing (Background & Optical Flow, followed by Mean Shift)", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	location.y += line_step;
	putText(default_image, "c. Camera Calibration", location, FONT_HERSHEY_SIMPLEX, 0.4, colour);
	location.y += line_step;
	putText(default_image, "m. My Application", location, FONT_HERSHEY_SIMPLEX, 0.4, colour);
	location.y += line_step;
	putText( default_image, "X. eXit", location, FONT_HERSHEY_SIMPLEX, 0.4, colour );
	Mat imageROI;
	imageROI = default_image(cv::Rect(0,0,default_image.cols,245));
	addWeighted(imageROI,2.5,imageROI,0.0,0.0,imageROI);

	int choice;
	do
	{
		imshow("Welcome", default_image);
		choice = cv::waitKey();
		cv::destroyAllWindows();
		switch (choice)
		{
		case '1':
			ImagesDemos(image[CHURCH_IMAGE_INDEX],image[FRUIT_IMAGE_INDEX],
				image[CREST_IMAGE_INDEX],image[ASTRONAUT_IMAGE_INDEX]);
			break;
		case '2':
			HistogramsDemos(image[CAMPANILE2_IMAGE_INDEX], image[FRUIT_IMAGE_INDEX],
				image[PEOPLE2_IMAGE_INDEX], image[SKIN_IMAGE_INDEX],
				image, number_of_images);
			break;
		case '3':
			BinaryDemos(image[PCB_IMAGE_INDEX], image[STATIONARY_IMAGE_INDEX]);
			break;
		case '4':
			RegionDemos(image[PCB_IMAGE_INDEX], image[COATS_IMAGE_INDEX], image[FRUIT_IMAGE_INDEX]);
			break;
		case '5':
			GeometricDemos(image[LICENSE_PLATE_IMAGE_INDEX],
				image[PETS124_IMAGE_INDEX],image[PETS129_IMAGE_INDEX]);
			break;
		case '6':
			EdgeDemos(image[BIKES_IMAGE_INDEX],image[COATS_IMAGE_INDEX]);
			break;
		case '7':
			FeaturesDemos(image[CHURCH_IMAGE_INDEX],
				image[PETS124_IMAGE_INDEX],image[PETS129_IMAGE_INDEX]);
			TrackFeaturesDemo( video[SURVEILLANCE_VIDEO_INDEX], 120, 159 );
			break;
		case '8':
			RecognitionDemos(image[OLDLIBRARY_IMAGE_INDEX],image[WINDOW1_IMAGE_INDEX],
				image[WINDOW2_IMAGE_INDEX],image[WINDOW1_LOCATIONS_IMAGE_INDEX],
				image[WINDOW2_LOCATIONS_IMAGE_INDEX],video[BICYCLES_VIDEO_INDEX],
				image[BICYCLE_BACKGROUND_IMAGE_INDEX],image[BICYCLE_MODEL_IMAGE_INDEX],
				video[SURVEILLANCE_VIDEO_INDEX],cascades[HAAR_FACE_CASCADE_INDEX],
				image[NUMBERS_IMAGE_INDEX],image[GOOD_ORINGS_IMAGE_INDEX],image[BAD_ORINGS_IMAGE_INDEX],image[UNKNOWN_ORINGS_IMAGE_INDEX]);
			break;
		case '9':
			VideoDemos(video[SURVEILLANCE_VIDEO_INDEX],120,false);
			MeanShiftDemo(video[ABANDONMENT_VIDEO_INDEX],Person_position_frame_100,100,230);
			break;
		case 'c':
			{
				string filename(file_location);
				filename.append("default.xml");
				CameraCalibration( filename );
			}
			break;
		case 'm':
			MyApplication();
			break;
		default:
			break;
		}
	} while ((choice != 'x') && (choice != 'X'));
}

