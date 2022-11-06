#include "Utilities.h"
#include <iostream>
#include <fstream>
#include <list>
#include <string> 
#include <experimental/filesystem> // C++-standard header file name
#include <filesystem> // Microsoft-specific implementation header file name
using namespace std::experimental::filesystem::v1;
using namespace std;

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"

using namespace cv;
using namespace std;

// Data provided:  Filename, White pieces, Black pieces
// Note that this information can ONLY be used to evaluate performance.  It must not be used during processing of the images.
const string GROUND_TRUTH_FOR_BOARD_IMAGES[][3] = {
	{"DraughtsGame1Move0.JPG", "1,2,3,4,5,6,7,8,9,10,11,12", "21,22,23,24,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move1.JPG", "1,2,3,4,5,6,7,8,10,11,12,13", "21,22,23,24,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move2.JPG", "1,2,3,4,5,6,7,8,10,11,12,13", "20,21,22,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move3.JPG", "1,2,3,4,5,7,8,9,10,11,12,13", "20,21,22,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move4.JPG", "1,2,3,4,5,7,8,9,10,11,12,13", "17,20,21,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move5.JPG", "1,2,3,4,5,7,8,9,10,11,12,22", "20,21,23,25,26,27,28,29,30,31,32"},
	{"DraughtsGame1Move6.JPG", "1,2,3,4,5,7,8,9,10,11,12", "17,20,21,23,25,27,28,29,30,31,32"},
	{"DraughtsGame1Move7.JPG", "1,2,3,4,5,7,8,10,11,12,13", "17,20,21,23,25,27,28,29,30,31,32"},
	{"DraughtsGame1Move8.JPG", "1,2,3,4,5,7,8,10,11,12,13", "17,20,21,23,25,26,27,28,29,31,32"},
	{"DraughtsGame1Move9.JPG", "1,2,3,4,5,7,8,10,11,12,22", "20,21,23,25,26,27,28,29,31,32"},
	{"DraughtsGame1Move10.JPG", "1,2,3,4,5,7,8,10,11,12", "18,20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move11.JPG", "1,2,3,4,5,7,8,10,11,16", "18,20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move12.JPG", "1,2,3,4,5,7,8,10,11,16", "14,20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move13.JPG", "1,2,3,4,5,7,8,11,16,17", "20,21,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move14.JPG", "1,2,3,4,5,7,8,11,16", "14,20,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move15.JPG", "1,3,4,5,6,7,8,11,16", "14,20,23,26,27,28,29,31,32"},
	{"DraughtsGame1Move16.JPG", "1,3,4,5,6,7,8,11,16", "14,20,22,23,27,28,29,31,32"},
	{"DraughtsGame1Move17.JPG", "1,3,4,5,7,8,9,11,16", "14,20,22,23,27,28,29,31,32"},
	{"DraughtsGame1Move18.JPG", "1,3,4,5,7,8,9,11,16", "14,18,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move19.JPG", "1,3,4,5,7,8,9,15,16", "14,18,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move20.JPG", "1,3,4,5,8,9,16", "K2,14,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move21.JPG", "1,3,4,5,8,16,18", "K2,20,23,27,28,29,31,32"},
	{"DraughtsGame1Move22.JPG", "1,3,4,5,8,16", "K2,14,20,27,28,29,31,32"},
	{"DraughtsGame1Move23.JPG", "1,4,5,7,8,16", "K2,14,20,27,28,29,31,32"},
	{"DraughtsGame1Move24.JPG", "1,4,5,7,8", "K2,11,14,27,28,29,31,32"},
	{"DraughtsGame1Move25.JPG", "1,4,5,8,16", "K2,14,27,28,29,31,32"},
	{"DraughtsGame1Move26.JPG", "1,4,5,8,16", "K7,14,27,28,29,31,32"},
	{"DraughtsGame1Move27.JPG", "1,4,5,11,16", "K7,14,27,28,29,31,32"},
	{"DraughtsGame1Move28.JPG", "1,4,5,11,16", "K7,14,24,28,29,31,32"},
	{"DraughtsGame1Move29.JPG", "4,5,6,11,16", "K7,14,24,28,29,31,32"},
	{"DraughtsGame1Move30.JPG", "4,5,6,11,16", "K2,14,24,28,29,31,32"},
	{"DraughtsGame1Move31.JPG", "4,5,9,11,16", "K2,14,24,28,29,31,32"},
	{"DraughtsGame1Move32.JPG", "4,5,9,11,16", "K2,10,24,28,29,31,32"},
	{"DraughtsGame1Move33.JPG", "4,5,11,14,16", "K2,10,24,28,29,31,32"},
	{"DraughtsGame1Move34.JPG", "4,5,11,14,16", "K2,7,24,28,29,31,32"},
	{"DraughtsGame1Move35.JPG", "4,5,11,16,17", "K2,7,24,28,29,31,32"},
	{"DraughtsGame1Move36.JPG", "4,5,11,16,17", "K2,K3,24,28,29,31,32"},
	{"DraughtsGame1Move37.JPG", "4,5,15,16,17", "K2,K3,24,28,29,31,32"},
	{"DraughtsGame1Move38.JPG", "4,5,15,16,17", "K2,K3,20,28,29,31,32"},
	{"DraughtsGame1Move39.JPG", "4,5,15,17,19", "K2,K3,20,28,29,31,32"},
	{"DraughtsGame1Move40.JPG", "4,5,15,17,19", "K2,K7,20,28,29,31,32"},
	{"DraughtsGame1Move41.JPG", "4,5,17,18,19", "K2,K7,20,28,29,31,32"},
	{"DraughtsGame1Move42.JPG", "4,5,17,18,19", "K2,K10,20,28,29,31,32"},
	{"DraughtsGame1Move43.JPG", "4,5,17,19,22", "K2,K10,20,28,29,31,32"},
	{"DraughtsGame1Move44.JPG", "4,5,17,19,22", "K2,K14,20,28,29,31,32"},
	{"DraughtsGame1Move45.JPG", "4,5,19,21,22", "K2,K14,20,28,29,31,32"},
	{"DraughtsGame1Move46.JPG", "4,5,19,21,22", "K2,K17,20,28,29,31,32"},
	{"DraughtsGame1Move47.JPG", "4,5,19,22,25", "K2,K17,20,28,29,31,32"},
	{"DraughtsGame1Move48.JPG", "4,5,19,25", "K2,20,K26,28,29,31,32"},
	{"DraughtsGame1Move49.JPG", "4,5,19,K30", "K2,20,K26,28,29,31,32"},
	{"DraughtsGame1Move50.JPG", "4,5,19,K30", "K2,20,K26,27,28,29,32"},
	{"DraughtsGame1Move51.JPG", "4,5,19,K23", "K2,20,27,28,29,32"},
	{"DraughtsGame1Move52.JPG", "4,5,19", "K2,18,20,28,29,32"},
	{"DraughtsGame1Move53.JPG", "4,5,23", "K2,18,20,28,29,32"},
	{"DraughtsGame1Move54.JPG", "4,5,23", "K2,15,20,28,29,32"},
	{"DraughtsGame1Move55.JPG", "4,5,26", "K2,15,20,28,29,32"},
	{"DraughtsGame1Move56.JPG", "4,5,26", "K2,11,20,28,29,32"},
	{"DraughtsGame1Move57.JPG", "4,5,K31", "K2,11,20,28,29,32"},
	{"DraughtsGame1Move58.JPG", "4,5,K31", "K2,11,20,27,28,29"},
	{"DraughtsGame1Move59.JPG", "4,5,K24", "K2,11,20,28,29"},
	{"DraughtsGame1Move60.JPG", "4,5", "K2,11,19,20,29"},
	{"DraughtsGame1Move61.JPG", "4,9", "K2,11,19,20,29"},
	{"DraughtsGame1Move62.JPG", "4,9", "K2,11,19,20,25"},
	{"DraughtsGame1Move63.JPG", "4,14", "K2,11,19,20,25"},
	{"DraughtsGame1Move64.JPG", "4,14", "K2,11,19,20,22"},
	{"DraughtsGame1Move65.JPG", "4,18", "K2,11,19,20,22"},
	{"DraughtsGame1Move66.JPG", "4", "K2,11,15,19,20"},
	{"DraughtsGame1Move67.JPG", "8", "K2,11,15,19,20"},
	{"DraughtsGame1Move68.JPG", "", "K2,K4,15,19,20"}
};

// Data provided:  Approx. frame number, From square number, To square number
// Note that the first move is a White move (and then the moves alternate Black, White, Black, White...)
// This data corresponds to the video:  DraughtsGame1.avi
// Note that this information can ONLY be used to evaluate performance.  It must not be used during processing of the video.
const int GROUND_TRUTH_FOR_DRAUGHTSGAME1_VIDEO_MOVES[][3] = {
	{17, 9, 13},
	{37, 24, 20},
	{50, 6, 9},
	{65, 22, 17},
	{85, 13, 22},
	{108, 26, 17},
	{123, 9, 13},
	{161, 30, 26},
	{180, 13, 22},
	{201, 25, 18},
	{226, 12, 16},
	{244, 18, 14},
	{266, 10, 17},
	{285, 21, 14},
	{308, 2, 6},
	{326, 26, 22},
	{343, 6, 9},
	{362, 22, 18},
	{393, 11, 15},
	{433, 18, 2},
	{453, 9, 18},
	{472, 23, 14},
	{506, 3, 7},
	{530, 20, 11},
	{546, 7, 16},
	{582, 2, 7},
	{617, 8, 11},
	{641, 27, 24},
	{673, 1, 6},
	{697, 7, 2},
	{714, 6, 9},
	{728, 14, 10},
	{748, 9, 14},
	{767, 10, 7},
	{781, 14, 17},
	{801, 7, 3},
	{814, 11, 15},
	{859, 24, 20},
	{870, 16, 19},
	{891, 3, 7},
	{923, 15, 18},
	{936, 7, 10},
	{955, 18, 22},
	{995, 10, 14},
	{1014, 17, 21},
	{1034, 14, 17},
	{1058, 21, 25},
	{1075, 17, 26},
	{1104, 25, 30},
	{1129, 31, 27},
	{1147, 30, 23},
	{1166, 27, 18},
	{1182, 19, 23},
	{1201, 18, 15},
	{1213, 23, 26},
	{1243, 15, 11},
	{1266, 26, 31},
	{1280, 32, 27},
	{1298, 31, 24},
	{1324, 28, 19},
	{1337, 5, 9},
	{1358, 29, 25},
	{1387, 9, 14},
	{1450, 25, 22},
	{-1, 14, 18},
	{-1, 22, 15},
	{1465, 4, 8},
	{1490, 11, 4} };



/// confusion matrix for part two
/// Predicted		Ground Truth
///			Empty	White	Black
/// Empty
/// White
/// Black
int confusionMatrix[3][3] =
{ 
	{0, 0, 0},
	{0, 0, 0},
	{0, 0, 0}
};

#define EMPTY_SQUARE 0
#define WHITE_MAN_ON_SQUARE 1
#define BLACK_MAN_ON_SQUARE 3
#define WHITE_KING_ON_SQUARE 2
#define BLACK_KING_ON_SQUARE 4
#define NUMBER_OF_SQUARES_ON_EACH_SIDE 8
#define NUMBER_OF_SQUARES (NUMBER_OF_SQUARES_ON_EACH_SIDE*NUMBER_OF_SQUARES_ON_EACH_SIDE/2)



class DraughtsBoard
{
private:
	Mat mOriginalImage;
	void loadGroundTruth(string pieces, int man_type, int king_type);
public:
	int mBoardGroundTruth[NUMBER_OF_SQUARES];
	DraughtsBoard(string filename, string white_pieces_ground_truth, string black_pieces_ground_truth);
};

DraughtsBoard::DraughtsBoard(string filename, string white_pieces_ground_truth, string black_pieces_ground_truth)
{
	for (int square_count = 1; square_count <= NUMBER_OF_SQUARES; square_count++)
	{
		mBoardGroundTruth[square_count - 1] = EMPTY_SQUARE;
	}
	loadGroundTruth(white_pieces_ground_truth, WHITE_MAN_ON_SQUARE, WHITE_KING_ON_SQUARE);
	loadGroundTruth(black_pieces_ground_truth, BLACK_MAN_ON_SQUARE, BLACK_KING_ON_SQUARE);
	string full_filename = "Media/" + filename;
	mOriginalImage = imread(full_filename, -1);
	if (mOriginalImage.empty())
		cout << "Cannot open image file: " << full_filename << endl;
	//else imshow(full_filename, mOriginalImage);
}

void DraughtsBoard::loadGroundTruth(string pieces, int man_type, int king_type)
{
	int index = 0;
	while (index < pieces.length())
	{
		bool is_king = false;
		if (pieces.at(index) == 'K')
		{
			is_king = true;
			index++;
		}
		int location = 0;
		while ((index < pieces.length()) && (pieces.at(index) >= '0') && (pieces.at(index) <= '9'))
		{
			location = location * 10 + (pieces.at(index) - '0');
			index++;
		}
		index++;
		if ((location > 0) && (location <= NUMBER_OF_SQUARES))
			mBoardGroundTruth[location - 1] = (is_king) ? king_type : man_type;
	}
}


// Part One 
Mat partOne (Mat& static_background_image) {

	Mat current_image = static_background_image;

	// First step blurring on the background image
	Mat blurred;
	medianBlur(static_background_image, blurred, 3);
	medianBlur(blurred, blurred, 3);

	// Backprojection of each of the samples
	string black_pieces_filename("Media/DraughtsGame1BlackPieces.jpg");
	Mat black_pieces_image = imread(black_pieces_filename, -1);
	Mat black_pieces = BackProjection(blurred, black_pieces_image);


	string white_squares_filename("Media/DraughtsGame1WhiteSquares.jpg");
	Mat white_squares_image = imread(white_squares_filename, -1);
	Mat white_squares = BackProjection(blurred, white_squares_image);

	Mat output = JoinImagesHorizontally(black_pieces, "Black Pieces", white_squares, "White Squares", 4);

	string white_pieces_filename("Media/DraughtsGame1WhitePieces.jpg");
	Mat white_pieces_image = imread(white_pieces_filename, -1);
	Mat white_pieces = BackProjection(blurred, white_pieces_image);

	output = JoinImagesHorizontally(output, "Original Image", white_pieces, "White Pieces", 4);

	string black_squares_filename("Media/DraughtsGame1BlackSquares.jpg");
	Mat black_squares_image = imread(black_squares_filename, -1);
	Mat black_squares = BackProjection(blurred, black_squares_image);

	output = JoinImagesHorizontally(output, "Original Image", black_squares, "Black Pieces", 4);

	imshow("Output of Backprojection", output);

	// Threshold + combine the backprojection using setTo

	Mat thresh;
	
	//adaptiveThreshold(black_pieces, thresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 15, -5);
	threshold(black_pieces, thresh, 30, 255, THRESH_BINARY | THRESH_OTSU);
	Mat closed_image;
	Mat five_by_five_element(5, 5, CV_8U, Scalar(1));

	dilate(thresh, closed_image, Mat());

	current_image.setTo(Scalar(255, 255, 255), closed_image);

	threshold(white_pieces, thresh, 30, 255, THRESH_BINARY | THRESH_OTSU);

	dilate(thresh, closed_image, Mat());

	current_image.setTo(Scalar(0, 0, 0), closed_image);

	//threshold(black_squares, thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);

	current_image.setTo(Scalar(0, 0, 255), black_squares);

	//threshold(white_squares, thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);

	current_image.setTo(Scalar(0, 255, 255), white_squares);

	imshow("Part One Result", current_image);

	return current_image;
}

// Part Two
bool isBlackSquare(int col, int row) {
	switch (col) {
		case 0:
			switch (row) {
			case 1:
				return true;
			case 3:
				return true;
			case 5:
				return true;
			case 7:
				return true;
			}
			return false;
		case 1:
			switch (row) {
			case 0:
				return true;
			case 2:
				return true;
			case 4:
				return true;
			case 6:
				return true;
			}
			return false;
		case 2:
			switch (row) {
			case 1:
				return true;
			case 3:
				return true;
			case 5:
				return true;
			case 7:
				return true;
			}
			return false;
		case 3:
			switch (row) {
			case 0:
				return true;
			case 2:
				return true;
			case 4:
				return true;
			case 6:
				return true;
			}
			return false;
		case 4:
			switch (row) {
			case 1:
				return true;
			case 3:
				return true;
			case 5:
				return true;
			case 7:
				return true;
			}
			return false;
		case 5:
			switch (row) {
			case 0:
				return true;
			case 2:
				return true;
			case 4:
				return true;
			case 6:
				return true;
			}
			return false;
		case 6: 
			switch (row) {
			case 1:
				return true;
			case 3:
				return true;
			case 5:
				return true;
			case 7:
				return true;
			}
			return false;
		case 7:
			switch (row) {
			case 0:
				return true;
			case 2:
				return true;
			case 4:
				return true;
			case 6:
				return true;
			}
			return false;
	}

	return false;
}

int piece(Mat& img) {

	Vec3b colours;
	int black_pixels = 0;
	int white_pixels = 0;
	int empty_piece = 0;

	for (int j = 0; j < (img.cols); j += 1) {

		for (int i = 0; i < (img.rows); i += 1) {

			Vec3b colours = img.at<Vec3b>(i, j);
			
			if ((colours[0] == 255 && colours[1] == 255 && colours[2] == 255)) {
				black_pixels++;
			} else if (colours[0] == 0 && colours[1] == 0 && colours[2] == 0) {
				white_pixels++;
			}
			else //if (colours[0] == 0 && colours[1] == 0 && colours[2] == 255)
			{
				empty_piece++;
			}
		}
	}

	//cout << "black : " << black_pixels << " white_pixels: " << white_pixels << " empty: " << empty_piece << "\n";
	if (empty_piece > 1898) {
		return EMPTY_SQUARE;
	}
	else if (black_pixels > white_pixels && black_pixels > 50) {
		return BLACK_MAN_ON_SQUARE;
	}
	else if (white_pixels > black_pixels && white_pixels > 50) {
		return WHITE_MAN_ON_SQUARE;
	}
	else {
		return EMPTY_SQUARE;
	}


}

string black = "";
string white = "";

int pieces [32];

void partTwo(Mat& current_img, int current) {

	DraughtsBoard current_board(GROUND_TRUTH_FOR_BOARD_IMAGES[current][0], GROUND_TRUTH_FOR_BOARD_IMAGES[current][1], GROUND_TRUTH_FOR_BOARD_IMAGES[current][2]);

	Point2f source[4] = {{ 114.0, 17.0 }, { 53.0, 245.0 }, { 355.0, 20.0 }, { 433.0, 241.0 }};
	Point2f destination[4] = { {0.0, 0.0}, {Point2f(0.0, current_img.rows)}, {Point2f(current_img.cols,0.0)},  {Point2f(current_img.cols, current_img.rows)} };

	Mat result;
	Mat perspective_matrix = getPerspectiveTransform(source, destination); 

	warpPerspective(current_img, result, perspective_matrix,result.size());
	imshow("Perspective change", result);

	Mat end;

	int indexRow = result.rows/ NUMBER_OF_SQUARES_ON_EACH_SIDE;
	int indexCol = result.cols/ NUMBER_OF_SQUARES_ON_EACH_SIDE;

	end = result(Rect(indexCol, 0, indexCol, indexRow));
	piece(end);

	int rowNum = 0;
	int colNum = 0;
	int index = 0;
	for (int j = 0; j < (result.cols); j += (indexCol)) {

		for (int i = 0; i < (result.rows - indexRow); i += (indexRow)) {

			end = result(Rect(j, i, indexCol, indexRow));

			if (isBlackSquare( colNum, rowNum)) {
				int piece_no = piece(end);
				if (piece_no == BLACK_MAN_ON_SQUARE) {
					black += std::to_string(index +1)+ ",";
					pieces[index] = BLACK_MAN_ON_SQUARE;
					if (current_board.mBoardGroundTruth[index] == BLACK_MAN_ON_SQUARE || current_board.mBoardGroundTruth[index] == BLACK_KING_ON_SQUARE) {
						confusionMatrix[2][2]++;
					}
					else if (current_board.mBoardGroundTruth[index] == EMPTY_SQUARE) {
						confusionMatrix[2][0]++;
					}
					else if (current_board.mBoardGroundTruth[index] == WHITE_MAN_ON_SQUARE || current_board.mBoardGroundTruth[index] == WHITE_KING_ON_SQUARE) {
						confusionMatrix[2][1]++;
					}
				} else if (piece_no == WHITE_MAN_ON_SQUARE) {
					white += std::to_string(index+1) + ",";
					pieces[index] = WHITE_MAN_ON_SQUARE;
					if (current_board.mBoardGroundTruth[index] == BLACK_MAN_ON_SQUARE || current_board.mBoardGroundTruth[index] == BLACK_KING_ON_SQUARE) {
						confusionMatrix[1][2]++;
					}
					else if (current_board.mBoardGroundTruth[index] == EMPTY_SQUARE) {
						confusionMatrix[1][0]++;
					}
					else if (current_board.mBoardGroundTruth[index] == WHITE_MAN_ON_SQUARE || current_board.mBoardGroundTruth[index] == WHITE_KING_ON_SQUARE) {
						confusionMatrix[1][1]++;
					}
				}
				else if (piece_no  == EMPTY_SQUARE) {
					pieces[index] = EMPTY_SQUARE;
					if (current_board.mBoardGroundTruth[index] == BLACK_MAN_ON_SQUARE || current_board.mBoardGroundTruth[index] == BLACK_KING_ON_SQUARE) {
						confusionMatrix[0][2]++;
					}
					else if (current_board.mBoardGroundTruth[index] == EMPTY_SQUARE) {
						confusionMatrix[0][0]++;
					}
					else if (current_board.mBoardGroundTruth[index] == WHITE_MAN_ON_SQUARE || current_board.mBoardGroundTruth[index] == WHITE_KING_ON_SQUARE) {
						confusionMatrix[0][1]++;
					}
				}

				index++;
			}
			rowNum++;
		}
		rowNum = 0;
		colNum++;
	}
}

// Part Thee
int lastMove[32] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

int newEmpty = 0;
int newMove = 0;

bool checkMoves() {
	bool move = false;
	newEmpty = 0;
	newMove = 0;
	for (int i = 0; i < 32; i++) {
		if (lastMove[i] != pieces[i]) {
			if (pieces[i] == EMPTY_SQUARE) {
				newEmpty = i+1;
			}
			if (lastMove[i] == EMPTY_SQUARE) {
				newMove = i+1;
			}
			
			move = true;
		}
	}
	
	if (move && (newMove != 0) && (newEmpty != 0)) {
		cout << newEmpty << ", " << newMove << "\n";
		std::copy(pieces, pieces + 32, lastMove);
		return true;
	}
	else {
		return false;
	}
	
}

// Part Three
void partThree(VideoCapture& video) {

	Ptr<BackgroundSubtractorMOG2> gmm = createBackgroundSubtractorMOG2();
	
	Mat current_frame, last_frame;
	video >> (current_frame);
	Mat ptOne = partOne(current_frame);
	partTwo(ptOne, 0);
	std::copy(pieces, pieces + 32, lastMove);
	bool firstFrame = true;
	int number_of_frames = 0;
	int count_of_stills = 0;
	int stills = 0;
	int trues = 0;

	while (!current_frame.empty())
	{
		last_frame = current_frame.clone();
		video >> (current_frame);
		if (!current_frame.empty()) {
			gmm->apply(current_frame, last_frame);
			Mat moving_points;
			threshold(last_frame, moving_points, 150, 255, THRESH_BINARY);
			Mat diff;
			absdiff(last_frame, moving_points, diff);
			threshold(last_frame, moving_points, 25, 255, THRESH_BINARY);
			double percentage = (countNonZero(diff) * 100) / diff.total();

			if (percentage == 0) {
				count_of_stills = 0;
				ptOne = partOne(current_frame);
				partTwo(ptOne, 0);
				bool isMove = checkMoves();
				if (isMove) {
						
					cout << "Current Frame: " << number_of_frames << "\n";
					if (GROUND_TRUTH_FOR_DRAUGHTSGAME1_VIDEO_MOVES[stills][1] == newEmpty && GROUND_TRUTH_FOR_DRAUGHTSGAME1_VIDEO_MOVES[stills][2] == newMove) {
						cout << "true move " << stills<<" \n";
					}
					if (GROUND_TRUTH_FOR_DRAUGHTSGAME1_VIDEO_MOVES[stills][1] == newEmpty) {
						trues++;
					}
					if (GROUND_TRUTH_FOR_DRAUGHTSGAME1_VIDEO_MOVES[stills][2] == newMove) {
						trues++;
					}
					stills++;
				}
			}


			number_of_frames++;
		}
		
	}

	cout << "Number of frames: " << stills << " trues: " << trues << "\n";

	

}

// Part Four
void partFour (Mat& static_img) {

	Mat blurred;
	medianBlur(static_img, blurred, 9);
	medianBlur(blurred, blurred, 5);


	Mat grayscale_board, binary_image;
	imshow("Static Image", blurred);
	cvtColor(blurred, grayscale_board, COLOR_BGR2GRAY, 0);
	//Apply threshold to image
	threshold(grayscale_board, binary_image, 255, 255, THRESH_BINARY | THRESH_OTSU);
	imshow("Binary Image", binary_image);

	// Hough Lines
	Mat canny_edge_image;
	Canny(grayscale_board, canny_edge_image, 75, 150);
	vector<Vec2f> hough_lines;
	// raised the threshold
	HoughLines(canny_edge_image, hough_lines, 1, PI / 200.0, 80);
	Mat hough_lines_image = static_img.clone();
	DrawLines(hough_lines_image, hough_lines, Scalar(0, 255, 0));
	imshow("hough", hough_lines_image);

	Canny(static_img, binary_image, 255, 255 * 2);
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(binary_image, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
	Mat drawing = Mat::zeros(binary_image.size(), CV_8UC3);
	for (size_t i = 0; i < contours.size(); i++)
	{
		Scalar colour(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF);
		drawContours(drawing, contours, (int)i, colour, 2, LINE_8, hierarchy, 0);
	}
	imshow("Contours", drawing);

	// Use of contour following and straight line segment extraction.
	findContours(canny_edge_image, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

	Mat clone = static_img.clone();
	for (int contour_number = 0; (contour_number < contours.size()); contour_number++) {
		Scalar colour(rand() & 0xFF, rand() & 0xFF, rand() & 0xFF); 
		drawContours(clone, contours, contour_number,colour, 1, 8, hierarchy);
	}
	imshow("finish", clone);


	//FindChessboardCorners()
	vector<Point2f> corners;

	Mat img_gray;
	cvtColor(static_img, img_gray, COLOR_BGR2GRAY);

	Size patternSize = Size(7, 7);
	bool patternFound = false;

	patternFound = findChessboardCorners(img_gray, patternSize, corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE);
	cvtColor(img_gray, img_gray, COLOR_GRAY2BGR);
	drawChessboardCorners(img_gray, patternSize, Mat(corners), patternFound);
	imshow("findChessboard", img_gray);
	if (findChessboardCorners) {
		cout << "Chessboard found \n";
		cout << "The corners: " << corners << "\n";

		Mat current_img = img_gray.clone();
		Point2f innerLeft = corners[0];
		Point2f topRight = corners[6];
		Point2f bottomLeft = corners[corners.size() - 8];
		Point2f bottomRight = corners[corners.size() - 1];
		Point2f source[4] = { innerLeft , topRight, bottomLeft , bottomRight };
		Point2f destination[4] = { {0.0, 0.0}, {Point2f(0.0, current_img.rows)}, {Point2f(current_img.cols,0.0)},  {Point2f(current_img.cols, current_img.rows)} };

		Mat result;
		Mat perspective_matrix = getPerspectiveTransform(source, destination);

		warpPerspective(current_img, result, perspective_matrix, result.size());
		imshow("find Board", result);
	}	
	else {
		cout << "no chessboard\n";
	}
	
}

// Part Five

/// Extended confusion matrix for part Five
/// Predicted		Ground Truth
///				Empty	White	Black   White King   Black King
/// Empty
/// White
/// Black
/// White King
/// Black King
int extendedConfusionMatrix [5][5] =
{
	{0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0},
	{0, 0, 0, 0, 0}
};

int piece_type(Mat& img) {

	Vec3b colours;
	int black_pixels = 0;
	int white_pixels = 0;
	int empty_piece = 0;

	for (int j = 0; j < (img.cols); j += 1) {

		for (int i = 0; i < (img.rows); i += 1) {

			Vec3b colours = img.at<Vec3b>(i, j);

			if ((colours[0] == 255 && colours[1] == 255 && colours[2] == 255)) {
				black_pixels++;
			}
			else if (colours[0] == 0 && colours[1] == 0 && colours[2] == 0) {
				white_pixels++;
			}
			else
			{
				empty_piece++;
			}
		}
	}

	//cout << "black : " << black_pixels << " white_pixels: " << white_pixels << " empty: " << empty_piece << "\n";
	if (empty_piece > 1898) {
		return EMPTY_SQUARE;
	}
	else if (black_pixels > white_pixels && black_pixels > 600) {
		return BLACK_KING_ON_SQUARE;
	}
	else if (white_pixels > black_pixels && white_pixels > 700) {
		return WHITE_KING_ON_SQUARE;
	}
	else if (black_pixels > white_pixels && black_pixels > 50) {
		return BLACK_MAN_ON_SQUARE;
	}
	else if (white_pixels > black_pixels && white_pixels > 50) {
		return WHITE_MAN_ON_SQUARE;
	}
	else {
		return EMPTY_SQUARE;
	}


}

// Part Five - Average threshold
void partFive (Mat& current_img, int current) {
	// Set an average threshold for pixels that are pieces

	DraughtsBoard current_board(GROUND_TRUTH_FOR_BOARD_IMAGES[current][0], GROUND_TRUTH_FOR_BOARD_IMAGES[current][1], GROUND_TRUTH_FOR_BOARD_IMAGES[current][2]);

	Point2f source[4] = { { 114.0, 17.0 }, { 53.0, 245.0 }, { 355.0, 20.0 }, { 433.0, 241.0 } };
	Point2f destination[4] = { {0.0, 0.0}, {Point2f(0.0, current_img.rows)}, {Point2f(current_img.cols,0.0)},  {Point2f(current_img.cols, current_img.rows)} };

	Mat result;
	Mat perspective_matrix = getPerspectiveTransform(source, destination);

	warpPerspective(current_img, result, perspective_matrix, result.size());
	imshow("Perspective change", result);

	Mat end;

	int indexRow = result.rows / NUMBER_OF_SQUARES_ON_EACH_SIDE;
	int indexCol = result.cols / NUMBER_OF_SQUARES_ON_EACH_SIDE;

	end = result(Rect(indexCol, 0, indexCol, indexRow));

	int rowNum = 0;
	int colNum = 0;
	int index = 0;
	for (int j = 0; j < (result.cols); j += (indexCol)) {

		for (int i = 0; i < (result.rows - indexRow); i += (indexRow)) {

			end = result(Rect(j, i, indexCol, indexRow));

			if (isBlackSquare(colNum, rowNum) == true) {
				int piece_no = piece_type(end);
				if (piece_no == BLACK_MAN_ON_SQUARE) {
					black += std::to_string(index+1) + ",";
					pieces[index] = BLACK_MAN_ON_SQUARE;
					if (current_board.mBoardGroundTruth[index] == BLACK_MAN_ON_SQUARE) {
						extendedConfusionMatrix[2][2]++;
					}
					else if (current_board.mBoardGroundTruth[index] == EMPTY_SQUARE) {
						extendedConfusionMatrix[2][0]++;
					}
					else if (current_board.mBoardGroundTruth[index] == WHITE_MAN_ON_SQUARE) {
						extendedConfusionMatrix[2][1]++;
					}
					else if (current_board.mBoardGroundTruth[index] == WHITE_KING_ON_SQUARE) {
						extendedConfusionMatrix[2][3]++;
					}
					else if (current_board.mBoardGroundTruth[index] == BLACK_KING_ON_SQUARE) {
						extendedConfusionMatrix[2][4]++;
					}
				}
				else if (piece_no == WHITE_MAN_ON_SQUARE) {
					white += std::to_string(index + 1) + ",";
					pieces[index] = WHITE_MAN_ON_SQUARE;
					if (current_board.mBoardGroundTruth[index] == BLACK_MAN_ON_SQUARE) {
						extendedConfusionMatrix[1][2]++;
					}
					else if (current_board.mBoardGroundTruth[index] == EMPTY_SQUARE) {
						extendedConfusionMatrix[1][0]++;
					}
					else if (current_board.mBoardGroundTruth[index] == WHITE_MAN_ON_SQUARE) {
						extendedConfusionMatrix[1][1]++;
					}
					else if (current_board.mBoardGroundTruth[index] == WHITE_KING_ON_SQUARE) {
						extendedConfusionMatrix[1][3]++;
					}
					else if (current_board.mBoardGroundTruth[index] == BLACK_KING_ON_SQUARE) {
						extendedConfusionMatrix[1][4]++;
					}
				}
				else if (piece_no == EMPTY_SQUARE) {
					pieces[index] = EMPTY_SQUARE;
					if (current_board.mBoardGroundTruth[index] == BLACK_MAN_ON_SQUARE) {
						extendedConfusionMatrix[0][2]++;
					}
					else if (current_board.mBoardGroundTruth[index] == EMPTY_SQUARE) {
						extendedConfusionMatrix[0][0]++;
					}
					else if (current_board.mBoardGroundTruth[index] == WHITE_MAN_ON_SQUARE) {
						extendedConfusionMatrix[0][1]++;
					}
					else if (current_board.mBoardGroundTruth[index] == WHITE_KING_ON_SQUARE) {
						extendedConfusionMatrix[0][3]++;
					}
					else if (current_board.mBoardGroundTruth[index] == BLACK_KING_ON_SQUARE) {
						extendedConfusionMatrix[0][4]++;
					}
				}
				else if (piece_no == WHITE_KING_ON_SQUARE) {
					pieces[index] = WHITE_KING_ON_SQUARE;
					white += "k" + std::to_string(index + 1) + ",";
					if (current_board.mBoardGroundTruth[index] == BLACK_MAN_ON_SQUARE) {
						extendedConfusionMatrix[3][2]++;
					}
					else if (current_board.mBoardGroundTruth[index] == EMPTY_SQUARE) {
						extendedConfusionMatrix[3][0]++;
					}
					else if (current_board.mBoardGroundTruth[index] == WHITE_MAN_ON_SQUARE) {
						extendedConfusionMatrix[3][1]++;
					}
					else if (current_board.mBoardGroundTruth[index] == WHITE_KING_ON_SQUARE) {
						extendedConfusionMatrix[3][3]++;
					}
					else if (current_board.mBoardGroundTruth[index] == BLACK_KING_ON_SQUARE) {
						extendedConfusionMatrix[3][4]++;
					}
				}
				else if (piece_no == BLACK_KING_ON_SQUARE) {
					pieces[index] = BLACK_KING_ON_SQUARE;
					black += "k" + std::to_string(index + 1) + ",";
					if (current_board.mBoardGroundTruth[index] == BLACK_MAN_ON_SQUARE) {
						extendedConfusionMatrix[4][2]++;
					}
					else if (current_board.mBoardGroundTruth[index] == EMPTY_SQUARE) {
						extendedConfusionMatrix[4][0]++;
					}
					else if (current_board.mBoardGroundTruth[index] == WHITE_MAN_ON_SQUARE) {
						extendedConfusionMatrix[4][1]++;
					}
					else if (current_board.mBoardGroundTruth[index] == WHITE_KING_ON_SQUARE) {
						extendedConfusionMatrix[4][3]++;
					}
					else if (current_board.mBoardGroundTruth[index] == BLACK_KING_ON_SQUARE) {
						extendedConfusionMatrix[4][4]++;
					}
				}
				index++;
			}
			rowNum++;
		}
		rowNum = 0;
		colNum++;
	}

}


void MyApplication()
{
	
	// Part One + Part Two

	
	for (int i = 0; i < sizeof(GROUND_TRUTH_FOR_BOARD_IMAGES)/sizeof(GROUND_TRUTH_FOR_BOARD_IMAGES[0]); i++) {
		black = "";
		white = "";
		string background_filename("Media/" + GROUND_TRUTH_FOR_BOARD_IMAGES[i][0]);
		Mat static_background_image = imread(background_filename, -1);

		if (static_background_image.empty())
			cout << "Cannot open image file: " << background_filename << endl;
		else {
			Mat pt1 = partOne(static_background_image);

			partTwo(pt1, i);
			cout << i << " white: " << white << "\n";
			cout << i << " black: " << black << "\n";
		}
		
	}

	// Printing the confusion Matrix
	cout << "Confusion Matrix \n";
	cout << "Predicted        Ground Truth\n";
	cout << "            Empty     White     Black" << "\n";
	cout << "Empty       " << confusionMatrix[0][0] << "       " << confusionMatrix[0][1] << "        " << confusionMatrix[0][2] << "\n";
	cout << "White       " << confusionMatrix[1][0] << "       " << confusionMatrix[1][1] << "         " << confusionMatrix[1][2] << "\n";
	cout << "Black       " << confusionMatrix[2][0] << "       " << confusionMatrix[2][1] << "        " << confusionMatrix[2][2] << "\n";

	float recall = ((static_cast<float>(confusionMatrix[1][1]) + static_cast<float>(confusionMatrix[2][2])) / static_cast<float>((confusionMatrix[1][1]) + static_cast<float>(confusionMatrix[2][2]) + static_cast<float>(confusionMatrix[0][1]) + static_cast<float>(confusionMatrix[0][2]))) * 100.0f;
	float precision = ((static_cast<float>(confusionMatrix[1][1]) + static_cast<float>(confusionMatrix[2][2])) / (static_cast<float>(confusionMatrix[1][1]) + static_cast<float>(confusionMatrix[2][2]) + static_cast<float>(confusionMatrix[1][0]) + static_cast<float>(confusionMatrix[2][0]))) * 100.0f;
	float accuracy = ((static_cast<float>(confusionMatrix[0][0]) + static_cast<float>(confusionMatrix[1][1]) + static_cast<float>(confusionMatrix[2][2])) / (static_cast<float>(confusionMatrix[0][0]) + static_cast<float>(confusionMatrix[0][1]) + static_cast<float>(confusionMatrix[0][2]) + static_cast<float>(confusionMatrix[1][0]) + static_cast<float>(confusionMatrix[1][1]) + static_cast<float>(confusionMatrix[1][2])  + static_cast<float>(confusionMatrix[2][0]) + static_cast<float>(confusionMatrix[2][1]) + static_cast<float>(confusionMatrix[2][2]))) * 100.0f;
	float specificty = static_cast<float>(confusionMatrix[0][0]) / (static_cast<float>(confusionMatrix[0][0]) + static_cast<float>(confusionMatrix[1][0]) + static_cast<float>(confusionMatrix[2][0])) * 100.0f;
	cout << "Recall: " << recall << "%\n";
	cout << "Precison: " << precision << "%\n";
	cout << "Accuracy: " << accuracy << "%\n";
	cout << "Specificity: " << specificty << "%\n";

	// Part Three - Video
	string video_filename("Media/DraughtsGame1.avi");
	VideoCapture video;
	video.open(video_filename);
	//partThree(video);
	
	// Part Four - Finding Corners
	string background_file("Media/DraughtsGame1EmptyBoard.jpg");
	Mat static_background_img = imread(background_file, -1);
	partFour(static_background_img);

	
	// Part Five
	for (int i = 0; i < sizeof(GROUND_TRUTH_FOR_BOARD_IMAGES) / sizeof(GROUND_TRUTH_FOR_BOARD_IMAGES[0]); i++) {
		black = "";
		white = "";
		string background_filename("Media/" + GROUND_TRUTH_FOR_BOARD_IMAGES[i][0]);
		Mat static_background_image = imread(background_filename, -1);

		if (static_background_image.empty())
			cout << "Cannot open image file: " << background_filename << endl;
		else {
			Mat pt1 = partOne(static_background_image);

			partFive(pt1, i);
			cout << i << " white: " << white << "\n";
			cout << i << " black: " << black << "\n";
		}

	}
	
	// Printing the confusion Matrix
	cout << "Confusion Matrix \n";
	cout << "Predicted        Ground Truth\n";
	cout << "            Empty     White     Black    White King    Black King" << "\n";
	cout << "Empty       " << extendedConfusionMatrix[0][0] << "       " << extendedConfusionMatrix[0][1] << "        " << extendedConfusionMatrix[0][2] << "        " << extendedConfusionMatrix[0][3] << "        " << extendedConfusionMatrix[0][4] << "\n";
	cout << "White       " << extendedConfusionMatrix[1][0] << "       " << extendedConfusionMatrix[1][1] << "         " << extendedConfusionMatrix[1][2] << "        " << extendedConfusionMatrix[1][3] << "        " << extendedConfusionMatrix[1][4] << "\n";
	cout << "Black       " << extendedConfusionMatrix[2][0] << "       " << extendedConfusionMatrix[2][1] << "        " << extendedConfusionMatrix[2][2] << "        " << extendedConfusionMatrix[2][3] << "        " << extendedConfusionMatrix[2][4] << "\n";
	cout << "White King  " << extendedConfusionMatrix[3][0] << "       " << extendedConfusionMatrix[3][1] << "        " << extendedConfusionMatrix[3][2] << "        " << extendedConfusionMatrix[3][3] << "        " << extendedConfusionMatrix[3][4] << "\n";
	cout << "Black King  " << extendedConfusionMatrix[4][0] << "       " << extendedConfusionMatrix[4][1] << "        " << extendedConfusionMatrix[4][2] << "        " << extendedConfusionMatrix[4][3] << "        " << extendedConfusionMatrix[4][4] << "\n";

	recall = (static_cast<float>(extendedConfusionMatrix[1][1]) + static_cast<float>(extendedConfusionMatrix[2][2]) + static_cast<float>(extendedConfusionMatrix[3][3]) + static_cast<float>(extendedConfusionMatrix[4][4])) / (static_cast<float>(extendedConfusionMatrix[1][1]) + static_cast<float>(extendedConfusionMatrix[2][2]) + static_cast<float>(extendedConfusionMatrix[3][3]) + static_cast<float>(extendedConfusionMatrix[4][4]) + static_cast<float>(extendedConfusionMatrix[0][1]) + static_cast<float>(extendedConfusionMatrix[0][2]) + static_cast<float>(extendedConfusionMatrix[0][3]) + static_cast<float>(extendedConfusionMatrix[0][4])) * 100.0f;
	precision = (static_cast<float>(extendedConfusionMatrix[1][1]) + static_cast<float>(extendedConfusionMatrix[2][2]) + static_cast<float>(extendedConfusionMatrix[3][3]) + static_cast<float>(extendedConfusionMatrix[4][4])) / (static_cast<float>(extendedConfusionMatrix[1][1]) + static_cast<float>(extendedConfusionMatrix[2][2]) + static_cast<float>(extendedConfusionMatrix[3][3]) + static_cast<float>(extendedConfusionMatrix[4][4]) + static_cast<float>(extendedConfusionMatrix[1][0]) + static_cast<float>(extendedConfusionMatrix[2][0]) + static_cast<float>(extendedConfusionMatrix[3][0]) + static_cast<float>(extendedConfusionMatrix[4][0])) * 100.0f;
	accuracy = ((static_cast<float>(extendedConfusionMatrix[0][0]) + static_cast<float>(extendedConfusionMatrix[1][1]) + static_cast<float>(extendedConfusionMatrix[2][2]) + static_cast<float>(extendedConfusionMatrix[3][3]) + static_cast<float>(extendedConfusionMatrix[3][3])) / (static_cast<float>(extendedConfusionMatrix[0][0]) + static_cast<float>(extendedConfusionMatrix[0][1]) + static_cast<float>(extendedConfusionMatrix[0][2]) + static_cast<float>(extendedConfusionMatrix[0][3]) + static_cast<float>(extendedConfusionMatrix[0][4]) + static_cast<float>(extendedConfusionMatrix[1][0]) + static_cast<float>(extendedConfusionMatrix[1][1]) + static_cast<float>(extendedConfusionMatrix[1][2]) + static_cast<float>(extendedConfusionMatrix[1][3]) + static_cast<float>(extendedConfusionMatrix[1][4]) + static_cast<float>(extendedConfusionMatrix[2][0]) + static_cast<float>(extendedConfusionMatrix[2][1]) + static_cast<float>(extendedConfusionMatrix[2][2]) + static_cast<float>(extendedConfusionMatrix[2][3]) + static_cast<float>(extendedConfusionMatrix[2][4]) + static_cast<float>(extendedConfusionMatrix[3][0]) + static_cast<float>(extendedConfusionMatrix[3][1]) + static_cast<float>(extendedConfusionMatrix[3][2]) + static_cast<float>(extendedConfusionMatrix[3][3]) + static_cast<float>(extendedConfusionMatrix[3][4]) + static_cast<float>(extendedConfusionMatrix[4][0]) + static_cast<float>(extendedConfusionMatrix[4][1]) + static_cast<float>(extendedConfusionMatrix[4][2]) + static_cast<float>(extendedConfusionMatrix[4][3]) + static_cast<float>(extendedConfusionMatrix[4][4]))) * 100.0f;
	specificty = static_cast<float>(extendedConfusionMatrix[0][0]) / (static_cast<float>(extendedConfusionMatrix[0][0]) + static_cast<float>(extendedConfusionMatrix[1][0]) + static_cast<float>(extendedConfusionMatrix[2][0]) + static_cast<float>(extendedConfusionMatrix[3][0]) + static_cast<float>(extendedConfusionMatrix[4][0])) * 100.0f;
	cout << "Recall: " << recall << "%\n";
	cout << "Precison: " << precision << "%\n";
	cout << "Accuracy: " << accuracy << "%\n";
	cout << "Specificity: " << specificty << "%\n";


	int pieces[32];
	string black_pieces_filename("Media/DraughtsGame1BlackPieces.jpg");
	Mat black_pieces_image = imread(black_pieces_filename, -1);
	string white_pieces_filename("Media/DraughtsGame1WhitePieces.jpg");
	Mat white_pieces_image = imread(white_pieces_filename, -1);
	string black_squares_filename("Media/DraughtsGame1BlackSquares.jpg");
	Mat black_squares_image = imread(black_squares_filename, -1);
	string white_squares_filename("Media/DraughtsGame1WhiteSquares.jpg");
	Mat white_squares_image = imread(white_squares_filename, -1);
	string background_filename("Media/DraughtsGame1EmptyBoard.JPG");
	Mat static_background_image = imread(background_filename, -1);

	if ((!video.isOpened()) || (black_pieces_image.empty()) || (white_pieces_image.empty()) || (black_squares_image.empty()) || (white_squares_image.empty())  || (static_background_image.empty()))
	{
		// Error attempting to load something.
		if (!video.isOpened())
			cout << "Cannot open video file: " << video_filename << endl;
		if (black_pieces_image.empty())
			cout << "Cannot open image file: " << black_pieces_filename << endl;
		if (white_pieces_image.empty())
			cout << "Cannot open image file: " << white_pieces_filename << endl;
		if (black_squares_image.empty())
			cout << "Cannot open image file: " << black_squares_filename << endl;
		if (white_squares_image.empty())
			cout << "Cannot open image file: " << white_squares_filename << endl;
		if (static_background_image.empty())
			cout << "Cannot open image file: " << background_filename << endl;
	}
	else
	{
		// Sample loading of image and ground truth
		int image_index = 21;
		DraughtsBoard current_board(GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][0], GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][1], GROUND_TRUTH_FOR_BOARD_IMAGES[image_index][2]);

		// Process video frame by frame
		Mat current_frame;
		video.set(cv::CAP_PROP_POS_FRAMES, 1);
		video >> current_frame;
		double last_time = static_cast<double>(getTickCount());
		double frame_rate = video.get(cv::CAP_PROP_FPS);
		double time_between_frames = 1000.0 / frame_rate;
		while (!current_frame.empty())
		{
			double current_time = static_cast<double>(getTickCount());
			double duration = (current_time - last_time) / getTickFrequency() / 1000.0;
			int delay = (time_between_frames > duration) ? ((int)(time_between_frames - duration)) : 1;
			last_time = current_time;
			imshow("Draughts video", current_frame);
			video >> current_frame;
			char c = cv::waitKey(delay);  // If you replace delay with 1 it will play the video as quickly as possible.
		}
		cv::destroyAllWindows();
	}
}
