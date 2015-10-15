#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

RNG rng(12345);

Mat sobel = (Mat_<float>(3,3) << -1/16., -2/16., -1/16., 0, 0, 0, 1/16., 2/16., 1/16.);

float dist(Point2i p1, Point2i p2, Vec3f p1_lab, Vec3f p2_lab, float compactness, float S);

int main (int argc, char* argv[])
{
	// Input Variables
	string input, output;
	int nx, ny;
	int m;

	// Default values
	output = "out.png";
	nx = 15;
	ny = 15;
	m = 20;

	if (argc == 1 || argc == 4) {
		cout << "SLIC superpixel segmentation" << endl;
		cout << "Usage: SLIC INPUT OUTPUT [nx ny] [m]" << endl;
		return 0;
	}
	if (argc >= 2) 
		input = argv[1];
	if (argc >= 3) 
		output = argv[2];
	if (argc >= 5) {
		nx = atoi(argv[3]);
		ny = atoi(argv[4]);
	}
	if (argc == 6)
		m = atoi(argv[5]);

	// Read in image
	Mat im = imread(input);

	if (!im.data) {
		cerr << "no image data at " << input << endl;
		return -1;
	}

	// Scale to [0,1] and l*a*b colorspace
	im.convertTo(im, CV_32F, 1/255.);
	Mat imlab;
	cvtColor(im, imlab, CV_BGR2Lab);

	int h = im.rows;
	int w = im.cols;
	int n = nx*ny;

	float dx = w / float(nx);
	float dy = h / float(ny);
	int S = (dx + dy + 1)/2; // window width

	// Initialize centers
	vector<Point2i> centers;
	for (int i = 0; i < ny; i++) {
		for (int j = 0; j < nx; j++) {
			centers.push_back( Point2f(j*dx+dx/2, i*dy+dy/2));
		}
	}

	// Initialize labels and distance maps
	vector<int> label_vec(n);
	for (int i = 0; i < n; i++)
		label_vec[i] = i*255*255/n;

	Mat labels = -1*Mat::ones(imlab.size(), CV_32S);
	Mat dists = -1*Mat::ones(imlab.size(), CV_32F);
	Mat window;
	Point2i p1, p2;
	Vec3f p1_lab, p2_lab;

	// Iterate 10 times. In practice more than enough to converge
	for (int i = 0; i < 10; i++) {
		// For each center...
		for (int c = 0; c < n; c++)
		{
			int label = label_vec[c];
			p1 = centers[c];
			p1_lab = imlab.at<Vec3f>(p1);
			int xmin = max(p1.x-S, 0);
			int ymin = max(p1.y-S, 0);
			int xmax = min(p1.x+S, w-1);
			int ymax = min(p1.y+S, h-1);

			// Search in a window around the center
			window = im(Range(ymin, ymax), Range(xmin, xmax));
			
			// Reassign pixels to nearest center
			for (int i = 0; i < window.rows; i++) {
				for (int j = 0; j < window.cols; j++) {
					p2 = Point2i(xmin + j, ymin + i);
					p2_lab = imlab.at<Vec3f>(p2);
					float d = dist(p1, p2, p1_lab, p2_lab, m, S);
					float last_d = dists.at<float>(p2);
					if (d < last_d || last_d == -1) {
						dists.at<float>(p2) = d;
						labels.at<int>(p2) = label;
					}
				}
			}
		}
	}

	// Calculate superpixel boundaries
	labels.convertTo(labels, CV_32F);
	Mat gx, gy, grad;
	filter2D(labels, gx, -1, sobel);
	filter2D(labels, gy, -1, sobel.t());
	magnitude(gx, gy, grad);
	grad = (grad > 1e-4)/255;
	Mat show = 1-grad;
	show.convertTo(show, CV_32F);

	// Draw boundaries on original image
	vector<Mat> rgb(3);
	split(im, rgb);
	for (int i = 0; i < 3; i++) 
		rgb[i] = rgb[i].mul(show);

	merge(rgb, im);

	imwrite(output, 255*im);

	return 1;
}

// Distance metric
float dist(Point2i p1, Point2i p2, Vec3f p1_lab, Vec3f p2_lab, float compactness, float S)
{
	float dl = p1_lab[0] - p2_lab[0];
	float da = p1_lab[1] - p2_lab[1];
	float db = p1_lab[2] - p2_lab[2];

	float d_lab = sqrtf(dl*dl + da*da + db*db);

	float dx = p1.x - p2.x;
	float dy = p1.y - p2.y;

	float d_xy = sqrtf(dx*dx + dy*dy);

	return d_lab + compactness/S * d_xy;
}