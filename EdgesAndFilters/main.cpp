#include <opencv2\opencv.hpp>

void getSobelXDerivativeKernel (cv::Mat& sobelXKernel);
void getSobelYDerivativeKernel (cv::Mat& sobelYKernel);
void task1(cv::Mat& input, char* imageNumber,cv::Mat& actualGradient,cv::Mat& normalizedSobelMagnitudeImage, int thresholdForMagtinudeImage);
void calcthresholdMagnitude(cv::Mat& original,cv::Mat& thresholdImage,uchar threshold);
void hough(cv::Mat& normalizedSobelMagnitudeImage,cv::Mat& actualGradient,int thresholdMagnitudeForDetectingCenters,int maxRadius,int minRadius,char * imageNumber,int minVotesNeededForCenterConsideration,int minDistanceBetweenCenters,cv::Mat& originalImage);
void sobel(cv::Mat& input,cv::Mat& normalizedSobelXDerivativeImage,cv::Mat& normalizedSobelYDerivativeImage,cv::Mat& normalizedSobelMagnitudeImage,cv::Mat& normalizedSobelGradientImage,cv::Mat& actualgradient);
void doTasks(char item);
int main( int argc, char ** argv )
{
	
	/*
	Task1 takes an image and displays the normalized sobel X derivative , Y derivative, Magnitude , gradient , and thresholded magnitude based on a threshold limit
	It also returns the normalized sobel magnitude matrix and actual sobel gradient matrix of the input image to be used for hough transform.
	Task1 also takes a label parameter to do the same stuff for different images and yet title the image windows appropriately

	Task2 or the hough function as it is named in this file takes a normalized sobel magnitude matrix and actual sobel gradient matrix and based on
	a minimum and maximum radius estimate and threshold magnitude , finds the best possible centers (determined by voting) of circles in the image.
	It also then finds out the best centers represents the hough space in 2D, and draws images of the best centers and circles on the original
	image using the best centers and best radius
	*/
	char item = '4';
	while(item!='Q'){
		printf("Which image do you want to work with [1 , 2 , 3 or 4 (see x64/Release/Images]?. Hit Q to exit\n");
		scanf("%c",&item);
		if(item=='Q')
			return 0;
		else{
			doTasks(item);
			item = cv::waitKey();
		}
	}

	
	
	return 0;
}
void doTasks(char item){

	//printf("Size of coins 1 %d %d\n",coins1.cols,coins1.rows);//441 341
	//printf("Size of coins 2 %d %d\n",coins2.cols,coins2.rows);//492 518
	//printf("Size of coins 3 %d %d\n",coins3.cols,coins3.rows);//360 318

	// Coins 1 the coins look to have diameters between 1/6th to 1/5th of breadth - so radius between between 35 to 45
	// Coins 2 the coins look to have diameters between 1/6th to 1/3th of breadth - so radius between between 40 to 90
	// Coins 1 the coins look to have diameters between 1/8th to 1/3th of breadth - so radius between between 20 to 75



	// define thresholds for normalizing and displaying the sobel magnitude image
	int thresholdForDisplayingMagnitudeImage;
	// based on rough estimation initialize the minimum and maximum radius of circles in the three images
	int minRadius;
	int maxRadius;
	// minimum votes needed to consider a point as center
	int minVotesNeededForCenterConsideration;
	// threshold magnitude for considering edges to detect centers
	int thresholdMagnitudeForDetectingCenters;
	int minDistanceBetweenCenters;

	cv::Mat input;
	cv::Mat output;
	char * imageNumber;
	switch(item){
	case '1':
		input = cv::imread("images/coins1.png", CV_LOAD_IMAGE_GRAYSCALE);
		output = cv::imread("images/coins1.png", CV_LOAD_IMAGE_COLOR);
		cv::namedWindow("Coins 1", CV_WINDOW_AUTOSIZE);
		cv::imshow("Coins 1", input);
		
		thresholdForDisplayingMagnitudeImage = 60;
		minRadius = 25;
		maxRadius = 50;
		minVotesNeededForCenterConsideration = 10;
		thresholdMagnitudeForDetectingCenters = 60;
		minDistanceBetweenCenters = 20;
		imageNumber = "1";
		break;
	case '2':
		input = cv::imread("images/coins2.png", CV_LOAD_IMAGE_GRAYSCALE);
		output = cv::imread("images/coins2.png", CV_LOAD_IMAGE_COLOR);

		cv::namedWindow("Coins 2", CV_WINDOW_AUTOSIZE);
		cv::imshow("Coins 2", input);

		thresholdForDisplayingMagnitudeImage = 120;
		minRadius = 40;
		maxRadius = 90;
		minVotesNeededForCenterConsideration = 10;
		thresholdMagnitudeForDetectingCenters = 120;
		minDistanceBetweenCenters = 40;
		imageNumber = "2";
		break;
	case '3':
		input = cv::imread("images/coins3.png", CV_LOAD_IMAGE_GRAYSCALE);
		output = cv::imread("images/coins3.png", CV_LOAD_IMAGE_COLOR);

		cv::namedWindow("Coins 3", CV_WINDOW_AUTOSIZE);
		cv::imshow("Coins 3", input);

		thresholdForDisplayingMagnitudeImage = 30;
		minRadius = 20;
		maxRadius = 80;
		minVotesNeededForCenterConsideration = 12;
		thresholdMagnitudeForDetectingCenters = 0;
		minDistanceBetweenCenters = 30;
		imageNumber = "3";
		break;
	case '4':
	input = cv::imread("images/coins4.jpg", CV_LOAD_IMAGE_GRAYSCALE);
		output = cv::imread("images/coins4.jpg", CV_LOAD_IMAGE_COLOR);

		cv::namedWindow("Coins 4", CV_WINDOW_AUTOSIZE);
		cv::imshow("Coins 4", input);

		thresholdForDisplayingMagnitudeImage = 40;
		minRadius = 15;
		maxRadius = 80;
		minVotesNeededForCenterConsideration = 10;
		thresholdMagnitudeForDetectingCenters = 40;
		minDistanceBetweenCenters = 20;
		imageNumber = "4";
		break;

	default:
		return;
	}
	printf("For task 1 we will use\n Threshold to display the threshold magnitude image = %d\n ",item,thresholdForDisplayingMagnitudeImage);

	cv::Mat normalizedSobelMagnitudeImage;
	cv::Mat actualGradient;

	actualGradient.create(input.size(), CV_64F); // will have floating values or radians
	normalizedSobelMagnitudeImage.create(input.size(), input.type()); // will have normalized values within 0 and 255

	task1(input,imageNumber,actualGradient,normalizedSobelMagnitudeImage,thresholdForDisplayingMagnitudeImage);

	printf("For task 2 we will use\n Max and Min radious = %d , %d\n Magnitude threshold as %d\n Minimum votes for center consideration as %d\n And minum distance between centers as %d\n",maxRadius,minRadius,thresholdMagnitudeForDetectingCenters,minVotesNeededForCenterConsideration,minDistanceBetweenCenters);

	printf("Working on hough space for image %c. Please wait....\n",item);
	hough(normalizedSobelMagnitudeImage,actualGradient,thresholdMagnitudeForDetectingCenters,maxRadius,minRadius,imageNumber,minVotesNeededForCenterConsideration,minDistanceBetweenCenters,output);
	printf("Centers computed for image %c\n",item);

}
/*
This function takes 
an input image 
a label string having the name/number of the image (to be used for displaying on corresponding image windows)
a reference to its thresholded normalizedSobelMagnitudeImage image  
a threshold parameter 
and a reference to its gradient image. 

The normalizedSobelMagnitudeImage and gradient is calculated on basis of
Sobel filter operation to detect edges.

The function also displays the results of Sobel filter operation in x axis,
y axis and the normalizedSobelMagnitudeImage , thresholded normalizedSobelMagnitudeImage and gradient images.
*/
void task1(cv::Mat& input, char* imageNumber,cv::Mat& actualGradient,cv::Mat& normalizedSobelMagnitudeImage, int thresholdForMagtinudeImage){

	cv::Mat normalizedSobelXDerivativeImage; // 
	cv::Mat normalizedSobelGradientImage;
	cv::Mat normalizedSobelYDerivativeImage;

	normalizedSobelXDerivativeImage.create(input.size(), input.type()); 
	normalizedSobelGradientImage.create(input.size(), input.type());
	normalizedSobelYDerivativeImage.create(input.size(), input.type());

	// Apply sobel to get x derivative,y derivative,magnitude and gradient images against an input image -
	sobel(input,normalizedSobelXDerivativeImage,normalizedSobelYDerivativeImage,normalizedSobelMagnitudeImage,normalizedSobelGradientImage,actualGradient);
	// -

	// Display the images - 
	
	std::string labelDx ; // label for X Derivative image
	std::string labelDy ; // label for Y Derivative image
	std::string labelM ; // label for Magnitude Derivative image
	std::string labelG ; // label for Gradient Derivative image

	labelDx.append(imageNumber);
	labelDy.append(imageNumber);
	labelM.append(imageNumber);
	labelG.append(imageNumber);

	labelDx.append(" X Derivative");
	labelDy.append(" Y Derivative");
	labelM.append(" Magnitude");
	labelG.append(" Gradient");

	cv::namedWindow(labelDx, CV_WINDOW_AUTOSIZE);
	cv::namedWindow(labelDy, CV_WINDOW_AUTOSIZE);
	cv::namedWindow(labelM, CV_WINDOW_AUTOSIZE);
	cv::namedWindow(labelG, CV_WINDOW_AUTOSIZE);
	cv::imshow(labelDx, normalizedSobelXDerivativeImage);
	cv::imshow(labelDy, normalizedSobelYDerivativeImage);
	cv::imshow(labelM, normalizedSobelMagnitudeImage);
	cv::imshow(labelG, normalizedSobelGradientImage);
	
	// -
	cv::Mat thresholdmagnitudeImage;

	// Get a threshold image from the magnitude image based on the threshold parameter -
	thresholdmagnitudeImage.create(normalizedSobelMagnitudeImage.size(), normalizedSobelMagnitudeImage.type());
	calcthresholdMagnitude(normalizedSobelMagnitudeImage,thresholdmagnitudeImage,thresholdForMagtinudeImage);
	// -

	// Display thresholded magnitude image
	std::string labelT ;
	labelT.append(imageNumber);
	labelT.append(" Threshold Magnitude");

	cv::namedWindow(labelT, CV_WINDOW_AUTOSIZE);
	cv::imshow(labelT, thresholdmagnitudeImage);
	//printf("rows cols %d %d\n",thresholdmagnitude.rows,thresholdmagnitude.cols);
	// -
	
	//sobelXDerivativeImage.release();
	//sobelYDerivativeImage.release();
	//normalizedSobelMagnitudeImage.release();
	
}
/*
This function releases the hough 3D space
*/
void releaseTrippleArray(int ***kernel, int ysize,int xsize){

	for(int i=0; i < ysize; i++)
	{
		for(int j=0; j < xsize; j++)
		{
			delete [] kernel[i][j];
		}
		delete [] kernel[i];
	}

	delete [] kernel;

}
/*
This function takes a sobel magnitude and gradient image and draws the centers of circles in the original image
Its works like the following
1)First it initializes a 3D array for the hough space. The Y and X dimentions are same as the input image or normalized sobel magnitude
image - whereas the depth is equal to the (max - min )radius of cicles in the image

2)It then loops though all pixels in the normalized sobel magnitude image and chooses pixels which are higher than input threshold

3)Once such a pixel is identified,it gets the actual gradient at that pixel from the actual gradient matrix

4) It next votes for all elements in the hough space whose y and x are in line with the gradient
and radius is between min and max radius. 
To find the inline with gradient Y and X co-ordinates the algorithm uses the fact that
center probableXCenter,probableYCenter will follow 
either the 						
probableYCenter = yImage - (int)(rad*sin(theta));
probableXCenter = xImage - (int)(rad*cos(theta));
Or
probableYCenter = yImage + (int)(rad*sin(theta));
probableXCenter = xImage + (int)(rad*cos(theta));

This comes from simple geometry where the theta angle and magnitude is between PI and -PI and 
that the quadrant tell you which of df/dy or df/dx was positive or negative

5) Once all the voting is complete, the method then loops through the hough space looking for pixels which have more than
a threshold number of votes on particular depths. These pixles are marked as the probable centers in a centers image .

6) It also shows the hough space as it is by summing the votes on every pixel (to represent a 3D variant). We can
clearly see that there is a cloud at the probable center area and only few pixles which are most bright. 
Indicating best centers

7) Then to find the best centers, this method goes through all the pixesl in the probable centers image and
when it hits a pixel which is 255 - it does a smaller iteration over a rectangle (area of pixels which 
all all probable centers for a cicle) - which is determined by the minimum distance between circle centers
which is passed as an input. This minimum distance between centers is again an approximatation based on the original
image. For each such small area , it finds the pixesl with the most votes and then the outer iteration
resumes. The trick here is to set all the probabcle centers to 0 within that small area - so that
the outer interation througout the probable centers image does not pick up those pixels again.

8) It then plots the circles with the best centers on the original image
*/
void hough(cv::Mat& normalizedSobelMagnitudeImage,cv::Mat& actualGradient,int thresholdMagnitudeForDetectingCenters,int maxRadius,int minRadius,char * imageNumber,int minVotesNeededForCenterConsideration,int minDistanceBetweenCenters,cv::Mat& originalImage){
	
	cv::Mat centers; // create a centers image to display the centers later
	centers.create(normalizedSobelMagnitudeImage.size(), normalizedSobelMagnitudeImage.type());
	int rangeRadius = maxRadius - minRadius;
	const double PI = atan(1.0)*4;
	
	// Create and initialize a 3D hough array
	int*** houghArray;
	//printf("rows cols %d %d\n",normalizedSobelMagnitudeImage.rows,normalizedSobelMagnitudeImage.cols);
	houghArray = new int**[normalizedSobelMagnitudeImage.rows];
	for(int i = 0 ; i < normalizedSobelMagnitudeImage.rows; i++ ){
		houghArray[i] = new int*[normalizedSobelMagnitudeImage.cols];
		for(int j = 0 ; j < normalizedSobelMagnitudeImage.cols; j++ ){
			houghArray[i][j] = new int[rangeRadius]; 
			for(int k = 0 ; k < rangeRadius ; k++ ){
				houghArray[i][j][k] = 0;
			}
		}
	}
	int vote;
	double theta;
	int mag;
	int pixelcount = 0;
	int maxvote = 0;
	int probableYCenterPlus,probableXCenterPlus,probableYCenterMinus,probableXCenterMinus;

	// Loop through the normalizedSobelMagnitudeImage image and check for pixels which are over the threshold supplied
	for(int yImage = 0 ; yImage < normalizedSobelMagnitudeImage.rows; yImage++ ){// y loop of normalizedSobelMagnitudeImage image
		for(int xImage = 0 ; xImage < normalizedSobelMagnitudeImage.cols; xImage++ ){// x loop of normalizedSobelMagnitudeImage image
			mag = (int)normalizedSobelMagnitudeImage.at<uchar>(yImage,xImage); // get magnitude of magnitude image at this pixel
			if(mag>thresholdMagnitudeForDetectingCenters){
				pixelcount++;
				// this pixel now has a required magnitude which can be considered for voting in the hough space
				
				theta = (double)actualGradient.at<double>(yImage,xImage); // get gradient of gradient image at this pixel
				//if(yImage==501 && xImage==37){
					//printf("%f\n",theta);
				//}

				for(int rad = 0 ; rad < rangeRadius; rad++ ){ // hough space radius/depth loop
									
					probableYCenterPlus = yImage + (int)((rad+minRadius)*sin(theta));
					probableXCenterPlus = xImage + (int)((rad+minRadius)*cos(theta));
					probableYCenterMinus = yImage - (int)((rad+minRadius)*sin(theta));
					probableXCenterMinus = xImage - (int)((rad+minRadius)*cos(theta));
					vote = (int)(mag-thresholdMagnitudeForDetectingCenters);
					if(probableYCenterPlus>0 && probableYCenterPlus<normalizedSobelMagnitudeImage.rows && probableXCenterPlus>0 && probableXCenterPlus<normalizedSobelMagnitudeImage.cols){

						//if(houghArray[probableYCenter][probableXCenter][rad]>3)
							//printf("Incrementing vote %d %d %d %d\n",probableYCenter,probableXCenter,rad,houghArray[probableYCenter][probableXCenter][rad]+1);

						houghArray[probableYCenterPlus][probableXCenterPlus][rad] = houghArray[probableYCenterPlus][probableXCenterPlus][rad] + 1;
						if(houghArray[probableYCenterPlus][probableXCenterPlus][rad] > maxvote)
							maxvote = houghArray[probableYCenterPlus][probableXCenterPlus][rad];

						//printf("Incrementing vote %d %d %d\n",probableYCenter,probableXCenter,rad);
					}
					if(probableYCenterMinus>0 && probableYCenterMinus<normalizedSobelMagnitudeImage.rows && probableXCenterMinus>0 && probableXCenterMinus<normalizedSobelMagnitudeImage.cols){

						//if(houghArray[probableYCenter][probableXCenter][rad]>3)
							//printf("Incrementing vote %d %d %d %d\n",probableYCenter,probableXCenter,rad,houghArray[probableYCenter][probableXCenter][rad]+1);

						houghArray[probableYCenterMinus][probableXCenterMinus][rad] = houghArray[probableYCenterMinus][probableXCenterMinus][rad] + 1;
						if(houghArray[probableYCenterMinus][probableXCenterMinus][rad] > maxvote)
							maxvote = houghArray[probableYCenterMinus][probableXCenterMinus][rad];
						//printf("Incrementing vote %d %d %d\n",probableYCenter,probableXCenter,rad);
					}
								
				}
			}
		}
	}
	//printf("Max vote %d\n",maxvote);
	double maxHoughRadSummation = 0;
	double houghRadSummation = 0;
	double maxHoughRadLogSummation = 0;


	cv::Mat houghSpaceVals =  cv::Mat::zeros(centers.rows, centers.cols, CV_64F);
	//cv::Mat houghSpaceLogVals =  cv::Mat::zeros(centers.rows, centers.cols, CV_64F);
	cv::Mat houghSpaceImage = cv::Mat::zeros(centers.rows, centers.cols, centers.type()); // create a centers image to display the centers later
	// To represent hough space in 3D add up the votes 
	for(int yCenter = 0 ; yCenter < normalizedSobelMagnitudeImage.rows; yCenter++ ){ // hough space y loop
		for(int xCenter = 0 ; xCenter < normalizedSobelMagnitudeImage.cols; xCenter++ ){// hough space x loop
			houghRadSummation = 0;
			for(int rad = 0 ; rad < rangeRadius; rad++ ){ // hough space radius loop
				houghRadSummation = houghRadSummation + houghArray[yCenter][xCenter][rad];
			}
			houghSpaceVals.at<double>(yCenter,xCenter) = houghRadSummation;
			//houghSpaceLogVals.at<double>(yCenter,xCenter) = log(houghRadSummation);
			if(houghRadSummation>maxHoughRadSummation)
				maxHoughRadSummation = houghRadSummation;

			//if(log(houghRadSummation)>maxHoughRadLogSummation)
				//maxHoughRadLogSummation = log(houghRadSummation);

		}
	}
	//Normalize the hough space based on max range of votes
	for(int yCenter = 0 ; yCenter < houghSpaceVals.rows; yCenter++ ){ // hough space y loop
		for(int xCenter = 0 ; xCenter < houghSpaceVals.cols; xCenter++ ){// hough space x loop
			houghSpaceImage.at<uchar>(yCenter,xCenter) = (int)(houghSpaceVals.at<double>(yCenter,xCenter)*(255/maxHoughRadSummation));
			//houghSpaceLogVals.at<uchar>(yCenter,xCenter) = (int)(houghSpaceLogVals.at<double>(yCenter,xCenter)*(255/maxHoughRadLogSummation));
		}
	}
	std::string labelH ; // label for Hough Space image

	labelH.append(imageNumber);
	labelH.append(" Hough Space");

	cv::namedWindow(labelH, CV_WINDOW_AUTOSIZE);
	cv::imshow(labelH, houghSpaceImage);

	//std::string labelL ; // label for Log image

	//labelL.append(imageNumber);
	//labelL.append(" Hough Log Space");

	//cv::namedWindow(labelL, CV_WINDOW_AUTOSIZE);
	//cv::imshow(labelL, houghSpaceLogVals);

	// Get the center regions
	// Get y , x and radius which have at least minVotesNeededForCenterConsideration votes and mark them as white in the center image
	for(int yCenter = 0 ; yCenter < normalizedSobelMagnitudeImage.rows; yCenter++ ){ // hough space y loop
		for(int xCenter = 0 ; xCenter < normalizedSobelMagnitudeImage.cols; xCenter++ ){// hough space x loop
			centers.at<uchar>(yCenter,xCenter) = (uchar)0;
			for(int rad = 0 ; rad < rangeRadius; rad++ ){ // hough space radius loop
				if(houghArray[yCenter][xCenter][rad] > minVotesNeededForCenterConsideration){
					//printf("Potential center (Y X Radius) %d %d %d %d\n",yCenter,xCenter,rad+minRadius,houghArray[yCenter][xCenter][rad]);
					centers.at<uchar>(yCenter,xCenter) = (uchar)255;
				}
			}
		}
	}

	cv::Mat perfectCenters = cv::Mat::zeros(centers.rows, centers.cols, centers.type());; // create a best centers image to display
		// Show the centers image -
	std::string labelC ; // label for Probable Centers image

	labelC.append(imageNumber);
	labelC.append(" Probable Centers");

	cv::namedWindow(labelC, CV_WINDOW_AUTOSIZE);
	cv::imshow(labelC, centers);
	// -
	// Get the best centers based on the logic described in the method info
	int maxVoteOnHoughX,maxVoteOnHoughY,maxVoteOnHoughRad, maxVote;
	int yHough , xHough , radHough , maxxHough , maxyHough,minxHough , minyHough;
	for(int yCenter = 0 ; yCenter < centers.rows; yCenter++ ){ // center space y loop
		for(int xCenter = 0 ; xCenter < centers.cols; xCenter++ ){// hough space x loop
			if(centers.at<uchar>(yCenter,xCenter) == 255){
				//printf("Starting at	%d %d\n",yCenter,xCenter);
				centers.at<uchar>(yCenter,xCenter) = (uchar)0;
				maxVoteOnHoughX=0;
				maxVoteOnHoughY=0;
				maxVoteOnHoughRad=0;
				maxVote = 0;
				maxyHough = yCenter + minDistanceBetweenCenters;
				if(maxyHough>normalizedSobelMagnitudeImage.rows)maxyHough=normalizedSobelMagnitudeImage.rows;
				maxxHough = xCenter + minDistanceBetweenCenters;
				if(maxxHough>normalizedSobelMagnitudeImage.cols)maxxHough=normalizedSobelMagnitudeImage.cols;

				minyHough = yCenter - minDistanceBetweenCenters;
				if(minyHough<0)minyHough=0;
				minxHough = xCenter - minDistanceBetweenCenters;
				if(minxHough<0)minxHough=0;
				//printf("Ending check at %d %d\n",minyHough,minxHough);
				for(yHough = minyHough ; yHough < maxyHough; yHough++ ){ // hough space y loop
					for(xHough = minxHough ; xHough < maxxHough; xHough++ ){// hough space x loop
						centers.at<uchar>(yHough,xHough) = (uchar)0;
						for(radHough=0 ; radHough < rangeRadius ; radHough++){
							if(houghArray[yHough][xHough][radHough] > maxVote ){
								maxVote = houghArray[yHough][xHough][radHough];
								maxVoteOnHoughX=xHough;
								maxVoteOnHoughY=yHough;
								maxVoteOnHoughRad=radHough+minRadius;
							}
						}
					}
				}
				//printf("Ending check at %d %d\n",yHough,xHough);
				perfectCenters.at<uchar>(maxVoteOnHoughY,maxVoteOnHoughX) = (uchar)255;
				cv::Point pt =  cv::Point(maxVoteOnHoughX,maxVoteOnHoughY);
				cv::circle(originalImage, pt,maxVoteOnHoughRad,cv::Scalar( 255, 0, 0 ), 2, 8, 0 );

				xCenter = xHough;
				//printf("Resume at %d %d\n",yCenter,xCenter);
			}
		}
	}

	// Show the probable centers and best centers image -
	std::string labelPC ; // label for best Centers image

	labelPC.append(imageNumber);
	labelPC.append(" Best Centers");

	cv::namedWindow(labelPC, CV_WINDOW_AUTOSIZE);
	cv::imshow(labelPC, perfectCenters);
	// -
	// Show the original image with best circles -
	std::string labelPCOnOriginalImage ; // label for Circles on Original image

	labelPCOnOriginalImage.append(imageNumber);
	labelPCOnOriginalImage.append(" Best Circles on Original");

	cv::namedWindow(labelPCOnOriginalImage, CV_WINDOW_AUTOSIZE);
	cv::imshow(labelPCOnOriginalImage, originalImage);
	// -

	// Release the hough space
	releaseTrippleArray(houghArray,normalizedSobelMagnitudeImage.rows,normalizedSobelMagnitudeImage.cols);
	//houghSpace.release();
}
/*
This function initializes a Sobel X Derivative kernel
*/

void getSobelXDerivativeKernel (cv::Mat& sobelXKernel){
	
	sobelXKernel.at<int>(0,0) = -1;
	sobelXKernel.at<int>(0,2) = 1;
	sobelXKernel.at<int>(1,0) = -2;
	sobelXKernel.at<int>(1,2) = 2;
	sobelXKernel.at<int>(2,0) = -1;
	sobelXKernel.at<int>(2,2) = 1;

}
/*
This function initializes a Sobel Y Derivative kernel
*/
void getSobelYDerivativeKernel (cv::Mat& sobelYKernel){
	
	sobelYKernel.at<int>(0,0) = -1;
	sobelYKernel.at<int>(2,0) = 1;
	sobelYKernel.at<int>(0,1) = -2;
	sobelYKernel.at<int>(2,1) = 2;
	sobelYKernel.at<int>(0,2) = -1;
	sobelYKernel.at<int>(2,2) = 1;

}
/*
This function takes any input image and returns a output image which has
pixels set to 255 if the input pixel was > Threshold
and set to 0 if the input pixel was < Threshold
*/
void calcthresholdMagnitude(cv::Mat& original,cv::Mat& thresholdImage,uchar threshold){
	uchar val;
	int noOfPixlesWithHighMagnitude = 0;
	for ( int i = 0; i < original.rows; i++ )
		{	
			for( int j = 0; j < original.cols; j++ )
			{
				val = original.at<uchar>(i,j);
				if(val > threshold){
					thresholdImage.at<uchar>(i,j) = (uchar)255;
					noOfPixlesWithHighMagnitude++;
				}else{
					thresholdImage.at<uchar>(i,j) = (uchar)0;
				}
				//if(i == 501 && j == 37){
					//thresholdImage.at<uchar>(i,j) = (uchar)255;
				//}
			}
	}
	printf("No of pixels with high magnitude %d",noOfPixlesWithHighMagnitude);
}
/*
This function works in the following manner
1) It initializes two Sobel operator kernels for X and Y derivatives
2) It pads the input image to apply the kernels from the extreme corner co-ordinates
3) It creates and initializes (with zero) 3 matrices of floating/double variable types , each of size 
equal to the size of the input image, and one each for storing the values of x Derivative,y Derivative,magnitude 
and one such matrix for storing the gradient is passed as reference to this function
4) It creates and initializes four local variables (mostly to keep track of sums and maximums and minimums of the above four quantities) for normalizing
5) It goes through each pixel of the input image and applies the sobel X kernel and y kernel individually on the padded input iamge to get 
values of x Derivative,y Derivative,magnitude and gradient on each pixel of the input image
6) Once it has the sobel outputs of x Derivative,y Derivative,magnitude and gradient - it normalizes their values by mapping them from
their individual range to [0 255]. Once normalized each value is stored in the corresponding normalized image matrix
7) Along with other matrices , the actual gradient matrix and normalized sobel magnitude matrix is returned to be used for a second task

*/
void sobel(cv::Mat& input,cv::Mat& normalizedSobelXDerivativeImage,cv::Mat& normalizedSobelYDerivativeImage,cv::Mat& normalizedSobelMagnitudeImage,cv::Mat& normalizedSobelGradientImage,cv::Mat& actualgradient){
	
	cv::Mat sobelXKernel = cv::Mat::zeros(3, 3, CV_64F);
	cv::Mat sobelYKernel = cv::Mat::zeros(3, 3, CV_64F);

	getSobelXDerivativeKernel(sobelXKernel);
	getSobelYDerivativeKernel(sobelYKernel);

	cv::Mat paddedInput;
	cv::copyMakeBorder( input, paddedInput, 
		1, 1, 1, 1,
		cv::BORDER_REPLICATE );


	cv::Mat xd = cv::Mat::zeros(input.rows, input.cols, CV_64F);
	cv::Mat yd = cv::Mat::zeros(input.rows, input.cols, CV_64F);
	cv::Mat mag = cv::Mat::zeros(input.rows, input.cols, CV_64F);

	double sumxDerivative = 0;
	double sumyDerivative = 0;

	// Following the so,bel kernls, individually  the max and min of (df/dx) or (df/dy) is 255*4 and 255*-4
	double maxxDerivative = (double)-4*255;
	double maxyDerivative = (double)-4*255;
	double minxDerivative = (double)255*4;
	double minyDerivative = (double)255*4;
	double rangexderivate;
	double rangeyderivate;

	double minMag = (double)(sqrt((double)2)*255*4); // max value of sqrt((df/dy)^2 + (df/dx)^2). Individually  the max of (df/dx) or (df/dy) is 255*4
	double maxMag = 0; // min value can only be zero as there are squares within the square root
	double tempmag;
	double rangemag;

	const double PI = atan(1.0)*4; 
	// gradient is atan2 of (df/dy)/(df/dx) - which can range from -PI to PI
	double minGrad = PI; 
	double maxGrad = -PI;
	double tempGrad = 0;
	double rangegrad;

	int imageval ;
	int xkernalval ;
	int ykernalval ;
	int imagex ;
	int imagey ;
	int kernelx;
	int kernely;
	int mulx;
	int muly;

	// find the x Derivative,y Derivative,magnitude and gradient -
	for ( int i = 0; i < input.rows; i++ )
		{	
			for( int j = 0; j < input.cols; j++ )
			{
				sumxDerivative = 0;
				sumyDerivative = 0;

				for( int m = -1; m <= 1; m++ )
				{
					for( int n = -1; n <= 1; n++ )
					{
						// find the correct indices we are using
						imagey = i + 1 - m;
						imagex = j + 1 - n;
						kernely = m + 1;
						kernelx = n + 1;
					
						// get the values from the padded image and the kernel
						imageval = ( int ) paddedInput.at<uchar>( imagey, imagex );
						//printf("Image Val %d\n",imageval);
					
						//int kernalval = kernel.at<int>( kernelx, kernely );
						xkernalval = sobelXKernel.at<int>( kernely, kernelx );
						
						ykernalval = sobelYKernel.at<int>( kernely, kernelx );

						//printf("Kernel Valxy %d %d\n",xkernalval,ykernalval);

						mulx = imageval * xkernalval;
						muly = imageval * ykernalval;

						//printf("Multiply Valxy %d %d\n",mulx,muly);
						//printf("sumxDerivative %d\n",sumxDerivative);
					
						sumxDerivative += (double)mulx;
						sumyDerivative += (double)muly;
					}
				
				}
				// Store the actual magnitude and gradient in temporary arrays to be normalized later
				tempmag = sqrt((sumyDerivative*sumyDerivative) + (sumxDerivative*sumxDerivative));

				tempGrad = atan2(sumyDerivative,sumxDerivative) ;

				//if((tempGrad>PI/2 || tempGrad <-PI/2)&& sumxDerivative!=0)
				//	printf("%lf %lf %lf\n",sumyDerivative,sumxDerivative,tempGrad);

				// Get the max and min of x Derivative,y Derivative,magnitude and gradient -
				if(sumxDerivative>maxxDerivative)
					maxxDerivative = sumxDerivative;

				if(sumxDerivative<minxDerivative)
					minxDerivative = sumxDerivative;

				if(sumyDerivative>maxyDerivative)
					maxyDerivative = sumyDerivative;

				if(sumyDerivative<minyDerivative)
					minyDerivative = sumyDerivative;

				if(tempmag<minMag)
					minMag = tempmag;

				if(tempmag>maxMag)
					maxMag = tempmag;

				if(tempGrad<minGrad)
					minGrad = tempGrad;

				if(tempGrad>maxGrad)
					maxGrad = tempGrad;

				// -

				//printf("Max Min Max Min %d %d %d %d",maxxDerivative,minxDerivative,maxyDerivative,minyDerivative);

				//if(sumxDerivative > 100 || sumxDerivative < -100)
					//printf("X Y %d %d %d %d\n",sumxDerivative,sumyDerivative,i,j);

				xd.at<double>(i, j) = sumxDerivative;
				yd.at<double>(i, j) = sumyDerivative;
				mag.at<double>(i, j) = tempmag;
				actualgradient.at<double>(i, j) = tempGrad;
			}
		
		}
		// -
	
		// Get the ranges -
		rangexderivate = maxxDerivative - minxDerivative;
		rangeyderivate = maxyDerivative - minyDerivative;
		rangemag = maxMag - minMag;
		rangegrad = maxGrad - minGrad;
		// - 

		//printf("Max Min Max Min %f %f %f %f %f %f %f %f %f %f\n",maxxDerivative,minxDerivative,maxyDerivative,minyDerivative,rangexderivate,rangeyderivate,rangemag,rangegrad,maxGrad ,minGrad);
		//printf("Max Min Range %f %f %f\n",maxGrad ,minGrad,rangegrad);
		//printf("Max Min Max Min %f",maxxDerivative);
		// find the derivatives

		// Normalize X Derivate , Y Derivative , Magnitude and Gradient by mapping their range into [0 255]-
		double xval, yval , xfactor , yfactor, magFactor , magval , gradval , gradFactor;
		xfactor = ((double)255/rangexderivate);
		yfactor = ((double)255/rangeyderivate);
		magFactor = ((double)255/rangemag);
		gradFactor = ((double)255/rangegrad);
		for ( int i = 0; i < xd.rows; i++ )
		{	
			for( int j = 0; j < xd.cols; j++ )
			{
				xval = xd.at<double>(i, j);
				yval = yd.at<double>(i, j);
				magval = mag.at<double>(i, j);
				gradval = actualgradient.at<double>(i, j);
				yval = (yval - minyDerivative)*yfactor;
				xval = (xval - minxDerivative)*xfactor;
				magval = (magval - minMag)*magFactor;
				//if(i==501 && j == 37)
					//printf("%f %f\n",gradval,gradval*180/PI);

				gradval = (gradval - minGrad)*gradFactor;
				

				normalizedSobelXDerivativeImage.at<uchar>(i, j) = (uchar)xval;
				normalizedSobelYDerivativeImage.at<uchar>(i, j) = (uchar)yval;
				normalizedSobelMagnitudeImage.at<uchar>(i, j) = (uchar)magval;
				normalizedSobelGradientImage.at<uchar>(i, j) = (uchar)gradval;

			}
			
			// Release matrices not referenced further -
			
			//sobelXKernel.release();
			//sobelYKernel.release();

			//paddedInput.release();
			//xd.release();
			//yd.release();
			//mag.release();
			// -

		}
		// -

}