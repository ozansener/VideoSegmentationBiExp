#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>
#include "graph.h"
#include <vector>
#include <sys/time.h>
#include "GMM.h"
#include "SLIC.h"
#include "imagegraph.h"

using namespace std;
using namespace cv;
//Global definitions
int main(){
    namedWindow("VS");

    VideoCapture cap("/home/ozan/Desktop/VideoData/processedInVideo/erhanIn.avi");

    ImageGraph* curFrame = new ImageGraph;
    ImageGraph* prevFrame = new ImageGraph;

    Mat Frame;
    Mat PrevFrame;
    Mat dummyMat;
    Mat canvIm;

    int width = 640;
    int height = 480;

    /* Initial Frame*/
    cap>>dummyMat;
    resize(dummyMat,PrevFrame,Size(width,height));
    imwrite("raw0.jpg",PrevFrame);
    prevFrame->setFrame(PrevFrame);
    prevFrame->overSegment(1000,20);
    prevFrame->setGroundTruth("/home/ozan/Desktop/VideoData/processedInVideo/gtErhan.jpg");
    prevFrame->learnGMM(15);
    prevFrame->calcGMMProb();
    prevFrame->solveGCut();
    prevFrame->getPixelWiseResidual();
    prevFrame->writeOutGraph("first.txt",false);
    prevFrame->smooth();
    prevFrame->getPixelResidual();
    prevFrame->writeOutGraph("firstS.txt",false);
    
    PrevFrame.copyTo(canvIm);
    prevFrame->getResult(canvIm);
    imwrite("Res0.png",canvIm);

    //First Frame to Segment
    char imName[1024];
    char aName[1024];
    clock_t start = clock();

    for(int i=0;i<350;i++)
    {
        cap>>dummyMat;
        resize(dummyMat,Frame,Size(width,height));
        curFrame->setFrame(Frame);
        curFrame->overSegment(1000,20);
    
        curFrame->getFromPrevFrame(prevFrame);
        curFrame->getPixelResidual();
                
        curFrame->smoothN();
                
        if(i%80==79)
        {
            curFrame->getPixelResidual();
            curFrame->solveGCutNoGMM(prevFrame); 
            curFrame->learnGMM2(15);
            curFrame->calcGMMProb();
            curFrame->solveGCut();
            curFrame->getPixelWiseResidual();
        }else{
            curFrame->getPixelResidual();
            curFrame->solveGCutNoGMM(prevFrame); 
            curFrame->getPixelWiseResidual();
        }
        sprintf(aName,"res%d.txt",i+1);
        curFrame->writeOutGraph(aName,false);
        curFrame->smooth();

        sprintf(aName,"resS%d.txt",i+1);
        curFrame->writeOutGraph(aName,false);

        Frame.copyTo(canvIm);
        curFrame->getResult(canvIm);
        sprintf(imName,"Res%d.png",i+1);
        imwrite(imName,canvIm);

        delete prevFrame;
        prevFrame = curFrame;
        curFrame = new ImageGraph;
    }
    printf("Time elapsed: %f\n", ((double)clock() - start) / CLOCKS_PER_SEC);

    return 0;

    Mat FrameClean;
    Mat FrameDiff;
    
    clock_t sTime1;
    clock_t sTime2;
    clock_t sTime3;
    
    return 0;

}
