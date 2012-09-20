#ifndef IMAGEGRAPH_H
#define IMAGEGRAPH_H

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "GMM.h"
#include "graph.h"
#include <iostream>
#include <set>

typedef Graph<float,float,float> FloatGraph;

typedef struct _CLR{
    float R;
    float G;
    float B;
}CLR;

typedef struct _Pos{
    float Y;
    float X;
}Pos;

typedef struct _OS{
    int numSpatN;
    int idSpatN[10];

    int numTempN;
    int idTempN[10];
    int arcID[10];

    int numPixel;

    CLR meanColor;
    Pos meanPos;

    float meanFG;
    float meanBG;

    float residual;

    Pos topLeft;     //Min
    Pos bottomRight; //Max

    int iD;  //Initial ID FG:1 BG:0 Undecided:-1
    int sID; //solution ID
}OS;


class ImageGraph
{
public:
    ImageGraph();
    ~ImageGraph();
    void setFrame(cv::Mat& inpMat);
    void learnGMM(int iter);
    void learnGMM2(int iter);
    void overSegment(int m_spcount,int m_compactness);
    void solveGCut();
    void solveGCutNoGMM(ImageGraph*);
    void calcGMMProb();
    void setGroundTruth(char *fileName);
    void getSPBoundary(cv::Mat& imCanvas);
    void getResult(cv::Mat& imCanvas);
    void writeOutGraph(char*,bool);
    void writeOutOS(char*);
    void sumNodeUp();
    void getPixelWiseResidual();
    void getPixelResidual();
    void biFilt(float*,float*,float);
    void smooth();
    void smoothN();
    void getFromPrevFrame(ImageGraph* prev);
private:
    float* outGraph;
    OS* grOS;
    int* segmentID;
    int numlabels;
    std::set<int> *prevOverlaps;

    GMM bgGMM;
    GMM fgGMM;
    void mergeS(OS* grOS,int s1,int s2);
    cv::Mat im;
    cv::Mat frameDiff;

    float meanRes;


    FloatGraph* gr;
    int width;
    int height;
    float meanSum[2];//mean of GMM probabilities

};

#endif // IMAGEGRAPH_H
