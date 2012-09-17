#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "GMM.h"
#include "graph.h"
#include "SLIC.h"
#include "imagegraph.h"
#include <iostream>
#include <fstream>
#include <set>
using namespace cv;
using namespace std;

ImageGraph::ImageGraph()
{
}

ImageGraph::~ImageGraph(){
    delete [] grOS;
    delete [] segmentID;

    fgGMM.initializeAndClear(3,5);
    bgGMM.initializeAndClear(3,5);

    delete gr;
    delete [] outGraph;
}

void ImageGraph::mergeS(OS* grOS,int s1,int s2){
    if(s1==s2)
        return;
    bool flag=true;
    for(int i=0;i<grOS[s1].numSpatN;i++)
        if(grOS[s1].idSpatN[i]==s2)
            flag=false;
    if(flag)
        if(grOS[s1].numSpatN<10)
        {
            grOS[s1].idSpatN[grOS[s1].numSpatN]=s2;
            grOS[s1].numSpatN++;
        }
    flag=true;
    for(int i=0;i<grOS[s2].numSpatN;i++)
        if(grOS[s2].idSpatN[i]==s1)
            flag=false;
    if(flag)
        if(grOS[s2].numSpatN<10)
        {
            grOS[s2].idSpatN[grOS[s2].numSpatN]=s1;
            grOS[s2].numSpatN++;
        }
}


void ImageGraph::setFrame(cv::Mat & inpMat){
    im=inpMat;
    width = im.cols;
    height = im.rows;

    grOS = new OS[2];
    segmentID = new int[width*height];

    fgGMM.initializeAndClear(3,5);
    bgGMM.initializeAndClear(3,5);

    gr = new FloatGraph(1,10);

    outGraph = new float[width*height];
}


void ImageGraph::overSegment(int m_spcount,int m_compactness){
    //SLIC
    int r,g,b;
    unsigned int* ubuff = new unsigned int[width*height];
    numlabels = 0;
    SLIC sl;

    for(int xS=0;xS<width;xS++)
        for(int yS=0;yS<height;yS++)
        {
            r=im.at<cv::Vec3b>(yS,xS)[2];
            g=im.at<cv::Vec3b>(yS,xS)[1];
            b=im.at<cv::Vec3b>(yS,xS)[0];
            ubuff[yS*width+xS]= (r<<16) + (g<<8) + (b);
        }
    sl.DoSuperpixelSegmentation_ForGivenK(ubuff, width, height, segmentID, numlabels, m_spcount, m_compactness);
    //COMPUTE STATISTICS

    delete [] grOS;
    grOS = new OS[numlabels];
    for(int k=0;k<numlabels;k++)
    {
        grOS[k].meanPos.X=0;
        grOS[k].meanPos.Y=0;

        grOS[k].numSpatN=0;

        grOS[k].meanColor.R=0;
        grOS[k].meanColor.G=0;
        grOS[k].meanColor.B=0;

        grOS[k].meanFG=0;
        grOS[k].meanBG=0;

        grOS[k].numPixel=0;

        grOS[k].topLeft.X = 0;
        grOS[k].topLeft.Y = 0;

        grOS[k].bottomRight.X = 1000;
        grOS[k].bottomRight.Y = 1000;

        grOS[k].sID=-1;
        grOS[k].iD=-1;
    }

    for(int xS=1;xS<width-1;xS++)
        for(int yS=1;yS<height-1;yS++)
        {
            if(segmentID[(yS)*width+(xS)]!=segmentID[(yS)*width+(xS+1)])
                mergeS(grOS,segmentID[(yS)*width+(xS)],segmentID[(yS)*width+(xS+1)]);
            if(segmentID[(yS)*width+(xS)]!=segmentID[(yS)*width+(xS-1)])
                mergeS(grOS,segmentID[(yS)*width+(xS)],segmentID[(yS)*width+(xS-1)]);
            if(segmentID[(yS)*width+(xS)]!=segmentID[(yS+1)*width+(xS)])
                mergeS(grOS,segmentID[(yS)*width+(xS)],segmentID[(yS+1)*width+(xS)]);
            if(segmentID[(yS)*width+(xS)]!=segmentID[(yS-1)*width+(xS)])
                mergeS(grOS,segmentID[(yS)*width+(xS)],segmentID[(yS-1)*width+(xS)]);

            if(xS<grOS[segmentID[(yS)*width+(xS)]].bottomRight.X)
                grOS[segmentID[(yS)*width+(xS)]].bottomRight.X=xS;
            if(xS>grOS[segmentID[(yS)*width+(xS)]].topLeft.X)
                grOS[segmentID[(yS)*width+(xS)]].topLeft.X=xS;
            if(yS<grOS[segmentID[(yS)*width+(xS)]].bottomRight.Y)
                grOS[segmentID[(yS)*width+(xS)]].bottomRight.Y=yS;
            if(yS>grOS[segmentID[(yS)*width+(xS)]].topLeft.Y)
                grOS[segmentID[(yS)*width+(xS)]].topLeft.Y=yS;

            grOS[segmentID[(yS)*width+(xS)]].meanPos.X=(grOS[segmentID[(yS)*width+(xS)]].meanPos.X*grOS[segmentID[(yS)*width+(xS)]].numPixel+xS)/(grOS[segmentID[(yS)*width+(xS)]].numPixel+1);
            grOS[segmentID[(yS)*width+(xS)]].meanPos.Y=(grOS[segmentID[(yS)*width+(xS)]].meanPos.Y*grOS[segmentID[(yS)*width+(xS)]].numPixel+yS)/(grOS[segmentID[(yS)*width+(xS)]].numPixel+1);

            grOS[segmentID[(yS)*width+(xS)]].meanColor.B =(grOS[segmentID[(yS)*width+(xS)]].meanColor.B*grOS[segmentID[(yS)*width+(xS)]].numPixel+im.at<cv::Vec3b>(yS,xS)[2])/(grOS[segmentID[(yS)*width+(xS)]].numPixel+1);
            grOS[segmentID[(yS)*width+(xS)]].meanColor.G =(grOS[segmentID[(yS)*width+(xS)]].meanColor.G*grOS[segmentID[(yS)*width+(xS)]].numPixel+im.at<cv::Vec3b>(yS,xS)[1])/(grOS[segmentID[(yS)*width+(xS)]].numPixel+1);
            grOS[segmentID[(yS)*width+(xS)]].meanColor.R =(grOS[segmentID[(yS)*width+(xS)]].meanColor.R*grOS[segmentID[(yS)*width+(xS)]].numPixel+im.at<cv::Vec3b>(yS,xS)[0])/(grOS[segmentID[(yS)*width+(xS)]].numPixel+1);
            grOS[segmentID[(yS)*width+(xS)]].numPixel++;
        }

}


void ImageGraph::learnGMM(int iter){
    fgGMM.clear();
    bgGMM.clear();

    float* inVecBG = new float[width*height*3];
    float* inVecFG = new float[width*height*3];

    int numBG=0;
    int numFG=0;

    for(int x=0;x<width;x=x+3)
        for(int y=0;y<height;y=y+3)
        {
            if(grOS[segmentID[y*width+x]].iD==0)
            {
                inVecBG[numBG*3+0]=im.at<cv::Vec3b>(y,x)[0];
                inVecBG[numBG*3+1]=im.at<cv::Vec3b>(y,x)[1];
                inVecBG[numBG*3+2]=im.at<cv::Vec3b>(y,x)[2];
                numBG++;
            }else if(grOS[segmentID[y*width+x]].iD==1)
            {
                inVecFG[numFG*3+0]=im.at<cv::Vec3b>(y,x)[0];
                inVecFG[numFG*3+1]=im.at<cv::Vec3b>(y,x)[1];
                inVecFG[numFG*3+2]=im.at<cv::Vec3b>(y,x)[2];
                numFG++;
            }
        }


    fgGMM.insertData(inVecFG,numFG);
    bgGMM.insertData(inVecBG,numBG);
    fgGMM.iterateGMM(iter);
    bgGMM.iterateGMM(iter);
    delete [] inVecBG;
    delete [] inVecFG;
}

void ImageGraph::calcGMMProb(){
    meanSum[0] = 0;
    meanSum[1] = 0;
    
    int numF=0;
    int numB=0;

    float inVec[3];
    for(int x=0;x<width;x++)
        for(int y=0;y<height;y++)
        {
            inVec[0]=im.at<cv::Vec3b>(y,x)[0];
            inVec[1]=im.at<cv::Vec3b>(y,x)[1];
            inVec[2]=im.at<cv::Vec3b>(y,x)[2];
            grOS[segmentID[y*width+x]].meanBG+=bgGMM.getLikelihood2(inVec);
            grOS[segmentID[y*width+x]].meanFG+=fgGMM.getLikelihood2(inVec);
        }

    for(int i=0;i<numlabels;i++)
    {
        if(grOS[i].numPixel!=0)
        {
            grOS[i].meanBG=grOS[i].meanBG/grOS[i].numPixel;
            grOS[i].meanFG=grOS[i].meanFG/grOS[i].numPixel;
        }
        if(grOS[i].iD==0){
            meanSum[0]+=grOS[i].meanBG;
            numB++;
        }else if(grOS[i].iD==1){
            meanSum[1]+=grOS[i].meanFG;
            numF++;
        }
    }
    meanSum[0]=meanSum[0]/numB;
    meanSum[1]=meanSum[1]/numF;
}

void ImageGraph::solveGCut(){
    //0.5 for ice skater
    //2 for erhan
    float e2c = 2; //Make it 1 finally :)
    float cp=(2.0/2.0);
    float dist;
    delete gr;
    gr = new FloatGraph(numlabels,numlabels*10);
    for(int i=0;i<numlabels;i++)
        gr->add_node();

    float meanE=0;
    int numE=0;
    for(int i=0;i<numlabels;i++)
        for(int k=0;k<grOS[i].numSpatN;k++)
        {
            dist =  (grOS[i].meanColor.B - grOS[grOS[i].idSpatN[k]].meanColor.B)*(grOS[i].meanColor.B - grOS[grOS[i].idSpatN[k]].meanColor.B);
            dist += (grOS[i].meanColor.G - grOS[grOS[i].idSpatN[k]].meanColor.G)*(grOS[i].meanColor.G - grOS[grOS[i].idSpatN[k]].meanColor.G);
            dist += (grOS[i].meanColor.R - grOS[grOS[i].idSpatN[k]].meanColor.R)*(grOS[i].meanColor.R - grOS[grOS[i].idSpatN[k]].meanColor.R);

            numE++;
            meanE=(meanE*(numE-1)+dist)/numE;
        }

    float edCol;
    float edFD;
    float tempFD;
    for(int i=0;i<numlabels;i++)
    {
        if(grOS[i].sID==1)
            gr->add_tweights(i,0,100);
        else if(grOS[i].sID==0)
            gr->add_tweights(i,100,0);
        else
        {
            if(grOS[i].iD==1)
                gr->add_tweights(i,(1/meanSum[0])*grOS[i].meanBG,(1/meanSum[1])*grOS[i].meanFG*2);
            else if(grOS[i].iD==0)
                gr->add_tweights(i,(1/meanSum[0])*grOS[i].meanBG*2,(1/meanSum[1])*grOS[i].meanFG);
            else
                gr->add_tweights(i,(1/meanSum[0])*grOS[i].meanBG,(1/meanSum[1])*grOS[i].meanFG);
        }
    //gr->add_tweights(i,(1/meanSum[0])*grOS[i].meanBG,(1/meanSum[1])*grOS[i].meanFG);
        


         for(int k=0;k<grOS[i].numSpatN;k++)
            if(grOS[i].idSpatN[k]<i)
            {
                dist = (grOS[i].meanColor.B - grOS[grOS[i].idSpatN[k]].meanColor.B)*(grOS[i].meanColor.B - grOS[grOS[i].idSpatN[k]].meanColor.B);
                dist += (grOS[i].meanColor.G - grOS[grOS[i].idSpatN[k]].meanColor.G)*(grOS[i].meanColor.G- grOS[grOS[i].idSpatN[k]].meanColor.G);
                dist += (grOS[i].meanColor.R - grOS[grOS[i].idSpatN[k]].meanColor.R)*(grOS[i].meanColor.B - grOS[grOS[i].idSpatN[k]].meanColor.R);

                edCol = exp((dist*cp*(-1))/meanE);

                gr->add_edge(i,grOS[i].idSpatN[k],e2c*(edCol),e2c*(edCol));
            }
    }
    sumNodeUp();
    gr->maxflow();
    for(int i=0;i<numlabels;i++)
        if(gr->what_segment(i)==FloatGraph::SOURCE)
            grOS[i].sID=0;
        else
            grOS[i].sID=1;

}

void ImageGraph::setGroundTruth(char *fileName){
    Mat dummyMat;
    Mat GTImage;
    for(int i=0;i<numlabels;i++)
        grOS[i].iD=0;

    dummyMat = imread(fileName,0);
    resize(dummyMat,GTImage,Size(width,height));
    for(int x=0;x<width;x++)
        for(int y=0;y<height;y++)
            if(GTImage.at<uchar>(y,x)<100)
                grOS[segmentID[y*width+x]].iD--;
            else
                grOS[segmentID[y*width+x]].iD++;

    for(int i=0;i<numlabels;i++)
        if(grOS[i].iD>(grOS[i].numPixel*0.2))
        {
            grOS[i].iD=1;
            grOS[i].sID=-1;
        }
        else if(grOS[i].iD<((-0.2)*grOS[i].numPixel))
        {
            grOS[i].iD=0;
            grOS[i].sID=-1;
        }
        else
        {
            grOS[i].iD=-1;
            grOS[i].sID=-1;
        }
}

void ImageGraph::getSPBoundary(cv::Mat& canvIm){
    for(int xS=1;xS<width-1;xS++)
        for(int yS=1;yS<height-1;yS++)
        {
            if((segmentID[(yS)*width+(xS)]!=segmentID[(yS+1)*width+(xS)])||(segmentID[(yS)*width+(xS)]!=segmentID[(yS)*width+(xS+1)]))
            {
                canvIm.at<cv::Vec3b>(yS,xS)[0]=0;
                canvIm.at<cv::Vec3b>(yS,xS)[1]=0;
                canvIm.at<cv::Vec3b>(yS,xS)[2]=255;
            }
        }
}

void ImageGraph::getResult(cv::Mat& canvIm){
    for(int xS=1;xS<width;xS++)
        for(int yS=1;yS<height;yS++)
        {
            if(grOS[segmentID[yS*width+xS]].sID==0)
                canvIm.at<cv::Vec3b>(yS,xS)[0]=im.at<cv::Vec3b>(yS,xS)[0]*0.4 + (0.6*255);
            else if(grOS[segmentID[yS*width+xS]].sID==1)
                canvIm.at<cv::Vec3b>(yS,xS)[1]=im.at<cv::Vec3b>(yS,xS)[1]*0.4 + (0.6*255);
            else
                canvIm.at<cv::Vec3b>(yS,xS)[2]=im.at<cv::Vec3b>(yS,xS)[2]*0.4 + (0.6*255);
        }
}

void ImageGraph::writeOutGraph(char* fName,bool isComputed){
    ofstream of;
    of.open(fName);
    if(isComputed)
    {
        for(int yS=1;yS<height;yS++)
        {
            for(int xS=1;xS<width;xS++)
            {
                of<<gr->get_trcap(segmentID[yS*width+xS])<<" ";
            }
            of<<endl;
        }
    }else{
        for(int y=0;y<height;y++)
        {
            for(int x=0;x<width;x++)
                of<<outGraph[y*width+x]<<" ";
            of<<endl;
        }
    }
    of.flush();
    of.close();
    
    

}

void ImageGraph::writeOutOS(char* fName){
    ofstream of;
    of.open(fName);
    for(int yS=1;yS<height;yS++)
    {
        for(int xS=1;xS<width;xS++)
        {
            of<<segmentID[yS*width+xS]<<" ";
        }
        of<<endl;
    }
    of.close();
}

void ImageGraph::getPixelWiseResidual(){
    for(int x=0;x<width;x++)
        for(int y=0;y<height;y++)
            outGraph[y*width+x]=gr->get_trcap(segmentID[y*width+x]);
}

void ImageGraph::getPixelResidual(){
    for(int x=0;x<width;x++)
        for(int y=0;y<height;y++)
            outGraph[y*width+x]=grOS[segmentID[y*width+x]].residual;
}


float perm(CLR c1,CLR c2,float sigma){

    return exp((-1)*sqrt( (c1.R-c2.R)*(c1.R-c2.R) + (c1.G-c2.G)*(c1.G-c2.G) + (c1.B-c2.B)*(c1.B-c2.B))*(1.0/sigma));
}

float perm(cv::Vec3b c1,cv::Vec3b c2,float sigma){
    return exp((-1)*sqrt( (c1[0]-c2[0])*(c1[0]-c2[0]) + (c1[1]-c2[1])*(c1[1]-c2[1]) + (c1[2]-c2[2])*(c1[2]-c2[2]))*(1.0/sigma));
}

void ImageGraph::biFilt(float* inp,float* out,float sigma=15){
    float* hLineR = new float[width];
    float* hLineL = new float[width];

    float* vLineU = new float[height];
    float* vLineB = new float[height];    
    
    float p;
    //Horizontal Pass
    for(int y=0;y<height;y++)
    {
        for(int x=0;x<width;x++){
            hLineL[x]=inp[y*width+x];
            hLineR[x]=inp[y*width+x];
        }
        for(int x=1;x<width;x++)
        {
            p = perm(im.at<cv::Vec3b>(y,x),im.at<cv::Vec3b>(y,x-1),sigma);
            hLineL[x]+=(hLineL[x-1]*p);
            p = perm(im.at<cv::Vec3b>(y,width-x-1),im.at<cv::Vec3b>(y,width-x),sigma);
            hLineR[width-x-1]+=(hLineR[width-x]*p);
        }
        for(int x=0;x<width;x++)
            out[y*width+x]=(hLineR[x]+hLineL[x])/2.0;
    }

    //Vertical Pass
    for(int x=0;x<width;x++)
    {
        for(int y=0;y<height;y++){
            vLineU[y]=out[y*width+x];
            vLineB[y]=out[y*width+x];
        }
        for(int y=1;y<height;y++)
        {
            vLineB[y]+=vLineB[y-1]*perm(im.at<cv::Vec3b>(y,x),im.at<cv::Vec3b>(y-1,x),sigma);
            vLineU[height-y-1]+=vLineU[height-y]*perm(im.at<cv::Vec3b>(height-y-1,x),im.at<cv::Vec3b>(height-y,x),sigma);
        }
        for(int y=0;y<height;y++)
        {
            out[y*width+x]=(vLineU[y]+vLineB[y])/2.0;
        }
    }
    delete [] hLineR;
    delete [] hLineL;

    delete [] vLineU;
    delete [] vLineB;
}

void ImageGraph::smooth(){
    
    float * outGraphTemp = 0;
    outGraphTemp = new float[width*height];
    float * ones = 0;
    ones = new float[width*height];
    for(int i=0;i<width*height;i++)
        ones[i]=1;    

    biFilt(outGraph,outGraphTemp,5);
    biFilt(ones,outGraph,5);

    for(int i=0;i<width*height;i++)
        outGraph[i]=outGraphTemp[i]/outGraph[i];     
    
    
    for(int i=0;i<numlabels;i++)
        grOS[i].residual=0; 
    for(int xS=0;xS<width;xS++)
        for(int yS=0;yS<height;yS++)
                grOS[segmentID[yS*width+xS]].residual+=outGraph[yS*width+xS];
    for(int i=0;i<numlabels;i++){
        grOS[i].residual=grOS[i].residual/grOS[i].numPixel;
    }

    delete [] ones;
    delete [] outGraphTemp;
}   

void ImageGraph::getFromPrevFrame(ImageGraph* prev){
    set<int> alreadyChecked;
    float normA;
    float ress;
    for(int i=0;i<numlabels;i++)
    {
            normA=0;
            ress=0;
            alreadyChecked.clear();
            for(int x=grOS[i].bottomRight.X;x<=grOS[i].topLeft.X;x++)
                for(int y=grOS[i].bottomRight.Y;y<=grOS[i].topLeft.Y;y++)
                {
                    if(alreadyChecked.count(prev->segmentID[y*width+x])>0)
                        continue;
                    alreadyChecked.insert(prev->segmentID[y*width+x]);
                    normA+=perm(prev->grOS[prev->segmentID[y*width+x]].meanColor,grOS[i].meanColor,15);
                    ress+=perm(prev->grOS[prev->segmentID[y*width+x]].meanColor,grOS[i].meanColor,15)*prev->grOS[prev->segmentID[y*width+x]].residual;
                }
            grOS[i].residual=ress/normA;
    }
    meanRes = 0;
    for(int i=0;i<numlabels;i++)
        meanRes+=abs(grOS[i].residual);
    meanRes/=numlabels;
  //  cout<<"Prev Mult:"<<meanRes<<" "<<prev->meanRes<<endl;
    float multC = 2/meanRes;//prev->meanRes/meanRes;
    for(int i=0;i<numlabels;i++)
        grOS[i].residual*=multC;
    
}

void ImageGraph::sumNodeUp(){
    meanRes = 0;
    for(int i=0;i<numlabels;i++)
        meanRes+=abs(gr->get_trcap(i));
    meanRes/=numlabels;
    //cout<<meanRes<<" "<<numlabels<<endl;
}

void ImageGraph::solveGCutNoGMM(){
    float e2c = 0.0001 ; //Make it 1 finally :)
    float cp=(2.0/2.0);
    float dist;
    delete gr;
    gr = new FloatGraph(numlabels,numlabels*10);
    for(int i=0;i<numlabels;i++)
        gr->add_node();

    float meanE=0;
    int numE=0;
    for(int i=0;i<numlabels;i++)
        for(int k=0;k<grOS[i].numSpatN;k++)
        {
            dist =  (grOS[i].meanColor.B - grOS[grOS[i].idSpatN[k]].meanColor.B)*(grOS[i].meanColor.B - grOS[grOS[i].idSpatN[k]].meanColor.B);
            dist += (grOS[i].meanColor.G - grOS[grOS[i].idSpatN[k]].meanColor.G)*(grOS[i].meanColor.G - grOS[grOS[i].idSpatN[k]].meanColor.G);
            dist += (grOS[i].meanColor.R - grOS[grOS[i].idSpatN[k]].meanColor.R)*(grOS[i].meanColor.R - grOS[grOS[i].idSpatN[k]].meanColor.R);

            numE++;
            meanE=(meanE*(numE-1)+dist)/numE;
        }

    float edCol;
    float edFD;
    float tempFD;
    for(int i=0;i<numlabels;i++)
    {
        if(grOS[i].residual<0)//FG
            gr->add_tweights(i,0,(-1)*grOS[i].residual);
        else
            gr->add_tweights(i,grOS[i].residual,0);        


         for(int k=0;k<grOS[i].numSpatN;k++)
            if(grOS[i].idSpatN[k]<i)
            {
                dist = (grOS[i].meanColor.B - grOS[grOS[i].idSpatN[k]].meanColor.B)*(grOS[i].meanColor.B - grOS[grOS[i].idSpatN[k]].meanColor.B);
                dist += (grOS[i].meanColor.G - grOS[grOS[i].idSpatN[k]].meanColor.G)*(grOS[i].meanColor.G- grOS[grOS[i].idSpatN[k]].meanColor.G);
                dist += (grOS[i].meanColor.R - grOS[grOS[i].idSpatN[k]].meanColor.R)*(grOS[i].meanColor.B - grOS[grOS[i].idSpatN[k]].meanColor.R);

                edCol = exp((dist*cp*(-1))/meanE);

                gr->add_edge(i,grOS[i].idSpatN[k],e2c*(edCol),e2c*(edCol));
            }
    }
    sumNodeUp();
    gr->maxflow();
    for(int i=0;i<numlabels;i++)
        if(gr->what_segment(i)==FloatGraph::SOURCE)
            grOS[i].sID=0;
        else
            grOS[i].sID=1;

}


void ImageGraph::smoothN(){
    
    float * outGraphTemp = 0;
    outGraphTemp = new float[width*height];
    float * ones = 0;
    ones = new float[width*height];
    for(int i=0;i<width*height;i++)
        ones[i]=1;    

    biFilt(outGraph,outGraphTemp,5);
    biFilt(ones,outGraph,5);

    for(int i=0;i<width*height;i++)
        outGraph[i]=outGraphTemp[i]/outGraph[i];     
    
    
    for(int i=0;i<numlabels;i++)
        grOS[i].residual=0; 
    for(int xS=0;xS<width;xS++)
        for(int yS=0;yS<height;yS++)
                grOS[segmentID[yS*width+xS]].residual+=outGraph[yS*width+xS];
    for(int i=0;i<numlabels;i++){
        grOS[i].residual=grOS[i].residual/grOS[i].numPixel;
    }


    meanRes = 0;
    for(int i=0;i<numlabels;i++)
        meanRes+=abs(grOS[i].residual);
    meanRes/=numlabels;
    float multC = 2/meanRes;//prev->meanRes/meanRes;
    for(int i=0;i<numlabels;i++)
        grOS[i].residual*=multC;

    delete [] ones;
    delete [] outGraphTemp;
}   

void ImageGraph::learnGMM2(int iter){
    fgGMM.clear();
    bgGMM.clear();

    float* inVecBG = new float[width*height*3];
    float* inVecFG = new float[width*height*3];

    int numBG=0;
    int numFG=0;

    for(int x=0;x<width;x=x+3)
        for(int y=0;y<height;y=y+3)
        {
            if(grOS[segmentID[y*width+x]].sID==0)
            {
                inVecBG[numBG*3+0]=im.at<cv::Vec3b>(y,x)[0];
                inVecBG[numBG*3+1]=im.at<cv::Vec3b>(y,x)[1];
                inVecBG[numBG*3+2]=im.at<cv::Vec3b>(y,x)[2];
                numBG++;
            }else if(grOS[segmentID[y*width+x]].sID==1)
            {
                inVecFG[numFG*3+0]=im.at<cv::Vec3b>(y,x)[0];
                inVecFG[numFG*3+1]=im.at<cv::Vec3b>(y,x)[1];
                inVecFG[numFG*3+2]=im.at<cv::Vec3b>(y,x)[2];
                numFG++;
            }
        }


    fgGMM.insertData(inVecFG,numFG);
    bgGMM.insertData(inVecBG,numBG);
    fgGMM.iterateGMM(iter);
    bgGMM.iterateGMM(iter);
    delete [] inVecBG;
    delete [] inVecFG;
}
