//
//  main.cpp
//  testCV
//
//  Created by Antonius Harijanto on 1/22/13.
//  Copyright (c) http://www.juergenwiki.de/work/wiki/doku.php?id=public%3ahog_descriptor_computation_and_visualization
//

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>

using namespace std;
using namespace cv;


vector<float> visualizeHOG(Mat img_raw);
Mat get_hogdescriptor_visu(Mat& origImg, vector<float>& descriptorValues);

std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    return split(s, delim, elems);
}

std::vector<float> split(std::vector<string> &src) {
    std::vector<float> dst;
    for(std::vector<string>::iterator it = src.begin(); it < src.end(); it++){
        dst.push_back(atof(it->c_str()));
    }
    return dst;
}

int main(int argc, const char * argv[])
{

    if(argc != 3 && argc != 5){
        //printf("This program source is taken from http://www.juergenwiki.de/work/wiki/doku.php?id=public%3ahog_descriptor_computation_and_visualization\n\n");
        printf("Visualize the HOG descriptor of an image");
        printf("Usage: ./%s [input image] [output image]\n", argv[0]);
        printf("Or\n");
        printf("Usage: ./%s -d [input image] [input descriptor]\n", argv[0]);
        
        exit(EXIT_FAILURE);
    }
    
    
    string option = argv[1];
    if(option == "-d"){
        string filename = argv[2];
        string input_descriptor = argv[3];
        string output_path = argv[4];
        
        string line;
        string whole_lines = "";
        
        ifstream myfile (input_descriptor.c_str());
        
        if (myfile.is_open())
        {
            while ( myfile.good() )
            {
                getline (myfile,line);
                whole_lines += line;
            }
            myfile.close();
        }
        
        vector<string> descriptors_str = split(whole_lines, ' ');
        vector<float> descriptors = split(descriptors_str);
        
        // insert code here...
        std::cout << "Hello, World!\n";
        
        Mat img_raw = imread(filename.c_str(), CV_LOAD_IMAGE_COLOR); // load as color image
        Mat img_gray = imread(filename.c_str(), CV_LOAD_IMAGE_GRAYSCALE); // load as color image
        Mat img_gray_scaled;
        
        resize(img_gray, img_gray_scaled, Size(64,128));
        
        // vector<float> descriptors = visualizeHOG(img_gray_scaled);
        printf("Descriptor size: %ld\n", descriptors.size());
        Mat result = get_hogdescriptor_visu(img_gray_scaled, descriptors);
        
        printf("Result is saved at %s\n", output_path.c_str());
        imwrite(output_path, result);
        
    }
    else{
        
        string input_path = argv[1];
        string output_path = argv[2];
        
        
        // insert code here...
        std::cout << "Hello, World!\n";
        
        string filename = input_path;
        Mat img_raw = imread(filename.c_str(), CV_LOAD_IMAGE_COLOR); // load as color image
        Mat img_gray = imread(filename.c_str(), CV_LOAD_IMAGE_GRAYSCALE); // load as color image
        Mat img_gray_scaled;
        
        resize(img_gray, img_gray_scaled, Size(64,128));
        
        vector<float> descriptors = visualizeHOG(img_gray_scaled);
        printf("Descriptor size: %ld\n", descriptors.size());
        Mat result = get_hogdescriptor_visu(img_gray_scaled, descriptors);
        
        printf("Result is saved at %s\n", output_path.c_str());
        imwrite(output_path, result);
    }
    
    return 0;
    
}

vector<float> visualizeHOG(Mat img){
    
    HOGDescriptor d(HOGDescriptor(Size(64,128), Size(16,16), Size(8,8), Size(8,8), 9));
    // Size(128,64), //winSize
    // Size(16,16), //blocksize
    // Size(8,8), //blockStride,
    // Size(8,8), //cellSize,
    // 9, //nbins,
    // 0, //derivAper,
    // -1, //winSigma,
    // 0, //histogramNormType,
    // 0.2, //L2HysThresh,
    // 0 //gammal correction,
    // //nlevels=64
    //);
    
    // void HOGDescriptor::compute(const Mat& img, vector<float>& descriptors,
    //                             Size winStride, Size padding,
    //                             const vector<Point>& locations) const
    

    
    vector<float> descriptorsValues;
    vector<Point> locations;
    d.compute( img, descriptorsValues, Size(0,0), Size(0,0), locations);
    
    
    cout << "HOG descriptor size is " << d.getDescriptorSize() << endl;
    cout << "img dimensions: " << img.cols << " width x " << img.rows << "height" << endl;
    cout << "Found " << descriptorsValues.size() << " descriptor values" << endl;
    cout << "Nr of locations specified : " << locations.size() << endl;
    
    
    return descriptorsValues;
}

Mat get_hogdescriptor_visu(Mat& origImg, vector<float>& descriptorValues)
{
    Mat color_origImg;
    cvtColor(origImg, color_origImg, CV_GRAY2RGB);
    
    float zoomFac = 3;
    Mat visu;
    resize(color_origImg, visu, Size(color_origImg.cols*zoomFac, color_origImg.rows*zoomFac));
    
    int blockSize       = 16;
    int cellSize        = 8;
    int gradientBinSize = 9;
    float radRangeForOneBin = M_PI/(float)gradientBinSize; // dividing 180° into 9 bins, how large (in rad) is one bin?
    
    // prepare data structure: 9 orientation / gradient strenghts for each cell
    int cells_in_x_dir = 64 / cellSize;
    int cells_in_y_dir = 128 / cellSize;
    int totalnrofcells = cells_in_x_dir * cells_in_y_dir;
    float*** gradientStrengths = new float**[cells_in_y_dir];
    int** cellUpdateCounter   = new int*[cells_in_y_dir];
    for (int y=0; y<cells_in_y_dir; y++)
    {
        gradientStrengths[y] = new float*[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x=0; x<cells_in_x_dir; x++)
        {
            gradientStrengths[y][x] = new float[gradientBinSize];
            cellUpdateCounter[y][x] = 0;
            
            for (int bin=0; bin<gradientBinSize; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }
    
    // nr of blocks = nr of cells - 1
    // since there is a new block on each cell (overlapping blocks!) but the last one
    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;
    
    // compute gradient strengths per cell
    int descriptorDataIdx = 0;
    int cellx = 0;
    int celly = 0;
    
    for (int blockx=0; blockx<blocks_in_x_dir; blockx++)
    {
        for (int blocky=0; blocky<blocks_in_y_dir; blocky++)
        {
            // 4 cells per block ...
            for (int cellNr=0; cellNr<4; cellNr++)
            {
                // compute corresponding cell nr
                int cellx = blockx;
                int celly = blocky;
                if (cellNr==1) celly++;
                if (cellNr==2) cellx++;
                if (cellNr==3)
                {
                    cellx++;
                    celly++;
                }
                
                for (int bin=0; bin<gradientBinSize; bin++)
                {
                    float gradientStrength = descriptorValues[ descriptorDataIdx ];
                    descriptorDataIdx++;
                    
                    gradientStrengths[celly][cellx][bin] += gradientStrength;
                    
                } // for (all bins)
                
                
                // note: overlapping blocks lead to multiple updates of this sum!
                // we therefore keep track how often a cell was updated,
                // to compute average gradient strengths
                cellUpdateCounter[celly][cellx]++;
                
            } // for (all cells)
            
            
        } // for (all block x pos)
    } // for (all block y pos)
    
    
    // compute average gradient strengths
    for (int celly=0; celly<cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {
            
            float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];
            
            // compute average gradient strenghts for each gradient bin direction
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }
    
    
    cout << "descriptorDataIdx = " << descriptorDataIdx << endl;
    
    // draw cells
    for (int celly=0; celly<cells_in_y_dir; celly++)
    {
        for (int cellx=0; cellx<cells_in_x_dir; cellx++)
        {
            int drawX = cellx * cellSize;
            int drawY = celly * cellSize;
            
            int mx = drawX + cellSize/2;
            int my = drawY + cellSize/2;
            
            rectangle(visu, Point(drawX*zoomFac,drawY*zoomFac), Point((drawX+cellSize)*zoomFac,(drawY+cellSize)*zoomFac), CV_RGB(100,100,100), 1);
            
            // draw in each cell all 9 gradient strengths
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                float currentGradStrength = gradientStrengths[celly][cellx][bin];
                
                // no line to draw?
                if (currentGradStrength==0)
                    continue;
                
                float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;
                
                float dirVecX = cos( currRad );
                float dirVecY = sin( currRad );
                float maxVecLen = cellSize/2;
                float scale = 2.5; // just a visualization scale, to see the lines better
                
                // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;
                
                // draw gradient visualization
                line(visu, Point(x1*zoomFac,y1*zoomFac), Point(x2*zoomFac,y2*zoomFac), CV_RGB(0,255,0), 1);
                
            } // for (all bins)
            
        } // for (cellx)
    } // for (celly)
    
    
    // don't forget to free memory allocated by helper data structures!
    for (int y=0; y<cells_in_y_dir; y++)
    {
        for (int x=0; x<cells_in_x_dir; x++)
        {
            delete[] gradientStrengths[y][x];            
        }
        delete[] gradientStrengths[y];
        delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;
    
    return visu;
    
} // get_hogdescriptor_visu

