#include "yolov7.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace jetflow;

int main()
{
    std::cout << "this is a yolov7 example with mnn!" << std::endl;
    yolov7 yolov7det;
    yolov7det.init("/data/code/MNN/build/jfcodedet.mnn");

    Mat image = imread("/data/datasets/jietouhe/barcode/images/VID_20221208_173318_00001.jpg");
    
    vector<BoxInfo> objects;
    yolov7det.detect(image, objects);
    yolov7det.draw(image, objects);

    imwrite("./test.jpg", image);

    return 0;
}