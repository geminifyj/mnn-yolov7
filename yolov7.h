#ifndef _YOLOV7_H_
#define _YOLOV7_H_

#include <iostream>
#include <algorithm>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <MNN/Interpreter.hpp>
#include <MNN/ImageProcess.hpp>

using namespace std;
using namespace cv;

namespace geminifyj
{

    typedef struct
    {
        int width;
        int height;
    } YoloSize;

    typedef struct
    {
        std::string name;
        int stride;
        std::vector<YoloSize> anchors;
    } YoloLayerData;

    class BoxInfo
    {
    public:
        int x1, y1, x2, y2, label, id;
        float score;
    };

    class yolov7
    {
    private:
        /* data */
        vector<string> labels = {
            "barcode", "qrcode"};

        unsigned char colors[19][3] = {
            {54, 67, 244},
            {99, 30, 233},
            {176, 39, 156},
            {183, 58, 103},
            {181, 81, 63},
            {243, 150, 33},
            {244, 169, 3},
            {212, 188, 0},
            {136, 150, 0},
            {80, 175, 76},
            {74, 195, 139},
            {57, 220, 205},
            {59, 235, 255},
            {7, 193, 255},
            {0, 152, 255},
            {34, 87, 255},
            {72, 85, 121},
            {158, 158, 158},
            {139, 125, 96}};

        shared_ptr<MNN::Interpreter> Inter;

        std::vector<YoloLayerData> yolov7_layers{
            {"1206", 32, {{142, 110}, {192, 243}, {459, 401}}},
            {"1221", 16, {{36, 75}, {76, 55}, {72, 146}}},
            {"1236", 8, {{12, 16}, {19, 36}, {40, 28}}},
        };

        int net_size = 416;

        float threshold;
        float nmsth;

        float mean_vals[3] = {0, 0, 0};

        float norm_255[3] = {1.f / 255, 1.f / 255.f, 1.f / 255};

        MNN::Session *session;
        MNN::Tensor *input;

    public:
        yolov7();
        ~yolov7();

        int init(string model, float threshold = 0.45, float mnsth = 0.45);

        int detect(const Mat &frame, vector<BoxInfo> &objects);

        void draw(Mat &frame, const vector<BoxInfo> &objects);

    private:
        double iou(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt);
        void nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH);
        void scale_coords(std::vector<BoxInfo> &boxes, int padd_w, int padd_h, float r, int width, int height);

        std::vector<BoxInfo> decode_infer(MNN::Tensor &data, int stride, int net_size, int num_classes,
                                          const std::vector<YoloSize> &anchors, float threshold);

        Mat letterbox(const Mat &frame, int &padd_w, int &padd_h, float &r);

        float sigmoid(float x)
        {
            return static_cast<float>(1.f / (1.f + exp(-x)));
        }
    };
}

#endif