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

// int main(int argc, char **argv)
// {

//     std::vector<std::string> labels = {
//         "person"};
//     unsigned char colors[][3] = {
//         {255, 0, 0}};

//     cv::Mat bgr = cv::imread("/home/ubuntu/yolov7/inference/images/bus.jpg");
//     ; // 预处理和源码不太一样，所以影响了后面的

//     int target_size = 640;

//     cv::Mat resize_img;
//     cv::resize(bgr, resize_img, cv::Size(target_size, target_size));
//     float cls_threshold = 0.25;

//     // MNN inference
//     auto mnnNet = std::shared_ptr<MNN::Interpreter>(
//         MNN::Interpreter::createFromFile("/home/ubuntu/yolov7/runs/train/exp3/weights/best.mnn"));
//     auto t1 = std::chrono::steady_clock::now();
//     MNN::ScheduleConfig netConfig;
//     netConfig.type = MNN_FORWARD_CPU;
//     netConfig.numThread = 4;

//     auto session = mnnNet->createSession(netConfig);
//     auto input = mnnNet->getSessionInput(session, nullptr);

//     mnnNet->resizeTensor(input, {1, 3, (int)target_size, (int)target_size});
//     mnnNet->resizeSession(session);
//     MNN::CV::ImageProcess::Config config;

//     const float mean_vals[3] = {0, 0, 0};

//     const float norm_255[3] = {1.f / 255, 1.f / 255.f, 1.f / 255};

//     std::shared_ptr<MNN::CV::ImageProcess> pretreat(
//         MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, mean_vals, 3,
//                                       norm_255, 3));

//     pretreat->convert(resize_img.data, (int)target_size, (int)target_size, resize_img.step[0], input);

//     mnnNet->runSession(session);

//     auto output = mnnNet->getSessionOutput(session, yolov7_layers[2].name.c_str());

//     MNN::Tensor outputHost(output, output->getDimensionType());
//     output->copyToHostTensor(&outputHost);

//     // 毫秒级
//     std::vector<float> vec_scores;
//     std::vector<float> vec_new_scores;
//     std::vector<int> vec_labels;
//     int outputHost_shape_c = outputHost.channel();
//     int outputHost_shape_d = outputHost.dimensions();
//     int outputHost_shape_w = outputHost.width();
//     int outputHost_shape_h = outputHost.height();

//     printf("shape_d=%d shape_c=%d shape_h=%d shape_w=%d outputHost.elementSize()=%d\n", outputHost_shape_d,
//            outputHost_shape_c, outputHost_shape_h, outputHost_shape_w, outputHost.elementSize());
//     auto yolov7_534 = mnnNet->getSessionOutput(session, yolov7_layers[1].name.c_str());

//     MNN::Tensor output_534_Host(yolov7_534, yolov7_534->getDimensionType());
//     yolov7_534->copyToHostTensor(&output_534_Host);

//     outputHost_shape_c = output_534_Host.channel();
//     outputHost_shape_d = output_534_Host.dimensions();
//     outputHost_shape_w = output_534_Host.width();
//     outputHost_shape_h = output_534_Host.height();
//     printf("shape_d=%d shape_c=%d shape_h=%d shape_w=%d output_534_Host.elementSize()=%d\n", outputHost_shape_d,
//            outputHost_shape_c, outputHost_shape_h, outputHost_shape_w, output_534_Host.elementSize());

//     auto yolov7_554 = mnnNet->getSessionOutput(session, yolov7_layers[0].name.c_str());

//     MNN::Tensor output_544_Host(yolov7_554, yolov7_554->getDimensionType());
//     yolov7_554->copyToHostTensor(&output_544_Host);

//     outputHost_shape_c = output_544_Host.channel();
//     outputHost_shape_d = output_544_Host.dimensions();
//     outputHost_shape_w = output_544_Host.width();
//     outputHost_shape_h = output_544_Host.height();
//     printf("shape_d=%d shape_c=%d shape_h=%d shape_w=%d output_544_Host.elementSize()=%d\n", outputHost_shape_d,
//            outputHost_shape_c, outputHost_shape_h, outputHost_shape_w, output_544_Host.elementSize());

//     std::vector<YoloLayerData> &layers = yolov7_layers;

//     std::vector<BoxInfo> result;
//     std::vector<BoxInfo> boxes;
//     float threshold = 0.3;
//     float nms_threshold = 0.7;

//     boxes = decode_infer(outputHost, layers[2].stride, target_size, labels.size(), layers[2].anchors, threshold);
//     result.insert(result.begin(), boxes.begin(), boxes.end());

//     boxes = decode_infer(output_534_Host, layers[1].stride, target_size, labels.size(), layers[1].anchors, threshold);
//     result.insert(result.begin(), boxes.begin(), boxes.end());

//     boxes = decode_infer(output_544_Host, layers[0].stride, target_size, labels.size(), layers[0].anchors, threshold);
//     result.insert(result.begin(), boxes.begin(), boxes.end());

//     nms(result, nms_threshold);
//     scale_coords(result, target_size, target_size, bgr.cols, bgr.rows);
//     cv::Mat frame_show = draw_box(bgr, result, labels, colors);
//     cv::imshow("out", bgr);
//     cv::imwrite("dp.jpg", bgr);
//     cv::waitKey(0);
//     mnnNet->releaseModel();
//     mnnNet->releaseSession(session);
//     return 0;
// }

namespace jetflow
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