#include "yolov7.h"

using namespace geminifyj;

yolov7::yolov7()
{
}

yolov7::~yolov7()
{
    if(session)
        Inter->releaseSession(session);
}

int yolov7::init(string model, float threshold, float mnsth)
{
    this->threshold = threshold;
    this->nmsth = mnsth;
    // MNN inference
    Inter = std::shared_ptr<MNN::Interpreter>(
        MNN::Interpreter::createFromFile(model.c_str()));
    MNN::ScheduleConfig netConfig;
    netConfig.type = MNN_FORWARD_CPU;
    netConfig.numThread = 4;

    session = Inter->createSession(netConfig);
    input = Inter->getSessionInput(session, nullptr);
    Inter->resizeTensor(input, {1, 3, (int)net_size, (int)net_size});
    Inter->resizeSession(session);

    return 0;
}

double yolov7::iou(cv::Rect_<float> bb_test, cv::Rect_<float> bb_gt)
{
    float in = (bb_test & bb_gt).area();
    float un = bb_test.area() + bb_gt.area() - in;

    if (un < DBL_EPSILON)
        return 0;

    return (double)(in / un);
}

void yolov7::nms(std::vector<BoxInfo> &input_boxes, float NMS_THRESH)
{
    std::sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b)
              { return a.score > b.score; });
    std::vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        for (int j = i + 1; j < int(input_boxes.size());)
        {
            float xx1 = std::max(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = std::max(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = std::min(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = std::min(input_boxes[i].y2, input_boxes[j].y2);
            float w = std::max(float(0), xx2 - xx1 + 1);
            float h = std::max(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);
            if (ovr >= NMS_THRESH)
            {
                input_boxes.erase(input_boxes.begin() + j);
                vArea.erase(vArea.begin() + j);
            }
            else
            {
                j++;
            }
        }
    }
}

void yolov7::scale_coords(std::vector<BoxInfo> &boxes, int padd_w, int padd_h, float r, int width, int height)
{
    std::cout << "padd_w: " << padd_w << " padd_h: " << padd_h << " scale: " << r << std::endl;
    for (auto &box : boxes)
    {
        // adjust offset to original unpadded
        float x0 = (box.x1 - padd_w) / r;
        float y0 = (box.y1 - padd_h) / r;
        float x1 = (box.x2 - padd_w) / r;
        float y1 = (box.y2 - padd_h) / r;

        // clip
        x0 = max(min(x0, (float) (width - 1)), 0.f);
        y0 = max(min(y0, (float) (height - 1)), 0.f);
        x1 = max(min(x1, (float) (width - 1)), 0.f);
        y1 = max(min(y1, (float) (height - 1)), 0.f);

        box.x1 = x0;
        box.x2 = x1;
        box.y1 = y0;
        box.y2 = y1;

    }

    return;
}

std::vector<BoxInfo> yolov7::decode_infer(MNN::Tensor &data, int stride, int net_size, int num_classes,
                                          const std::vector<YoloSize> &anchors, float threshold)
{
    std::vector<BoxInfo> result;
    int batchs, channels, height, width, pred_item;
    batchs = data.shape()[0];
    channels = data.shape()[1];
    height = data.shape()[2];
    width = data.shape()[3];
    pred_item = data.shape()[4];

    auto data_ptr = data.host<float>();
    for (int bi = 0; bi < batchs; bi++)
    {
        auto batch_ptr = data_ptr + bi * (channels * height * width * pred_item);
        for (int ci = 0; ci < channels; ci++)
        {
            auto channel_ptr = batch_ptr + ci * (height * width * pred_item);
            for (int hi = 0; hi < height; hi++)
            {
                auto height_ptr = channel_ptr + hi * (width * pred_item);
                for (int wi = 0; wi < width; wi++)
                {
                    auto width_ptr = height_ptr + wi * pred_item;
                    auto cls_ptr = width_ptr + 5;

                    auto confidence = sigmoid(width_ptr[4]);

                    for (int cls_id = 0; cls_id < num_classes; cls_id++)
                    {
                        float score = sigmoid(cls_ptr[cls_id]) * confidence;
                        if (score > threshold)
                        {
                            float cx = (sigmoid(width_ptr[0]) * 2.f - 0.5f + wi) * (float)stride;
                            float cy = (sigmoid(width_ptr[1]) * 2.f - 0.5f + hi) * (float)stride;
                            float w = pow(sigmoid(width_ptr[2]) * 2.f, 2) * anchors[ci].width;
                            float h = pow(sigmoid(width_ptr[3]) * 2.f, 2) * anchors[ci].height;

                            BoxInfo box;

                            box.x1 = std::max(0, std::min(net_size, int((cx - w / 2.f))));
                            box.y1 = std::max(0, std::min(net_size, int((cy - h / 2.f))));
                            box.x2 = std::max(0, std::min(net_size, int((cx + w / 2.f))));
                            box.y2 = std::max(0, std::min(net_size, int((cy + h / 2.f))));

                            box.score = score;
                            box.label = cls_id;
                            result.push_back(box);
                        }
                    }
                }
            }
        }
    }

    return result;
}

int yolov7::detect(const Mat &frame, vector<BoxInfo> &objects)
{
    assert(!frame.empty());
    assert(frame.rows > 0);
    assert(frame.cols > 0);

    int padd_w;
    int padd_h;
    float r;
    Mat resImage = letterbox(frame, padd_w, padd_h, r);
    // std::cout << "padd_w: " << padd_w << " padd_h: " << padd_h << " scale: " << r << std::endl;
    shared_ptr<MNN::CV::ImageProcess> pretreat(MNN::CV::ImageProcess::create(MNN::CV::BGR, MNN::CV::RGB, mean_vals, 3, norm_255, 3));
    pretreat->convert(resImage.data, net_size, net_size, resImage.step[0], input);

    Inter->runSession(session);

    auto output_32 = Inter->getSessionOutput(session, yolov7_layers[2].name.c_str());

    MNN::Tensor output_32_Host(output_32, output_32->getDimensionType());
    output_32->copyToHostTensor(&output_32_Host);

    // 毫秒级
    std::vector<float> vec_scores;
    std::vector<float> vec_new_scores;
    std::vector<int> vec_labels;
    int output_32_Host_shape_c = output_32_Host.channel();
    int output_32_Host_shape_d = output_32_Host.dimensions();
    int output_32_Host_shape_w = output_32_Host.width();
    int output_32_Host_shape_h = output_32_Host.height();

    printf("shape_d=%d shape_c=%d shape_h=%d shape_w=%d output_32_Host.elementSize()=%d\n", output_32_Host_shape_d,
           output_32_Host_shape_c, output_32_Host_shape_h, output_32_Host_shape_w, output_32_Host.elementSize());

    auto output_16 = Inter->getSessionOutput(session, yolov7_layers[1].name.c_str());

    MNN::Tensor output_16_Host(output_16, output_16->getDimensionType());
    output_16->copyToHostTensor(&output_16_Host);

    int output_16_Host_shape_c = output_16_Host.channel();
    int output_16_Host_shape_d = output_16_Host.dimensions();
    int output_16_Host_shape_w = output_16_Host.width();
    int output_16_Host_shape_h = output_16_Host.height();
    printf("shape_d=%d shape_c=%d shape_h=%d shape_w=%d output_16_Host.elementSize()=%d\n", output_16_Host_shape_d,
           output_16_Host_shape_c, output_16_Host_shape_h, output_16_Host_shape_w, output_16_Host.elementSize());

    auto output_8 = Inter->getSessionOutput(session, yolov7_layers[0].name.c_str());

    MNN::Tensor output_8_Host(output_8, output_8->getDimensionType());
    output_8->copyToHostTensor(&output_8_Host);

    int output_8_Host_shape_c = output_8_Host.channel();
    int output_8_Host_shape_d = output_8_Host.dimensions();
    int output_8_Host_shape_w = output_8_Host.width();
    int output_8_Host_shape_h = output_8_Host.height();
    printf("shape_d=%d shape_c=%d shape_h=%d shape_w=%d output_8_Host.elementSize()=%d\n", output_8_Host_shape_d,
           output_8_Host_shape_c, output_8_Host_shape_h, output_8_Host_shape_w, output_8_Host.elementSize());

    std::vector<YoloLayerData> &layers = yolov7_layers;

    objects.clear();
    std::vector<BoxInfo> boxes;

    boxes = decode_infer(output_32_Host, layers[2].stride, net_size, labels.size(), layers[2].anchors, threshold);
    objects.insert(objects.begin(), boxes.begin(), boxes.end());

    boxes = decode_infer(output_16_Host, layers[1].stride, net_size, labels.size(), layers[1].anchors, threshold);
    objects.insert(objects.begin(), boxes.begin(), boxes.end());

    boxes = decode_infer(output_8_Host, layers[0].stride, net_size, labels.size(), layers[0].anchors, threshold);
    objects.insert(objects.begin(), boxes.begin(), boxes.end());

    nms(objects, nmsth);
    scale_coords(objects, padd_w, padd_h, r, frame.cols, frame.rows);

    return 0;
}

void yolov7::draw(Mat &frame, const vector<BoxInfo> &objects)
{
    for (auto box : objects)
    {
        int width = box.x2 - box.x1;
        int height = box.y2 - box.y1;
        cv::Point p = cv::Point(box.x1, box.y1);
        cv::Rect rect = cv::Rect(box.x1, box.y1, width, height);
        cv::rectangle(frame, rect, cv::Scalar(colors[box.label][0], colors[box.label][1], colors[box.label][2]));
        string text = labels[box.label] + ":" + std::to_string(box.score);
        cv::putText(frame, text, p, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(colors[box.label][0], colors[box.label][1], colors[box.label][2]));
    }
    return;
}

Mat yolov7::letterbox(const Mat &src, int &padd_w, int &padd_h, float &r)
{
    // 以下为带边框图像生成
    int in_w = src.cols;
    int in_h = src.rows;
    int tar_w = net_size;
    int tar_h = net_size;
    // 哪个缩放比例小选用哪个
    r = min(float(tar_h) / in_h, float(tar_w) / in_w);
    int inside_w = round(in_w * r);
    int inside_h = round(in_h * r);
    padd_w = tar_w - inside_w;
    padd_h = tar_h - inside_h;
    // 内层图像resize
    Mat resize_img;
    resize(src, resize_img, Size(inside_w, inside_h));
    // cvtColor(resize_img, resize_img, COLOR_BGR2RGB);

    padd_w = padd_w / 2;
    padd_h = padd_h / 2;
    // 外层边框填充灰色
    int top = int(round(padd_h - 0.1));
    int bottom = int(round(padd_h + 0.1));
    int left = int(round(padd_w - 0.1));
    int right = int(round(padd_w + 0.1));
    copyMakeBorder(resize_img, resize_img, top, bottom, left, right, BORDER_CONSTANT, Scalar(114, 114, 114));
    return resize_img;
}