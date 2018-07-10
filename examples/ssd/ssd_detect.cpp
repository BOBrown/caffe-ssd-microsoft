// This is a demo code for using a SSD model to do detection.
// The code is modified from examples/cpp_classification/classification.cpp.
// Usage:
//    ssd_detect [FLAGS] model_file weights_file list_file
//
// where model_file is the .prototxt file defining the network architecture, and
// weights_file is the .caffemodel file containing the network parameters, and
// list_file contains a list of image files with the format as follows:
//    folder/img1.JPEG
//    folder/img2.JPEG
// list_file can also contain a list of video files with the format as follows:
//    folder/video1.mp4
//    folder/video2.mp4
//
#include <caffe/caffe.hpp>
#include <algorithm>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <iostream>
#include <cassert>
#include<math.h> 
#include<windows.h>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)

int HEIGHT = 380;

#define WIDTH 190
#define VECTOR_TXT 6 //读取的txt的格式长度
#define REFLECTION_WIDTH 380 //映射图像的宽度
#define REFLECTION_HEIGHT 800 //映射图像的高度

class Detector {
 public:
  Detector();

  std::vector<vector<float> > Detect(const cv::Mat& img);
  void init(const string& model_file, const string& weights_file, const string& mean_file, const string& mean_value, int gpu_id);
 private:
  void SetMean(const string& mean_file, const string& mean_value);

  void WrapInputLayer(std::vector<cv::Mat>* input_channels);

  void Preprocess(const cv::Mat& img,
                  std::vector<cv::Mat>* input_channels);

 private:
  shared_ptr<Net<float> > net_;
  cv::Size input_geometry_;
  int num_channels_;
  cv::Mat mean_;
};

Detector::Detector()
{}

void Detector::init(const string& model_file,
                   const string& weights_file,
                   const string& mean_file,
                   const string& mean_value,
				   int gpu_id) {


  Caffe::set_mode(Caffe::GPU);
  Caffe::SetDevice(gpu_id);
  
  /* Load the network. */
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(weights_file);

  CHECK_EQ(net_->num_inputs(), 1) << "Network should have exactly one input.";
  CHECK_EQ(net_->num_outputs(), 1) << "Network should have exactly one output.";

  Blob<float>* input_layer = net_->input_blobs()[0];
  num_channels_ = input_layer->channels();
  CHECK(num_channels_ == 3 || num_channels_ == 1)
    << "Input layer should have 1 or 3 channels.";
  input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

  /* Load the binaryproto mean file. */
  SetMean(mean_file, mean_value);
}

std::vector<vector<float> > Detector::Detect(const cv::Mat& img) {
  Blob<float>* input_layer = net_->input_blobs()[0];
  input_layer->Reshape(1, num_channels_,
                       input_geometry_.height, input_geometry_.width);
  /* Forward dimension change to all layers. */
  net_->Reshape();

  std::vector<cv::Mat> input_channels;
  WrapInputLayer(&input_channels);
  
  Preprocess(img, &input_channels);
  
  net_->Forward();

  /* Copy the output layer to a std::vector */
  Blob<float>* result_blob = net_->output_blobs()[0];
  const float* result = result_blob->cpu_data();
  const int num_det = result_blob->height();
  vector<vector<float> > detections;
  for (int k = 0; k < num_det; ++k) {
    if (result[0] == -1) {
      // Skip invalid detection.
      result += 7;
      continue;
    }
    vector<float> detection(result, result + 7);
	//need to judge the new detection result has no IOU with the existing proposals

	detections.push_back(detection);
	
    result += 7;
  }
  return detections;
}

/* Load the mean file in binaryproto format. */
void Detector::SetMean(const string& mean_file, const string& mean_value) {
  cv::Scalar channel_mean;
  if (!mean_file.empty()) {
    CHECK(mean_value.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    mean_blob.FromProto(blob_proto);
    CHECK_EQ(mean_blob.channels(), num_channels_)
      << "Number of channels of mean file doesn't match input layer.";

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    std::vector<cv::Mat> channels;
    float* data = mean_blob.mutable_cpu_data();
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
      channels.push_back(channel);
      data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */
    channel_mean = cv::mean(mean);
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
  }
  if (!mean_value.empty()) {
    CHECK(mean_file.empty()) <<
      "Cannot specify mean_file and mean_value at the same time";
    stringstream ss(mean_value);
    vector<float> values;
    string item;
    while (getline(ss, item, ',')) {
      float value = std::atof(item.c_str());
      values.push_back(value);
    }
    CHECK(values.size() == 1 || values.size() == num_channels_) <<
      "Specify either 1 mean_value or as many as channels: " << num_channels_;

    std::vector<cv::Mat> channels;
    for (int i = 0; i < num_channels_; ++i) {
      /* Extract an individual channel. */
      cv::Mat channel(input_geometry_.height, input_geometry_.width, CV_32FC1,
          cv::Scalar(values[i]));
      channels.push_back(channel);
    }
    cv::merge(channels, mean_);
  }
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Detector::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}

void Detector::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
  /* Convert the input image to the input image format of the network. */
  cv::Mat sample;
  if (img.channels() == 3 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
  else if (img.channels() == 4 && num_channels_ == 1)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
  else if (img.channels() == 4 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
  else if (img.channels() == 1 && num_channels_ == 3)
    cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
  else
    sample = img;

  cv::Mat sample_resized;
  if (sample.size() != input_geometry_)
    cv::resize(sample, sample_resized, input_geometry_);
  else
    sample_resized = sample;

  cv::Mat sample_float;
  if (num_channels_ == 3)
    sample_resized.convertTo(sample_float, CV_32FC3);
  else
    sample_resized.convertTo(sample_float, CV_32FC1);

  cv::Mat sample_normalized;
  cv::subtract(sample_float, mean_, sample_normalized);

  /* This operation will write the separate BGR planes directly to the
   * input layer of the network because it is wrapped by the cv::Mat
   * objects in input_channels. */
  cv::split(sample_normalized, *input_channels);

  CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
        == net_->input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";
}

DEFINE_string(mean_file, "",
    "The mean file used to subtract from the input image.");
DEFINE_string(mean_value, "28,28,28",
    "If specified, can be one value or can be same as image channels"
    " - would subtract from the corresponding channel). Separated by ','."
    "Either mean_file or mean_value should be provided, not both.");
DEFINE_string(file_type, "image",
    "The file type in the list_file. Currently support image and video.");
DEFINE_string(out_file, "",
    "If provided, store the detection results in the out_file.");
DEFINE_double(confidence_threshold, 0.9,
    "Only store detections with score higher than the threshold.");


extern "C"_declspec(dllexport) void global_init();
extern "C"_declspec(dllexport) void detect_init_front_refl(const char *network_pt, const char *caffemodel, int GPUID, int h, int max_in, int min_in, float thresh);
extern "C"_declspec(dllexport) void detect_init_front(const char *caffemodel, const char *network_pt, int GPUID, int h, int max_in, int min_in, float thresh);
extern "C"_declspec(dllexport) int detect_txt_front(const char *input_txt, bool rotate, const char *output_txt, bool vis);//
extern "C"_declspec(dllexport) void detect_init_front_refl(const char *network_pt, const char *caffemodel, int GPUID, int h, int max_in, int min_in, float thresh);
extern "C"_declspec(dllexport) void detect_init_back_refl(const char *network_pt, const char *caffemodel, int GPUID, int h, int max_in, int min_in, float thresh);
extern "C"_declspec(dllexport) void detect_init_back(const char *caffemodel, const char *network_pt, int GPUID, int h, int max_in, int min_in, float thresh);
extern "C"_declspec(dllexport) int detect_txt_back(const char *input_txt, bool rotate, const char *output_txt, bool vis);
extern "C"_declspec(dllexport) int detect_txt_front_refl(const char *input_img_txt, const char *input_result_txt, bool rotate, int &privacy_x, int &privacy_y, const char *output_txt);
extern "C"_declspec(dllexport) int detect_txt_back_refl(const char *input_img_txt, const char *input_result_txt, bool rotate, int &privacy_x, int &privacy_y, const char *output_txt);

Detector detector_front, detector_back, detector_front_refl, detector_back_refl;
float confidence_threshold_front, confidence_threshold_back, confidence_threshold_front_refl, confidence_threshold_back_relf;

int calcIOU(int one_x, int one_y, int one_w, int one_h, int two_x, int two_y, int two_w, int two_h)
{
	int calcIOU;
	if ((abs(one_x - two_x) < ((one_w + two_w) / 2.0)) && (abs(one_y - two_y) < ((one_h + two_h) / 2.0)))
	{
		int lu_x_inter = MAX((one_x - (one_w / 2.0)), (two_x - (two_w / 2.0)));
		int lu_y_inter = MIN((one_y + (one_h / 2.0)), (two_y + (two_h / 2.0)));

		int rd_x_inter = MIN((one_x + (one_w / 2.0)), (two_x + (two_w / 2.0)));
		int rd_y_inter = MAX((one_y - (one_h / 2.0)), (two_y - (two_h / 2.0)));

		int inter_w = abs(rd_x_inter - lu_x_inter);
		int inter_h = abs(lu_y_inter - rd_y_inter);

		int inter_square = inter_w * inter_h;
		int union_square = (one_w * one_h) + (two_w * two_h) - inter_square;

		calcIOU = inter_square / union_square * 1.0;
		printf("calcIOU:", calcIOU);
	}
	else
	{
		printf("No intersection!");
		calcIOU = 0;
	}
	return calcIOU;
}

int txt_to_image(const char *txt_file_name, cv::Mat image, bool rotate)
{
	FILE *f = NULL;
	f = fopen(txt_file_name, "rb");

	unsigned char *data_ptr = NULL;
	data_ptr = new unsigned char[WIDTH *HEIGHT];

	if (f == NULL)
	{
		printf("cannot open file %s ! \n", txt_file_name);
		return -1;
	}

	if (data_ptr == NULL)
	{
		printf("cannot get memory for data_ptr ! \n");
		return -2;
	}
	uchar *ip = image.data;
	fread(data_ptr, sizeof(unsigned char), HEIGHT *WIDTH, f);
	if (!rotate){
		for (int i = 0; i < HEIGHT*WIDTH; i++)
		{
			ip[3 * i] = data_ptr[i];
			ip[3 * i + 1] = data_ptr[i];
			ip[3 * i + 2] = data_ptr[i];
		}
	}
	else
	{
		for (int i = 0; i < HEIGHT*WIDTH; i++)
		{
			ip[3 * i] = data_ptr[HEIGHT*WIDTH - i + 2 * (i % WIDTH) - WIDTH];
			ip[3 * i + 1] = data_ptr[HEIGHT*WIDTH - i + 2 * (i % WIDTH) - WIDTH];
			ip[3 * i + 2] = data_ptr[HEIGHT*WIDTH - i + 2 * (i % WIDTH) - WIDTH];
		}
	}
	fclose(f);
	f = NULL;
	delete[] data_ptr;
	data_ptr = NULL;
	return 1;
}


/*you may change the class Detection to the type of SSD*/
int write_to_txt(const char *output_path, std::vector<vector<float> > detections, int pro_flag, float thre)
{
	std::streambuf* buf = std::cout.rdbuf();
	std::ostream out(buf);

	FILE *f = NULL;
	f = fopen(output_path, "w");
	if (f == NULL)
	{
		printf("cannot open or create file %s ! \n", output_path);
		return -1;
	}
	fprintf(f, "%d\n", pro_flag);
	for (int i = 0; i < detections.size(); i++)
	{
		const vector<float>& d = detections[i];
		// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
		CHECK_EQ(d.size(), 7);
		const float score = d[2];
		if (score >= thre)
		{
			float w = float(d[5] * WIDTH) - float(d[3] * WIDTH);
			float h = float(d[6] * HEIGHT) - float(d[4] * HEIGHT);
			out << static_cast<int>(d[1]) << " ";
			out << score << " ";
			out << static_cast<int>(d[3] * WIDTH) << " ";
			out << static_cast<int>(d[4] * HEIGHT) << " ";
			out << w << " ";
			out << h << std::endl;
			
			fprintf(f, "%f %f %f %f %f %d\n", float(d[3] * WIDTH), float(d[4] * HEIGHT), w, h, score, int(d[1]));
		}
		
	}
	fclose(f);
	f = NULL;
	return 1;
}

__declspec(dllexport) void global_init()
{

}



__declspec(dllexport) void detect_init_front(const char *network_pt, const char *caffemodel, int GPUID, int h, int max_in, int min_in, float thresh)
{
	detector_front.init(network_pt, caffemodel, FLAGS_mean_file, FLAGS_mean_value, GPUID);
	confidence_threshold_front = thresh;
	printf(" detector.init() for front -> done.\n");
}

__declspec(dllexport) void detect_init_front_refl(const char *network_pt, const char *caffemodel, int GPUID, int h, int max_in, int min_in, float thresh)
{
	detector_front_refl.init(network_pt, caffemodel, FLAGS_mean_file, FLAGS_mean_value, GPUID);
	confidence_threshold_front_refl = thresh;
	printf(" detector.init() for refletion front -> done.\n");
}

__declspec(dllexport) void detect_init_back(const char *network_pt, const char *caffemodel, int GPUID, int h, int max_in, int min_in, float thresh)
{
	detector_back.init(network_pt, caffemodel, FLAGS_mean_file, FLAGS_mean_value, GPUID);
	confidence_threshold_back = thresh;
	printf(" detector.init() for back -> done.\n");
}

__declspec(dllexport) void detect_init_back_refl(const char *network_pt, const char *caffemodel, int GPUID, int h, int max_in, int min_in, float thresh)
{
	detector_back_refl.init(network_pt, caffemodel, FLAGS_mean_file, FLAGS_mean_value, GPUID);
	confidence_threshold_front_refl = thresh;
	printf(" detector.init() for refletion back -> done.\n");
}

__declspec(dllexport) int detect_txt_front(const char *input_txt, bool rotate, const char *output_txt, bool vis)
{
	int pro_flag = 0;

	std::streambuf* buf = std::cout.rdbuf();
	std::ostream out(buf);
	cv::Mat img(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
	int returnvalue = 0;
	returnvalue = txt_to_image(input_txt, img, rotate);
	if (returnvalue < 0)
		return returnvalue;
	CHECK(!img.empty()) << "Unable to decode image " << input_txt;
	//cv::imshow("Result", img);
	//cv::waitKey(0);

	
	std::vector<vector<float> > detections = detector_front.Detect(img);


	for (int i = 0; i < detections.size(); ++i) {
		const vector<float>& d = detections[i];
		// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
		CHECK_EQ(d.size(), 7);
		const float score = d[2];
		if (score >= confidence_threshold_front) {
			pro_flag++;
		}
	}
	//changed by holobo
	write_to_txt(output_txt, detections, pro_flag, confidence_threshold_front);
	printf("-->detect done!\n");
	if (returnvalue < 0)
		return returnvalue;
	return 1;
}

//changed by holobo
__declspec(dllexport) int detect_txt_back(const char *input_txt, bool rotate, const char *output_txt, bool vis)
{
	int pro_flag = 0;

	std::streambuf* buf = std::cout.rdbuf();
	std::ostream out(buf);
	cv::Mat img(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
	int returnvalue = 0;
	returnvalue = txt_to_image(input_txt, img, rotate);
	if (returnvalue < 0)
		return returnvalue;
	CHECK(!img.empty()) << "Unable to decode image " << input_txt;
	//cv::imshow("Result", img);
	//cv::waitKey(0);
	std::vector<vector<float> > detections = detector_back.Detect(img);

	for (int i = 0; i < detections.size(); ++i) {
		const vector<float>& d = detections[i];
		// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
		CHECK_EQ(d.size(), 7);
		const float score = d[2];
		if (score >= confidence_threshold_back) {
			pro_flag++;
		}
	}
	//changed by holobo
	write_to_txt(output_txt, detections, pro_flag, confidence_threshold_back);
	printf("-->detect done!\n");
	if (returnvalue < 0)
		return returnvalue;
	return 1;
}

float solve_distance(vector<float> object1, vector<float> object2)//vector[0]存放x坐标，vector[1]存放y坐标
{
	float distance = sqrt(pow(object2[0] - object1[0], 2) + pow(object2[1] - object1[1], 2));
	return distance;
}

//changed by holobo
int reflection_from_vector(vector<float>& bb, std::vector<vector<float> > detection_result, const char *output_txt)
{
	std::vector<vector<float> > original_img_centers;
	std::vector<vector<float> > object_centers;

	//初始化中心点坐标,按顺序分别是人体的十个区域
	int reflection_center[20] = { 55, 119, 327, 119, 60, 198, 314, 204, 168, 281, 189, 445, 139, 555, 236, 555, 138, 703, 237, 703 };//数组中存放10个中心点坐标，这里写死在程序中，或采用读取txt的形式获得
	std::vector<vector<float> > reflection_centers;
	for (int i = 0; i < 10; ++i) {
		vector<float> ref_center;
		int center_x = reflection_center[0+2*i];
		int center_y = reflection_center[1+2*i];
		ref_center.push_back(center_x);
		ref_center.push_back(center_y);
		reflection_centers.push_back(ref_center);
		ref_center.clear();
	}
	//计算detection_result的中心点坐标
	for (int i = 0; i < detection_result.size(); ++i) {
		const vector<float>& d = detection_result[i];
		int w = int(d[5] * WIDTH) - int(d[3] * WIDTH);
		int h = int(d[6] * HEIGHT) - int(d[4] * HEIGHT);
		int center_x = int(d[3] * WIDTH) + int(w / 2);
		int center_y = int(d[4] * HEIGHT) + int(h / 2);
		vector<float> img_center;
		img_center.push_back(center_x);
		img_center.push_back(center_y);
		original_img_centers.push_back(img_center);
		img_center.clear();
	}
	//计算vector bb的中心点坐标，若存在多个object，计算对应数量的中心点
	for (int i = 0; i < int(bb.size() / VECTOR_TXT); ++i) {
		int w = bb[2 + VECTOR_TXT*i];
		int h = bb[3 + VECTOR_TXT*i];
		int center_x = int(bb[0 + VECTOR_TXT*i]) + int(w / 2);
		int center_y = int(bb[1 + VECTOR_TXT*i]) + int(h / 2);
		vector<float> object_center;
		object_center.push_back(center_x);
		object_center.push_back(center_y);
		object_centers.push_back(object_center);
		object_center.clear();
	}
	//object的中心点，在detection_result的中心点中找一个最近的，之后调用映射函数,把坐标写入txt
	std::streambuf* buf = std::cout.rdbuf();
	std::ostream out(buf);
	FILE *f = NULL;
	f = fopen(output_txt, "w");
	fprintf(f, "%d\n", object_centers.size());
	for (int i = 0; i < object_centers.size(); ++i) {
		const vector<float>& object_center = object_centers[i];
		float min_distance = 1000;
		int area_id = 0;
		for (int j = 0; j < original_img_centers.size(); ++j) {
			const vector<float>& img_center = original_img_centers[j];
			//最近的original_img区域
			float distance = solve_distance(object_center, img_center);//两个vector求欧式距离
			if (distance < min_distance)
			{
				area_id = j;
				min_distance = distance;
			}
		}
		//映射，给定object的在原始图片的全部信息和area_id，输出output_txt，即在映射图上的坐标信息
		//求出映射后bb的w h，左上角坐标
		//object in ori_img关于center的偏移
		int ori_minx = bb[0 + VECTOR_TXT * i];
		int ori_miny = bb[1 + VECTOR_TXT * i];
		float bias_x = (ori_minx - original_img_centers[area_id][0])/WIDTH;
		float bias_y = (ori_miny - original_img_centers[area_id][1])/HEIGHT;
		float ori_w = bb[2 + VECTOR_TXT * i] / WIDTH;
		float ori_h = bb[3 + VECTOR_TXT * i] / HEIGHT;
		
		//reflection的10个中心点坐标
		float ref_center_x = reflection_centers[area_id][0] / REFLECTION_WIDTH;
		float ref_center_y = reflection_centers[area_id][1] / REFLECTION_HEIGHT;
		float ref_x = (ref_center_x + bias_x) * REFLECTION_WIDTH;
		float ref_y = (ref_center_y + bias_y) * REFLECTION_HEIGHT;
		float ref_w = ori_w * REFLECTION_WIDTH;
		float ref_h = ori_h * REFLECTION_HEIGHT;

		//写入txt
		if (f == NULL)
		{
			printf("cannot open or create file %s ! \n", output_txt);
			return -1;
		}

		float num = float(bb[4 + VECTOR_TXT * i]);
		fprintf(f, "%d %d %d %d %f %d\n", int(ref_x), int(ref_y), int(ref_w), int(ref_h), num, int(bb[5 + VECTOR_TXT * i]));
	}
	fclose(f);
	return 1;
}

//changed by holobo
void stringTOnum1(string s, vector<float>& pdata)
{
	bool temp = false;        //读取一个数据标志位  
	float data = 0;             //分离的一个数据  
	float fraction = 0;             //保存fraction 
	int m = 0;                //数组索引值  
	bool floatflag = false;
	for (int i = 0; i<s.length(); i++)
	{
		while (((s[i] >= '0') && (s[i] <= '9')) || s[i] == '.')       //当前字符是数据，并一直读后面的数据，只要遇到不是数字为止  
		{
			if (s[i] == '.')//这个while中先判断是否是'.'
			{
				floatflag = true;
			}
			if (!floatflag && s[i] != '.')
			{
				temp = true;      //读数据标志位置位  
				data *= 10;
				data += (s[i] - '0');       //字符在系统以ASCII码存储，要得到其实际值必须减去‘0’的ASCII值  
			}
			if (floatflag && s[i] != '.')//浮点数操作
			{
				temp = true;      //读数据标志位置位  
				fraction *= 10;
				fraction += (s[i] - '0');
			}
			i++;
		}
		//刚读取了数据  
		if (temp)        //判断是否完全读取一个数据  
		{
			do
			{
				fraction = fraction / 10;
			} while (fraction > 1);
			pdata.push_back(data + fraction);      //赋值  
			m++;
			data = 0;
			fraction = 0;
			floatflag = false;
			temp = false;     //标志位复位  
		}
	}
}

//changed by holobo
//The return value is lower than 0:fail in txt2img
//1:success in reflection
//2:reflection results are lower than 10: That stands for the ineffciency of human part.
__declspec(dllexport) int detect_txt_front_refl(const char *input_img_txt, const char *input_result_txt, bool rotate, int &privacy_x, int &privacy_y, const char *output_txt)
{

	std::streambuf* buf = std::cout.rdbuf();
	std::ostream out(buf);
	cv::Mat img(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
	int returnvalue = 0;
	returnvalue = txt_to_image(input_img_txt, img, rotate);
	if (returnvalue < 0)
		return returnvalue;
	CHECK(!img.empty()) << "Unable to decode image " << input_img_txt;
	//cv::imshow("Result", img);
	//cv::waitKey(0);
	std::vector<vector<float> > detections = detector_front_refl.Detect(img);
	std::vector<vector<float> > detections_refl;
	std::vector<vector<float> > detections_refl_rectify;

	//write_to_txt(output_txt, detections, pro_flag, confidence_threshold_front_refl);

	for (int i = 0; i < detections.size(); ++i) {
		const vector<float>& d = detections[i];
		// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
		CHECK_EQ(d.size(), 7);
		const float score = d[2];
		if (score >= confidence_threshold_front_refl) {
			detections_refl.push_back(d);
		}
	}

	if (detections_refl.size() != 10) //当对原始图片检测时，身体部位的个数不等于10个，需要进行对称变化，或者是添加默认框，使size==10
	{
		std::streambuf* buf = std::cout.rdbuf();
		std::ostream out(buf);
		int flag = 0;
		FILE *f = NULL;
		f = fopen(output_txt, "w");
		if (f == NULL)
		{
			printf("cannot open or create file %s ! \n", output_txt);
			return -1;
		}
		fprintf(f, "%d\n", flag);  //如果没有检测到人体，就写一个内容为0的文件
		fclose(f);

		//return the coordinate of top body part which is helpful for protecting the privacy.
		//if detections_refl.size() != 10, then we can not confirm which index is the human body part. Therefore, we employ the search method to pick up the human body.
		int flag_privacy = 0;
		for (int i = 0; i < detections_refl.size(); ++i) {
			const vector<float>& d = detections_refl[i];
			if (d[1] == 3)//d[1] stores the id information.
			{
				int w = int(d[5] * WIDTH) - int(d[3] * WIDTH);
				privacy_x = int(d[3] * WIDTH) + int(w / 2);
				privacy_y = int(d[4] * HEIGHT);
				flag_privacy = 1;
				break;
			}
			if (((i + 1) == detections_refl.size()) && flag_privacy == 0)
			{
				privacy_x = 96;
				privacy_y = 96;
			}
		}
		return 2;
	}

	//矫正在detections中存储的元素，按照每一类的xmin从小到大排序
	for (int i = 0; i < 5; ++i) {
		const vector<float>& d = detections_refl[2 * i];
		const vector<float>& d_next = detections_refl[2 * i + 1];
		if (i == 2)
		{
			detections_refl_rectify.push_back(d);
			detections_refl_rectify.push_back(d_next);
			continue;
		}
		// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
		CHECK_EQ(d.size(), 7);
		CHECK_EQ(d_next.size(), 7);
		if (d_next[3] > d[3]) {
			//交换d和d_next
			detections_refl_rectify.push_back(d);
			detections_refl_rectify.push_back(d_next);
		}
		else{
			detections_refl_rectify.push_back(d_next);
			detections_refl_rectify.push_back(d);
		}
	}

	//read txt object file
	std::ifstream infile;
	infile.open(string(input_result_txt).data());
	assert(infile.is_open());
	string s;
	vector<float> pdata;
	while (getline(infile, s))
	{
		if (s.length() > 1)
		{
			stringTOnum1(s, pdata);
		}	
	}
	infile.close();
	//std::cout << "text:" << std::endl;
	//for (int i = 0; i < pdata.size(); i++)
		//std::cout << pdata[i] << " ";
	//return the coordinate of top body part which is helpful for protecting the privacy.
	const vector<float>& d_privacy = detections_refl_rectify[4];
	int w = int(d_privacy[5] * WIDTH) - int(d_privacy[3] * WIDTH);
	privacy_x = int(d_privacy[3] * WIDTH) + int(w / 2);
	privacy_y = int(d_privacy[4] * HEIGHT);

	int refl_result = reflection_from_vector(pdata, detections_refl_rectify, output_txt);
	//int iou = calcIOU(10, 20, 10, 10, 10, 20, 5, 5);
	
	printf("-->Reflection detect done!\n");
	return refl_result;
}

__declspec(dllexport) int detect_txt_back_refl(const char *input_img_txt, const char *input_result_txt, bool rotate, int &privacy_x, int &privacy_y, const char *output_txt)
{
	std::streambuf* buf = std::cout.rdbuf();
	std::ostream out(buf);
	cv::Mat img(HEIGHT, WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
	int returnvalue = 0;
	returnvalue = txt_to_image(input_img_txt, img, rotate);
	if (returnvalue < 0)
		return returnvalue;
	CHECK(!img.empty()) << "Unable to decode image " << input_img_txt;
	//cv::imshow("Result", img);
	//cv::waitKey(0);
	std::vector<vector<float> > detections = detector_back_refl.Detect(img);
	std::vector<vector<float> > detections_refl;
	std::vector<vector<float> > detections_refl_rectify;

	for (int i = 0; i < detections.size(); ++i) {
		const vector<float>& d = detections[i];
		// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
		CHECK_EQ(d.size(), 7);
		const float score = d[2];
		if (score >= confidence_threshold_front_refl) {
			detections_refl.push_back(d);
		}
	}

	if (detections_refl.size() != 10) //当对原始图片检测时，身体部位的个数不等于10个，需要进行对称变化，或者是添加默认框，使size==10
	{
		std::streambuf* buf = std::cout.rdbuf();
		std::ostream out(buf);
		int flag = 0;
		FILE *f = NULL;
		f = fopen(output_txt, "w");
		if (f == NULL)
		{
			printf("cannot open or create file %s ! \n", output_txt);
			return -1;
		}
		fprintf(f, "%d\n", flag);  //如果没有检测到人体，就写一个内容为0的文件
		fclose(f);

		//return the coordinate of top body part which is helpful for protecting the privacy.
		//if detections_refl.size() != 10, then we can not confirm which index is the human body part. Therefore, we employ the search method to pick up the human body.
		int flag_privacy = 0;
		for (int i = 0; i < detections_refl.size(); ++i) {
			const vector<float>& d = detections_refl[i];
			if (d[1] == 3)//d[1] stores the id information.
			{
				int w = int(d[5] * WIDTH) - int(d[3] * WIDTH);
				privacy_x = int(d[3] * WIDTH) + int(w / 2);
				privacy_y = int(d[4] * HEIGHT);
				flag_privacy = 1;
				break;
			}
			if (((i + 1) == detections_refl.size()) && flag_privacy == 0)
			{
				privacy_x = 96;
				privacy_y = 96;
			}
		}
		return 2;
	}

	//矫正在detections中存储的元素，按照每一类的xmin从小到大排序
	for (int i = 0; i < 5; ++i) {
		const vector<float>& d = detections_refl[2 * i];
		const vector<float>& d_next = detections_refl[2 * i + 1];
		if (i == 2)
		{
			detections_refl_rectify.push_back(d);
			detections_refl_rectify.push_back(d_next);
			continue;
		}
		// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
		CHECK_EQ(d.size(), 7);
		CHECK_EQ(d_next.size(), 7);
		if (d_next[3] > d[3]) {
			//交换d和d_next
			detections_refl_rectify.push_back(d);
			detections_refl_rectify.push_back(d_next);
		}
		else{
			detections_refl_rectify.push_back(d_next);
			detections_refl_rectify.push_back(d);
		}
	}

	//read txt object file
	std::ifstream infile;
	infile.open(string(input_result_txt).data());
	assert(infile.is_open());
	string s;
	vector<float> pdata;
	while (getline(infile, s))
	{
		if (s.length() > 1)
		{
			stringTOnum1(s, pdata);
		}
	}
	infile.close();
	//std::cout << "text:" << std::endl;
	//for (int i = 0; i < pdata.size(); i++)
	//std::cout << pdata[i] << " ";
	//return the coordinate of top body part which is helpful for protecting the privacy.
	const vector<float>& d_privacy = detections_refl_rectify[4];
	int w = int(d_privacy[5] * WIDTH) - int(d_privacy[3] * WIDTH);
	privacy_x = int(d_privacy[3] * WIDTH) + int(w / 2);
	privacy_y = int(d_privacy[4] * HEIGHT);

	int refl_result = reflection_from_vector(pdata, detections_refl_rectify, output_txt);
	//int iou = calcIOU(10, 20, 10, 10, 10, 20, 5, 5);

	printf("-->Reflection detect done!\n");
	return refl_result;
}

//detect_txt_front_refl是读取图像2进制文件，以及在原图的检测box文件，生成映射图的box文件。
//需要注意：1，是否翻转：需要将倒立的图片翻转成为正立的
//2，原图检测的box文件的格式应该是如下->否则读取数组的时候长度会越界：（int int int int float int）
//1
//129,274,12,26,0.95846,4
//3，映射的卡通图的大小是380*800，并且是man_f_new.png这张图，因为中心点已经写死在程序中。
//4，映射检测人体的时候需要将thre设置0.95甚至可以更高，因为若太低可能导致检测到的object数量大于10个，进入到返回值为2的情况。


int main() {


  const char * model_file = "D:\\CODE\\ssd_models\\models\\VGGNet\\VOC0712Plus\\SSD_300x300\\deploy.prototxt";
  const char * weights_file = "D:\\CODE\\ssd_models\\models\\VGGNet\\VOC0712Plus\\SSD_300x300\\VGG_VOC0712_SSD_300x300_iter_42000.caffemodel";

  const char * model_file_refl = "D:\\CODE\\ssd_models\\models\\VGGNet\\VOC0712Plus\\deploy_refl.prototxt";
  const char * weights_file_refl = "D:\\CODE\\ssd_models\\models\\VGGNet\\VOC0712Plus\\SSD_Reflection.caffemodel";
  
  detect_init_front(model_file, weights_file, 0, 380, 300, 300, 0.4);
  detect_init_back(model_file, weights_file, 0, 380, 300, 300, 0.4);

  long t1 = GetTickCount();
  detect_txt_front("D:/CODE/ssd_models/3image_finish1_b.txt", 1, "D:/CODE/ssd_models/result.txt", 0);
  long t2 = GetTickCount();
  std::cout << "forward time：" << (t2 - t1) << std::endl;
  detect_txt_back("D:/CODE/ssd_models/3image_finish1_b.txt", 0, "D:/CODE/ssd_models/result_back.txt", 0);
  


  detect_init_front_refl(model_file_refl, weights_file_refl, 0, 380, 300, 300, 0.98);
  detect_init_back_refl(model_file_refl, weights_file_refl, 0, 380, 300, 300, 0.98);

  //注意：detect_txt_front_refl这个函数读取的object_txt文件的格式是[minx,miny,w,h,score,id]//不能缺少id信息
  int privacy_x;
  int privacy_y;
  long t3 = GetTickCount();
  int a = detect_txt_front_refl("D:/CODE/ssd_models/3image_finish1_b.txt", "D:/CODE/ssd_models/result.txt", 1, privacy_x, privacy_y, "D:/CODE/ssd_models/result_refl.txt");
  long t4 = GetTickCount();
  std::cout << "reflection forward time：" << (t4 - t3) << std::endl;


  int b = detect_txt_back_refl("D:/CODE/ssd_models/3image_finish_b.txt", "D:/CODE/ssd_models/result_back.txt", 0, privacy_x, privacy_y,"D:/CODE/ssd_models/result_back_refl.txt");
  std::cout << "privacy_x：" << privacy_x << std::endl;
  return 0;
}

#endif  // USE_OPENCV
