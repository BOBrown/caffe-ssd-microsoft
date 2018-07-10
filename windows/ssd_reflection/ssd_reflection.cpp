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
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif  // USE_OPENCV
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef USE_OPENCV
using namespace caffe;  // NOLINT(build/namespaces)

int HEIGHT = 380;

#define WIDTH 190


class Detector {
public:
	Detector();

	std::vector<vector<float> > Detect(const cv::Mat& img);
	void init(const string& model_file, const string& weights_file, const string& mean_file, const string& mean_value);
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
	const string& mean_value) {
#ifdef CPU_ONLY
	Caffe::set_mode(Caffe::CPU);
#else
	Caffe::set_mode(Caffe::GPU);
#endif

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
DEFINE_double(confidence_threshold, 0.01,
	"Only store detections with score higher than the threshold.");


extern "C"_declspec(dllexport) void global_init();
extern "C"_declspec(dllexport) void detect_init_front(const char *caffemodel, const char *network_pt, int GPUID, int h, int max_in, int min_in, float thresh);
extern "C"_declspec(dllexport) int detect_txt_front(const char *input_txt, bool rotate, const char *output_txt, bool vis);//
extern "C"_declspec(dllexport) void detect_init_back(const char *caffemodel, const char *network_pt, int GPUID, int h, int max_in, int min_in, float thresh);
extern "C"_declspec(dllexport) int detect_txt_back(const char *input_txt, bool rotate, const char *output_txt, bool vis);

Detector detector;
float confidence_threshold;

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
int write_to_txt(const char *output_path, std::vector<vector<float> > detections, int pro_flag)
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
		if (score >= confidence_threshold)
		{
			int w = int(d[5] * WIDTH) - int(d[3] * WIDTH);
			int h = int(d[6] * HEIGHT) - int(d[4] * HEIGHT);
			out << static_cast<int>(d[1]) << " ";
			out << score << " ";
			out << static_cast<int>(d[3] * WIDTH) << " ";
			out << static_cast<int>(d[4] * HEIGHT) << " ";
			out << w << " ";
			out << h << std::endl;

			fprintf(f, "%d %d %d %d %f %d\n", int(d[3] * WIDTH), int(d[4] * HEIGHT), w, h, score, int(d[1]));
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
	detector.init(network_pt, caffemodel, FLAGS_mean_file, FLAGS_mean_value);
	confidence_threshold = thresh;
	printf(" detector.init() for front -> done.\n");
}

__declspec(dllexport) void detect_init_back(const char *network_pt, const char *caffemodel, int GPUID, int h, int max_in, int min_in, float thresh)
{
	detector.init(network_pt, caffemodel, FLAGS_mean_file, FLAGS_mean_value);
	confidence_threshold = thresh;
	printf(" detector.init() for back -> done.\n");
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
	std::vector<vector<float> > detections = detector.Detect(img);

	for (int i = 0; i < detections.size(); ++i) {
		const vector<float>& d = detections[i];
		// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
		CHECK_EQ(d.size(), 7);
		const float score = d[2];
		if (score >= confidence_threshold) {
			pro_flag++;
		}
	}
	//changed by holobo
	write_to_txt(output_txt, detections, pro_flag);
	printf("-->detect done!\n");
	if (returnvalue < 0)
		return returnvalue;
	return 1;
}

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

	std::vector<vector<float> > detections = detector.Detect(img);

	for (int i = 0; i < detections.size(); ++i) {
		const vector<float>& d = detections[i];
		// Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
		CHECK_EQ(d.size(), 7);
		const float score = d[2];
		if (score >= confidence_threshold) {
			pro_flag++;
		}
	}
	//changed by holobo
	write_to_txt(output_txt, detections, pro_flag);
	printf("-->detect done!\n");
	if (returnvalue < 0)
		return returnvalue;
	return 1;
}

/*int calcIOU(int one_x, int one_y, int one_w, int one_h, int two_x, int two_y, int two_w, int two_h)
{
if ((abs(one_x - two_x) < ((one_w + two_w) / 2.0)) && (abs(one_y - two_y) < ((one_h + two_h) / 2.0)))
{
lu_x_inter = max((one_x - (one_w / 2.0)), (two_x - (two_w / 2.0)))
lu_y_inter = min((one_y + (one_h / 2.0)), (two_y + (two_h / 2.0)))

rd_x_inter = min((one_x + (one_w / 2.0)), (two_x + (two_w / 2.0)))
rd_y_inter = max((one_y - (one_h / 2.0)), (two_y - (two_h / 2.0)))

inter_w = abs(rd_x_inter - lu_x_inter)
inter_h = abs(lu_y_inter - rd_y_inter)

inter_square = inter_w * inter_h
union_square = (one_w * one_h) + (two_w * two_h) - inter_square

calcIOU = inter_square / union_square * 1.0
print("calcIOU:", calcIOU)
}
else
{
print("No intersection!")
calcIOU = 0
}
return calcIOU
}
*/

int main() {


	const char * model_file = "D:\\CODE\\ssd_models\\models\\VGGNet\\VOC0712Plus\\SSD_300x300\\deploy.prototxt";
	const char * weights_file = "D:\\CODE\\ssd_models\\models\\VGGNet\\VOC0712Plus\\SSD_300x300\\changed_lap_VGG_VOC0712_SSD_300x300_iter_42000.caffemodel";


	detect_init_front(model_file, weights_file, 0, 380, 300, 300, 0.4);
	detect_init_back(model_file, weights_file, 0, 380, 300, 300, 0.4);
	detect_txt_front("D:/CODE/ssd_models/image_finish1_b.txt", 1, "D:/CODE/ssd_models/result.txt", 0);
	detect_txt_back("D:/CODE/ssd_models/image_finish_b.txt", 0, "D:/CODE/ssd_models/result_back.txt", 0);


	return 0;
}

#endif  // USE_OPENCV
