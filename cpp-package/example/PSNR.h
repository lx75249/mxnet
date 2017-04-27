#pragma once

#include <mxnet-cpp/MxNetCpp.h>

//#include <opencv2\opencv.hpp>

using namespace mxnet::cpp;

/*
class PSNR : public EvalMetric
{
public:
	PSNR() : EvalMetric("PSNR")
	{
	}

	// gray scale
	void Update(NDArray labels, NDArray preds)
	{
		CHECK_EQ(labels.GetShape().size(), 4);

		mx_uint len = 1;
		for (const auto &size : labels.GetShape()) {
			len *= size;
		}

		std::vector<mx_float> pred_data(len);
		std::vector<mx_float> label_data(len);

		preds.ArgmaxChannel().SyncCopyToCPU(&pred_data, len);
		labels.SyncCopyToCPU(&label_data, len);

		NDArray::WaitAll();

		sum_metric += GetPSNR(cv::Mat(label_data), cv::Mat(pred_data));
		num_inst += 1;
	}

	double GetPSNR(const cv::Mat& I1, const cv::Mat& I2)
	{
		cv::Mat s1;
		cv::absdiff(I1, I2, s1);       // |I1 - I2|
		s1.convertTo(s1, CV_32F, 1 / 255.0);  // cannot make a square on 8 bits
		s1 = s1.mul(s1);           // |I1 - I2|^2

		cv::Scalar s = cv::sum(s1);         // sum elements per channel

		double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

		if (sse <= 1e-10) // for small values return zero
			return 0;
		else {
			double mse = sse / (double)(I1.channels() * I1.total());
			double psnr = 10.0*log10((255 * 255) / mse);
			return psnr;
		}
	}

	double GetPSNREx(const cv::Mat& I1, const cv::Mat &I2)
	{
		if (!I1.isContinuous() || !I2.isContinuous()) {
			return GetPSNR(I1, I2);
		}

		cv::Mat s1;
		cv::absdiff(I1, I2, s1);
		s1.convertTo(s1, CV_32F, 1 / 255.0);
		return 0;
	}
};

static int count = 0;

class MSError : public EvalMetric
{
public:
	MSError() : EvalMetric("MSError")
	{
	}

	// gray scale
	void Update(NDArray labels, NDArray preds)
	{
		CHECK_EQ(labels.GetShape().size(), 4);

		mx_uint len = 1;
		for (const auto &size : labels.GetShape()) {
			len *= size;
		}

		std::vector<mx_float> pred_data(len);
		std::vector<mx_float> label_data(len);

		preds.ArgmaxChannel().SyncCopyToCPU(&pred_data, len);
		labels.SyncCopyToCPU(&label_data, len);

		NDArray::WaitAll();

		sum_metric += eqm(cv::Mat(label_data), cv::Mat(pred_data));
		num_inst += 1;
	}

	double eqm(const cv::Mat & img1, const cv::Mat & img2)
	{
		double res = 0;

		cv::Mat diff;
		cv::absdiff(img1, img2, diff);
		diff.convertTo(diff, CV_32F, 1 / 255.0);

		int height = img1.rows;
		int width = img1.cols;

		concurrency::parallel_for(int(0), height, [&](int i) {
			for (int j = 0; j < width; j++) {
				res += (diff.at<double>(i, j) * diff.at<double>(i, j));
			}
		});

		//		res /= height * width;
		return res;
	}
};
*/

class DiffError : public EvalMetric
{
public:
	float m = 65535.0;

	DiffError() : EvalMetric("DiffError")
	{
	}

	// gray scale
	void Update(NDArray labels, NDArray preds)
	{
		CHECK_EQ(labels.GetShape().size(), 4);

		mx_uint len = 1;
		for (const auto &size : labels.GetShape()) {
			len *= size;
		}

		std::vector<mx_float> pred_data(len);
		std::vector<mx_float> label_data(len);

		preds.SyncCopyToCPU(&pred_data);
		labels.SyncCopyToCPU(&label_data);

		NDArray::WaitAll();

		int batch_size = preds.GetShape()[0];
		int width = preds.GetShape()[2];
		int height = preds.GetShape()[3];

		// concurrency::parallel_for(int(0), int(batch_size), [&](int i) {
		for (int i = 0; i < batch_size; ++i) {
			float sum = 0;
			for (int j = 0; j < width * height; ++j) {
				sum += fabs(pred_data[i] - label_data[i])/255.0;
			}

      /*
			if (sum <= 840) {
				// save
				cv::Mat pred(pred_data);
				pred.convertTo(pred, CV_8UC1);
				pred.reshape(pred.channels(), 41);
				pred.rows = 2624;
				pred.cols = 41;
				//cv::imwrite("pred_data.bmp", pred);

				cv::Mat lable(label_data);
				lable.reshape(lable.channels(), 41);
				lable.rows = 2624;
				lable.cols = 41;
				//cv::imwrite("label_data.bmp", lable);
			}
      */

			sum_metric += sum;
		}
		//});

		num_inst += batch_size;
	}
};
