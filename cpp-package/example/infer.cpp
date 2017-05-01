#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>
#include <map>
#include <chrono>

using namespace mxnet::cpp;
using namespace std;

int main(int argc, char *argv[])
{
	int batch_size = 1;
	int depth = stoi(argv[2]);
	int patch_size = stoi(argv[3]);
  string param_str = "d" + to_string(depth) + "p" + to_string(patch_size);
  string param_name = "vdsr_" + param_str + "_param_" + argv[4];
	auto ctx = Context::gpu(3);
  string path = argv[1];

	auto model = Symbol::Load("vdsr." + param_str + ".model");
	std::map<std::string, NDArray> args_map;

	args_map["data"] = NDArray(Shape(batch_size, 1, patch_size, patch_size), ctx, false);
	args_map["data_label"] = NDArray(Shape(batch_size, 1, patch_size, patch_size), ctx, false);
	model.InferArgsMap(ctx, &args_map, args_map);

  NDArray::Load(param_name, nullptr, &args_map);

  auto *exec = model.SimpleBind(ctx, args_map);

  vector<float> data_buf(patch_size * patch_size);
  vector<float> label_buf(patch_size * patch_size);
  int batch = 0;
  cv::Mat origin = cv::imread(path);
  cv::Mat yuv;
  cv::cvtColor(origin, yuv, cv::COLOR_BGR2YUV);
  cv::Mat float_mat, half_mat;
  yuv.convertTo(float_mat, CV_32F, 1/255.0);
  cv::resize(float_mat, half_mat, cv::Size(0, 0), 0.5, 0.5, cv::INTER_CUBIC);
  cv::resize(half_mat, half_mat, float_mat.size(), 0, 0, cv::INTER_CUBIC);
  
  assert(float_mat.channels() == 3);
  vector<cv::Mat> channels, half_channels;
  cv::split(float_mat, channels);
  cv::Mat image = channels[0];
  cv::split(half_mat, half_channels);
  cv::Mat half_img = half_channels[0];
  cv::Mat result(half_img.size(), CV_32F);
  cv::Mat overlap(half_img.size(), CV_32F);
  auto shape = image.size();
  int w = shape.width;
  int h = shape.height;
  PSNR acu;
  for (int i = 0; i + patch_size <= w; i += patch_size/4) {
    for (int j = 0; j + patch_size <= h; j += patch_size/4) {
      auto patch = image({i, j, patch_size, patch_size});
      for (int r = 0; r < patch.rows; ++r) {
        label_buf.insert(label_buf.end(),
            patch.ptr<float>(r), patch.ptr<float>(r)+patch.cols);
      }

      auto half_patch = half_img({i, j, patch_size, patch_size});
      for (int r = 0; r < half_patch.rows; ++r) {
        data_buf.insert(data_buf.end(),
            half_patch.ptr<float>(r), half_patch.ptr<float>(r)+half_patch.cols);
      }

      NDArray label(label_buf, Shape(batch_size, 1, patch_size, patch_size), ctx);
      NDArray resi(data_buf, Shape(batch_size, 1, patch_size, patch_size), ctx);
      resi.CopyTo(&args_map["data"]);
      label.CopyTo(&args_map["data_label"]);
      
      NDArray::WaitAll();
      exec->Forward(false);
      exec->outputs[0].SyncCopyToCPU(&data_buf);
      acu.Update(label, exec->outputs[0]);
      cv::Mat waifu(patch_size, patch_size, CV_32F, data_buf.data());

      //waifu.copyTo(result({i, j, patch_size, patch_size}));
      result({i, j, patch_size, patch_size}) += waifu;
      overlap({i, j, patch_size, patch_size}) += 1.0f;

      label_buf.clear();
      data_buf.clear();
    }
  }

  for (int x = 0; x + patch_size/4 <= w; x += patch_size/4) {
    for (int y = 0; y + patch_size/4 <= h; y += patch_size/4) {
      result({x, y, patch_size/4, patch_size/4}) /=
        overlap({x, y, patch_size/4, patch_size/4});
    }
  }

  LG << acu.Get();

  cv::Mat out_channels[3];
  out_channels[0] = result;
  out_channels[1] = half_channels[1];
  out_channels[2] = half_channels[2];
  cv::Mat output_yuv;
  merge(out_channels, 3, output_yuv);
  cv::Mat output_float;
  cv::cvtColor(output_yuv, output_float, cv::COLOR_YUV2BGR);
  cv::Mat output_image;
  output_float.convertTo(output_image, CV_8U, 255.0);
  cv::imwrite("output.png", output_image, {CV_IMWRITE_PNG_COMPRESSION, 0});

  cv::cvtColor(half_mat, output_float, cv::COLOR_YUV2BGR);
  output_float.convertTo(output_image, CV_8U, 255.0);
  cv::imwrite("half.png", output_image, {CV_IMWRITE_PNG_COMPRESSION, 0});

	MXNotifyShutdown();
	return 0;
}
