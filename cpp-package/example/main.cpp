#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/opencv.hpp>
#include <experimental/filesystem>

#include <string>
#include <vector>
#include <map>
#include <chrono>

#include "PSNR.h"

using namespace mxnet::cpp;
using namespace std;
namespace fs = std::experimental::filesystem;

inline Symbol ConvFactory(Symbol data,
              int num_filter,
						  Shape kernel,
						  Shape stride = Shape(1, 1),
						  Shape pad = Shape(0, 0),
						  const string & name = "",
						  bool relu = true)
{
	Symbol conv_w("conv" + name + "_w"), conv_b("conv" + name + "_b");

	Symbol conv = Convolution("conv" + name, data,
							  conv_w, conv_b, kernel,
							  num_filter, stride, Shape(1, 1), pad);
	return relu ? Activation("relu" + name, conv, ActivationActType::relu) : conv;
}

Symbol Model(int patch_size, int depth)
{
	Symbol data = Symbol::Variable("data");
	Symbol data_label = Symbol::Variable("data_label");

	auto conv1 = ConvFactory(data, 64, Shape(3, 3), Shape(1, 1), Shape(1, 1), "1");

	vector<Symbol> conv;
	conv.push_back(conv1);
	for (int i = 2; i <= depth-1; ++i) {
		string layerID = to_string(i);
		Symbol convx = ConvFactory(conv.back(), 64, Shape(3, 3), Shape(1, 1), Shape(1, 1), layerID);
		conv.push_back(convx);
	}

	auto conv20 = ConvFactory(conv.back(), 1, Shape(3, 3), Shape(1, 1), Shape(1, 1), to_string(depth), false);

  auto pred = conv20 + data;
  auto diff = pred - data_label;
  //auto pred = conv20;
  //auto diff = conv20;

  auto l2 = sum("sum", diff*diff / 2.0f, Shape(2, 3)) ;

  auto loss = MakeLoss(l2);

  return Symbol::Group({BlockGrad(pred), loss});
	//auto res = broadcast_add("sum", data, conv20);

	//return LinearRegressionOutput("output", res, data_label);
	//return LinearRegressionOutput("output", conv20, data_label);
}

vector<pair<NDArray, NDArray>> ReadImages(vector<string> paths, int patch_size, int batch_size)
{
  vector<pair<NDArray, NDArray>> results;
  vector<float> data_buf(batch_size * patch_size * patch_size);
  vector<float> label_buf(batch_size * patch_size * patch_size);
  int batch = 0;
  for (auto& path : paths) {
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
    auto shape = image.size();
    int w = shape.width;
    int h = shape.height;
    int count = 0;
    for (int i = 0; i + patch_size <= w; i += patch_size) {
      for (int j = 0; j + patch_size <= h; j += patch_size) {
        ++batch;
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

        if (batch == batch_size) {
          NDArray label(label_buf, Shape(batch_size, 1, patch_size, patch_size), Context::cpu());
          NDArray resi(data_buf, Shape(batch_size, 1, patch_size, patch_size), Context::cpu());
          label_buf.clear();
          data_buf.clear();
          results.emplace_back(resi, label);
          batch = 0;
        }
        ++count;
      }
    }
    cout << "\rSplit " << path << " into " << count << " patches";
    cout.flush();
  }
  cout << "\rTotal patches: " << results.size() << "*" << batch_size
       << " = " << results.size() * batch_size << endl;
  return results;
}

pair<vector<string>, vector<string>> ListFolder(string folder, float ratio,
    string suffix = ".png") {
  vector<string> train, val;
  for (auto &entry : fs::directory_iterator(folder)) {
    string name = entry.path().string();
    if (name.substr(name.length() - suffix.length()) == suffix) {
      train.push_back(name);
    }
  }
  size_t count = train.size();
  size_t train_count = count * ratio;
  size_t val_count = count - train_count;
  val = vector<string>(train.begin(), train.begin() + val_count);
  train.erase(train.begin(), train.begin() + val_count);
  return {train, val};
}

int main(int argc, char *argv[])
{
  int depth = 22;
	int batch_size = 48;
	int max_epoch = 250;
	int patch_size = 128;
	float learning_rate = 0.00001;
	float weight_decay = 1e-4;
	auto ctx = Context::gpu(3);

  string param_str = "d" + to_string(depth) + "p" + to_string(patch_size);

  vector<string> train_paths, val_paths;
  tie(train_paths, val_paths) = ListFolder("/data/xlidc/png", 0.9);
  auto train_data = ReadImages(train_paths, patch_size, batch_size);
  auto val_data = ReadImages(val_paths, patch_size, batch_size);

	auto model = Model(patch_size, depth);
	map<string, NDArray> args_map;

	args_map["data"] = NDArray(Shape(batch_size, 1, patch_size, patch_size), ctx, false);
	args_map["data_label"] = NDArray(Shape(batch_size, 1, patch_size, patch_size), ctx, false);
	model.InferArgsMap(ctx, &args_map, args_map);

	model.Save("vdsr." + param_str + ".model");

  int start_iter = 0;
  if (argc == 1) {
    LG << "Training from scratch";
    auto initializer = Normal(0, sqrt(2.0 / (9.0 * patch_size)));
    for (auto &arg : args_map) {
      if (arg.first.find('w') != string::npos) {
        initializer(arg.first, &arg.second);
      }
      if (arg.first.find('b') != string::npos) {
        Zero()(arg.first, &arg.second);
      }
    }

    initializer = Normal(0, sqrt(2 / 9.0 / patch_size));
    initializer("conv1_w", &(args_map["conv1_w"]));
  } else {
    string param_name = "vdsr_" + param_str + "_param_" + argv[1];
    LG << "Loading " << param_name;
    start_iter = stoi(argv[1]) + 1;
    NDArray::Load(param_name, nullptr, &args_map);
  }

	// use grayscale 41x41 bmp

	Optimizer* opt = OptimizerRegistry::Find("adam");
	opt->SetParam("rescale_grad", 1.0 / batch_size);

  Monitor mon(100, regex("conv.*_output|conv.*_w"));
  //Monitor shapemon(100, regex("conv.*_output|conv.*_w"), [](NDArray in) {
  //vector<float> s; for (auto d : in.GetShape()) s.push_back(d); return NDArray( s, Shape(in.GetShape().size()), in.GetContext());
  //    return Operator("sum").SetInput("data", in).Invoke()[0];
  //});

  //Monitor mon(100);
  auto *exec = model.SimpleBind(ctx, args_map);

  //mon.install(exec);
  //shapemon.install(exec);

	for (int iter = start_iter; iter < max_epoch; ++iter) {
		LG << "Epoch: " << iter;
		auto tic = chrono::system_clock::now();
		int samples = 0;

    PSNR train_psnr;
    for (auto& data : train_data) {
      //mon.tic();
      //shapemon.tic();
			samples += batch_size;
			data.first.CopyTo(&args_map["data"]);
			data.second.CopyTo(&args_map["data_label"]);
			NDArray::WaitAll();

			exec->Forward(true);
			exec->Backward();
			exec->UpdateAll(opt, learning_rate, weight_decay);
      train_psnr.Update(data.second, exec->outputs[0]);
      //mon.toc_print();
      //shapemon.toc_print();

			if (samples % (100 * batch_size) == 0) {
				LG << "Epoch:\t" << iter << " : " << samples << " PSNR: " << train_psnr.Get();
			}
		}

		auto toc = chrono::system_clock::now();

		PSNR acu, data_ac;
		if (1) {
      for (auto &data : val_data) {
				data.first.CopyTo(&args_map["data"]);
				data.second.CopyTo(&args_map["data_label"]);
				NDArray::WaitAll();

				exec->Forward(false);
				NDArray::WaitAll();
				acu.Update(data.second, exec->outputs[0]);
				data_ac.Update(data.first, data.second);
			}
		}

		string save_path_param = "./vdsr_" + param_str + "_param_" + to_string(iter);
		auto save_args = args_map;
		save_args.erase(save_args.find("data"));
		save_args.erase(save_args.find("data_label"));
		LG << "ITER:\t" << iter << " Saving to..." << save_path_param;
		NDArray::Save(save_path_param, save_args);

		float duration = chrono::duration_cast<chrono::milliseconds>(toc - tic).count() / 1000.0;
		LG << "Epoch:\t" << iter << " "
			<< samples / duration * patch_size * patch_size
			<< " pixel/sec in "
			<< duration << "s PSNR: " << acu.Get() << "/" << data_ac.Get();
	}

	MXNotifyShutdown();
	return 0;
}
