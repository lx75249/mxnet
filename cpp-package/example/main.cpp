#include <mxnet-cpp/MxNetCpp.h>

#include <string>
#include <vector>
#include <map>
#include <chrono>

#include "PSNR.h"

using namespace mxnet::cpp;

inline Symbol ConvFactory(Symbol data, int num_filter,
						  Shape kernel,
						  Shape stride = Shape(1, 1),
						  Shape pad = Shape(0, 0),
						  const std::string & name = "",
						  bool relu = true)
{
	Symbol conv_w("conv" + name + "_w"), conv_b("conv" + name + "_b");

	Symbol conv = Convolution(data,
							  conv_w, conv_b, kernel,
							  num_filter, stride, Shape(1, 1), pad);
	return relu ? Activation(conv, ActivationActType::relu) : conv;
}

Symbol Model(int patch_size)
{
	Symbol data = Symbol::Variable("data");
	Symbol data_label = Symbol::Variable("data_label");

	auto conv1 = ConvFactory(data, 64, Shape(3, 3), Shape(1, 1), Shape(1, 1), "1");

	std::vector<Symbol> conv;
	conv.push_back(conv1);
	for (int i = 2; i <= 5; ++i) {
		std::string layerID = std::to_string(i);
		Symbol convx = ConvFactory(conv.back(), 64, Shape(3, 3), Shape(1, 1), Shape(1, 1), layerID);
		conv.push_back(convx);
	}

	auto conv20 = ConvFactory(conv.back(), 1, Shape(3, 3), Shape(1, 1), Shape(1, 1), "20", false);

  auto pred = conv20 + data;
  auto diff = pred - data_label;
  //auto pred = conv20;
  //auto diff = conv20;

  auto l2 = sum(diff*diff, Shape(2, 3)) / (1.0f * patch_size * patch_size);

  auto loss = MakeLoss(l2);

  return Symbol::Group({BlockGrad(pred), loss});
	//auto res = broadcast_add("sum", data, conv20);

	//return LinearRegressionOutput("output", res, data_label);
	//return LinearRegressionOutput("output", conv20, data_label);
}

int main()
{
	int batch_size = 64;
	int max_epoch = 20;
	int patch_size = 41;
	float learning_rate = 0.01;
	float weight_decay = 1e-4;
	auto ctx = Context::gpu();
	// auto ctx = Context::cpu();

	auto model = Model(patch_size);
	std::map<std::string, NDArray> args_map;
	std::map<std::string, NDArray> aux_map;

	args_map["data"] = NDArray(Shape(batch_size, 1, patch_size, patch_size), ctx, false);
	args_map["data_label"] = NDArray(Shape(batch_size, 1, patch_size, patch_size), ctx, false);
	model.InferArgsMap(ctx, &args_map, args_map);

	auto json = model.ToJSON();

	auto initializer = Normal(0, sqrt(2.0 / 9.0 / 64.0));
	for (auto &arg : args_map) {
		if (arg.first.find('w')) {
			initializer(arg.first, &arg.second);
		}
		if (arg.first.find('b')) {
			Zero()(arg.first, &arg.second);
		}
	}

	initializer = Normal(0, sqrt(2 / 9.0));
	initializer("conv1_w", &(args_map["conv1_w"]));

	// use grayscale 41x41 bmp
	auto train_iter = MXDataIter("ImageRecordIter")
		.SetParam("path_imglist", "./list_folder/train.lst")
		.SetParam("path_imgrec", "./list_folder/train.rec")
		.SetParam("data_shape", Shape(1, patch_size, patch_size))
		.SetParam("batch_size", batch_size)
		.SetParam("label_width", 1)
		.CreateDataIter();

	auto train_label_iter = MXDataIter("ImageRecordIter")
		.SetParam("path_imglist", "./list_folder/train_label.lst")
		.SetParam("path_imgrec", "./list_folder/train_label.rec")
		.SetParam("data_shape", Shape(1, patch_size, patch_size))
		.SetParam("batch_size", batch_size)
		.SetParam("label_width", 1)
		.CreateDataIter();

	auto val_iter = MXDataIter("ImageRecordIter")
		.SetParam("path_imglist", "./list_folder/val.lst")
		.SetParam("path_imgrec", "./list_folder/val.rec")
		.SetParam("data_shape", Shape(1, patch_size, patch_size))
		.SetParam("batch_size", batch_size)
		.SetParam("label_width", 1)
		.CreateDataIter();

	auto val_label_iter = MXDataIter("ImageRecordIter")
		.SetParam("path_imglist", "./list_folder/val_label.lst")
		.SetParam("path_imgrec", "./list_folder/val_label.rec")
		.SetParam("data_shape", Shape(1, patch_size, patch_size))
		.SetParam("batch_size", batch_size)
		.SetParam("label_width", 1)
		.CreateDataIter();

  /*
	Optimizer* opt = OptimizerRegistry::Find("adam");
	opt->SetParam("rescale_grad", 1.0 / batch_size)
     ->SetParam("beta1", 0.9)
     ->SetParam("beta2", 0.999)
     ->SetParam("epsilon", 1e-8);
  */
	Optimizer* opt = OptimizerRegistry::Find("sgd");
	opt->SetParam("rescale_grad", 1.0 / batch_size);
  //opt->SetParam("clip_gradient", 100);
	opt->SetParam("momentum", 0.9);
  opt->SetParam("clip_gradient", 0.1);

	for (int iter = 0; iter < max_epoch; ++iter) {
		LG << "Epoch: " << iter;
		auto tic = std::chrono::system_clock::now();
		int samples = 0;

		train_iter.Reset();
		train_label_iter.Reset();
		while (train_iter.Next(), train_label_iter.Next()) {
			samples += batch_size;
			auto data_batch = train_iter.GetDataBatch();
			auto label_batch = train_label_iter.GetDataBatch();
      NDArray norm_d = data_batch.data/256.0;
      NDArray norm_l = label_batch.data/256.0;
			norm_d.CopyTo(&args_map["data"]);
			norm_l.CopyTo(&args_map["data_label"]);
			NDArray::WaitAll();

			auto *exec = model.SimpleBind(ctx, args_map);
			exec->Forward(true);
			exec->Backward();
			exec->UpdateAll(opt, learning_rate, weight_decay);

			if (samples % (100 * batch_size) == 0) {
				LG << "Epoch:\t" << iter << " : " << samples;// << " Acc: " << acu.Get();
			}

			delete exec;
		}

		auto toc = std::chrono::system_clock::now();

		DiffError acu;
		if (1) {
			val_iter.Reset();
			val_label_iter.Reset();
			while (val_iter.Next(), val_label_iter.Next()) {
				auto data_batch = val_iter.GetDataBatch();
				auto label_batch = val_label_iter.GetDataBatch();
        NDArray norm_d = data_batch.data/256.0;
        NDArray norm_l = label_batch.data/256.0;
				norm_d.CopyTo(&args_map["data"]);
				norm_l.CopyTo(&args_map["data_label"]);
				NDArray::WaitAll();

				auto *exec = model.SimpleBind(ctx, args_map);
				exec->Forward(false);
				NDArray::WaitAll();
				acu.Update(norm_l, exec->outputs[0]);
				delete exec;
			}
		}

    /*
		std::string save_path_param = "./model/vdsr_param_" + std::to_string(iter);
		auto save_args = args_map;
		save_args.erase(save_args.find("data"));
		save_args.erase(save_args.find("data_label"));
		LG << "ITER:\t" << iter << " Saving to..." << save_path_param;
		NDArray::Save(save_path_param, save_args);
    */

		float duration = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count() / 1000.0;
		LG << "Epoch:\t" << iter << " "
			<< samples / duration * patch_size * patch_size
			<< " pixel/sec in "
			<< duration << "s Accuracy: " << acu.Get();
	}

	MXNotifyShutdown();
	return 0;
}
