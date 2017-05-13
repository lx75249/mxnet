#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/opencv.hpp>

#include <array>
#include <thread>
#include <condition_variable>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <cmath>
#include <experimental/filesystem>

#include "PSNR.h"

using namespace mxnet::cpp;
namespace fs = std::experimental::filesystem;

inline Symbol ConvFactory(Symbol data,
              int num_filter,
              Shape kernel,
              Shape stride = Shape(1, 1),
              Shape pad = Shape(0, 0),
              const std::string & name = "",
              bool lrelu = true)
{
  Symbol conv_w("conv" + name + "_w"), conv_b("conv" + name + "_b");

  Symbol conv = Convolution("conv" + name, data,
                conv_w, conv_b, kernel,
                num_filter, stride, Shape(1, 1), pad);
  return lrelu ? LeakyReLU("leakyrelu_" + name, conv, LeakyReLUActType::leaky, 0.2) : conv;
}

inline Symbol ConvTranspose(Symbol data, int num_filter, int scale,
              const std::string & name = "",
              bool lrelu = true)
{
  Symbol conv_w("dconv" + name + "_w"), conv_b("dconv" + name + "_b");
  Symbol deconv = Deconvolution("dconv" + name, data,
                  conv_w, conv_b,
                  Shape(scale * 2, scale * 2), num_filter,
                  Shape(scale, scale), Shape(),
                  Shape(scale/2, scale/2), Shape(), Shape(), 1, 512, false);
  return lrelu ? LeakyReLU("leakyrelu_" + name, deconv, LeakyReLUActType::leaky, 0.2) : deconv;
}

inline Symbol FeatureExtFactory(Symbol data, int level, int depth = 5)
{
  std::vector<Symbol> conv;
  //auto padding = Pad(data, PadMode::constant, Shape(0, 1, 1, 1), 0.0);
  //conv.push_back(padding);
  conv.push_back(data);
  for (int i = 1; i <= depth; ++i) {
    std::string layerID = std::to_string(level) + "_" + std::to_string(i);
    Symbol convx = ConvFactory(conv.back(), 64, Shape(3, 3), Shape(1, 1), Shape(1, 1), layerID);
    conv.push_back(convx);
  }

  return ConvTranspose(conv.back(), 64, 2, std::to_string(level));
}

Symbol Modelx4(int depth = 10, int depthx2 = 0)
{
  if (depthx2 < 1) {
    depthx2 = depth;
  }

  Symbol data = Symbol::Variable("data");
  Symbol label_x2 = Symbol::Variable("data_label_x2");
  Symbol label_x4 = Symbol::Variable("data_label_x4");
  // Symbol label_x8 = Symbol::Variable("data_label_x8");

  auto conv0 = ConvFactory(data, 64, Shape(3, 3), Shape(1, 1), Shape(1, 1), "0");

  auto convF1 = FeatureExtFactory(conv0, 1, depthx2);
  auto convI1 = ConvTranspose(data, 1, 2, "I1", false);
  auto convR1 = ConvFactory(convF1, 1, Shape(3, 3), Shape(1, 1), Shape(1, 1), "R1", false);
  auto HRx2 = convI1 + convR1; // x2
  auto R2 = label_x2 - HRx2;
  auto L1 = sum("sum1", sqrt(square(R2) + 1e-3*1e-3), Shape(2, 3));

  auto convF2 = FeatureExtFactory(convF1, 2, depth);
  auto convI2 = ConvTranspose(HRx2, 1, 2, "I2", false);
  auto convR2 = ConvFactory(convF2, 1, Shape(3, 3), Shape(1, 1), Shape(1, 1), "R2", false);
  auto HRx4 = convI2 + convR2; // x4
  auto R4 = label_x4 - HRx4;
  auto L2 = sum("sum2", sqrt(square(R4) + 1e-3*1e-3), Shape(2, 3));

  //auto convF3 = FeatureExtFactory(convF2, 3, 5);
  //auto convI3 = ConvTranspose(HRx4, 1, 2, "I3");
  //auto convR3 = ConvFactory(convF3, 1, Shape(4, 4), Shape(1, 1), Shape(1, 1), "R3");
  //auto HRx8 = convI3 + convR3; // x8
  //auto R8 = label_x8 - HRx8;
  //auto L3 = sum("sum3", sqrt(square(R8) + 1e-3*1e-3), Shape(2, 3));

  auto robustLoss = L1 + L2;
  auto loss = MakeLoss(robustLoss);

  return Symbol::Group({BlockGrad(HRx2), BlockGrad(HRx4), loss});
  // return Symbol::Group({ BlockGrad(Symbol::Group({HRx2, HRx4})), loss });
}

template <int N = 1>
std::vector<std::array<NDArray, N+1>> ReadImages(std::vector<std::string> paths, int patch_size, int batch_size)
{
  std::vector<std::array<NDArray, N+1>> results;
  std::mutex mutex;
  auto read_thread = [patch_size, batch_size, &results, &mutex](std::vector<std::string> paths) {
    int batch = 0;
    std::array<std::vector<float>, N+1> bufs;
    for (int i = 0; i <= N; ++i) {
      bufs[i].reserve(batch_size * patch_size * patch_size << (2*i));
    }

    for (auto& path : paths) {
      cv::Mat origin = cv::imread(path);
      cv::Mat yuv;
      cv::cvtColor(origin, yuv, cv::COLOR_BGR2YUV);
      cv::Mat mat;

      yuv.convertTo(mat, CV_32F, 1 / 255.0);
      assert(mat.channels() == 3);

      std::array<cv::Mat, N+1> scaled_mats;

      std::vector<cv::Mat> channels;
      for (int i = 0; i <= N; ++i) {
        float scale = 1.0f / (1 << (N-i));
        cv::Mat scaled_mat;
        cv::resize(mat, scaled_mat, cv::Size(0, 0), scale, scale, cv::INTER_CUBIC);
        channels.clear();
        cv::split(scaled_mat, channels);
        scaled_mats[i] = channels[0];
      }

      auto shape = scaled_mats[0].size();
      int w = shape.width;
      int h = shape.height;
      int count = 0;
      for (int i = 0; i + patch_size <= w; i += patch_size) {
        for (int j = 0; j + patch_size <= h; j += patch_size) {
          ++batch;
          for (int s = 0; s <= N; ++s) {
            auto patch = scaled_mats[s]({ i << s, j << s, patch_size << s, patch_size << s });
            for (int r = 0; r < patch.rows; ++r) {
              bufs[s].insert(bufs[s].end(),
                  patch.template ptr<float>(r),
                  patch.template ptr<float>(r) + patch.cols);
            }
          }

          if (batch == batch_size) {
            std::array<NDArray, N+1> data_batch;
            for (int s = 0; s <= N; ++s) {
              data_batch[s] = NDArray(bufs[s],
                  Shape(batch_size, 1, patch_size<<s, patch_size<<s),
                  Context::cpu());
              bufs[s].clear();
            }
            {
              std::lock_guard<std::mutex> guard(mutex);
              results.push_back(std::move(data_batch));
            }
            batch = 0;
          }
          ++count;
        }
      }
      std::cerr << "\33[2K\rSplit " << path << " into " << count << " patches";
    }
  };

  const int num_threads = std::min(std::thread::hardware_concurrency(), 8U);
  std::vector<std::thread> threads;
  for (int i = 0; i < num_threads; ++i) {
    int piece = (paths.size() + num_threads - 1) / num_threads;
    auto begin = paths.begin() + piece * i;
    auto end = (i == num_threads-1) ? paths.end() : begin + piece;
    threads.emplace_back(read_thread, std::vector<std::string>(begin, end));
  }

  for (auto& thread : threads) {
    thread.join();
  }

  std::cerr << "\33[2K\rTotal patches: " << results.size() << "*" << batch_size
    << " = " << results.size() * batch_size << std::endl;
  return results;
}

std::pair<std::vector<std::string>, std::vector<std::string>> ListFolder(std::string folder, float ratio,
                                     std::string suffix = ".png")
{
  std::vector<std::string> train, val;
  for (auto &entry : fs::directory_iterator(folder)) {
    std::string name = entry.path().string();
    if (name.substr(name.length() - suffix.length()) == suffix) {
      train.push_back(name);
    }
  }
  size_t count = train.size();
  size_t train_count = count * ratio;
  size_t val_count = count - train_count;
  val = std::vector<std::string>(train.begin(), train.begin() + val_count);
  train.erase(train.begin(), train.begin() + val_count);
  return { train, val };
}

constexpr int level = 2;
const int depth = 22;
const int batch_size = 48;
const int max_epoch = 100;
const int patch_size = 16;
const float learning_rate = 0.0001;
const float weight_decay = 1e-4;
int num_train_threads;

std::mutex param_init_mutex, push_mutex;
std::condition_variable param_init_cv;

void train_thread(int id, Context ctx,
    std::vector<std::array<NDArray, level+1>>::const_iterator train_begin,
    std::vector<std::array<NDArray, level+1>>::const_iterator train_end,
    const std::vector<std::array<NDArray, level+1>> &val_data,
    int argc, char* argv[])
{
  auto model = Modelx4(patch_size, depth);
  std::map<std::string, NDArray> args_map;
  args_map["data"] = NDArray(Shape(batch_size, 1, patch_size, patch_size), ctx);
  for (int i = 0; i < level; ++i) {
    int scale = 2 << i;
    args_map["data_label_x" + std::to_string(scale)] =
      NDArray(Shape(batch_size, 1, patch_size * scale, patch_size * scale), ctx);
  }

  model.InferArgsMap(ctx, &args_map, args_map);

  std::string param_str = "d" + std::to_string(depth) + "p" + std::to_string(patch_size);


  int start_iter = argc == 1 ? 0 : std::stoi(argv[1]) + 1;

  if (id == 0) {
    model.Save("lapsrn." + param_str + ".model");
    std::cerr << "Model saved" << std::endl;

    if (argc == 1) {
      LG << "Training from scratch";
      auto initializer = Normal(0, sqrt(2.0 / (9.0 * patch_size * patch_size)));
      for (auto &arg : args_map) {
        if (arg.first.find('w') != std::string::npos) {
          initializer(arg.first, &arg.second);
        }
        if (arg.first.find('b') != std::string::npos) {
          Zero()(arg.first, &arg.second);
        }
      }
    } else {
      std::string param_name = "lapsrn" + param_str + "_param_" + argv[1];
      LG << "Loading " << param_name;
      NDArray::Load(param_name, nullptr, &args_map);
    }
  }

  auto *exec = model.SimpleBind(ctx, args_map);
  std::vector<int> arg_keys(exec->arg_arrays.size());
  std::iota(arg_keys.begin(), arg_keys.end(), 0);

  if (id == 0) {
    KVStore::Init(arg_keys, exec->arg_arrays);
    std::unique_ptr<Optimizer> opt(OptimizerRegistry::Find("adam"));
    opt->SetParam("rescale_grad", 1.0 / batch_size / num_train_threads);
    opt->SetParam("lr", learning_rate);
    opt->SetParam("wd", weight_decay);
    KVStore::SetOptimizer(std::move(opt), true);
    
    LG << "Parameters inited";
    param_init_cv.notify_all();
  } else {
    std::unique_lock<std::mutex> param_init_lock(param_init_mutex);
    param_init_cv.wait(param_init_lock);
  }

  Monitor mon(1, std::regex("conv.*_output|conv.*_w"));

  //mon.install(exec);

  LG << "Thread #" << id << " starts training";
  for (int iter = start_iter; iter < max_epoch; ++iter) {
    LG << "Thread #" << id << " Epoch: " << iter;
    auto tic = std::chrono::system_clock::now();
    int samples = 0;

    PSNR train_psnr;
    for (auto it = train_begin; it != train_end; ++it) {
      auto& data = *it;
      //mon.tic();
      KVStore::Pull(arg_keys, &exec->arg_arrays);
      samples += batch_size;
      data.front().CopyTo(&args_map["data"]);
      for (int i = 1; i <= level; ++i) {
        data[i].CopyTo(&args_map["data_label_x" + std::to_string(1<<i)]);
      }
      NDArray::WaitAll();

      exec->Forward(true);
      exec->Backward();
      {
        std::lock_guard<std::mutex> guard(push_mutex);
        KVStore::Push(arg_keys, exec->grad_arrays);
      }
      //exec->UpdateAll(opt, learning_rate, weight_decay);
      train_psnr.Update(data[1], exec->outputs[0]);
      //mon.toc_print();

      if (samples % (100 * batch_size) == 0) {
        LG << "Thread #" << id <<
          " Epoch:\t" << iter << " : " << samples << " PSNR: " << train_psnr.Get();
      }
    }

    auto toc = std::chrono::system_clock::now();

    if (id == 0) {
      LG << "Validating...";
      PSNR acu, data_ac;
      for (auto &data : val_data) {
        data.front().CopyTo(&args_map["data"]);
        for (int i = 1; i <= level; ++i) {
          data[i].CopyTo(&args_map["data_label_x" + std::to_string(1<<i)]);
        }
        NDArray::WaitAll();

        exec->Forward(false);
        NDArray::WaitAll();
        acu.Update(data[1], exec->outputs[0]);
        //data_ac.Update(data.front, data.second);
      }

      std::string save_path_param = "./lapsrn_" + param_str + "_param_" + std::to_string(iter);
      auto save_args = args_map;
      save_args.erase(save_args.find("data"));
      for (int i = 1; i <= level; ++i) {
        save_args.erase(save_args.find("data_label_x" + std::to_string(1<<i)));
      }
      LG << "ITER:\t" << iter << " Saving to..." << save_path_param;
      NDArray::Save(save_path_param, save_args);

      float duration = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count() / 1000.0;
      LG << "Epoch:\t" << iter << " "
        << samples / duration * patch_size * patch_size
        << " pixel/sec in "
        << duration << "s PSNR: " << acu.Get() ;//<< "/" << data_ac.Get();
    }
  }
}

int main(int argc, char *argv[])
{
  const std::vector<Context> ctxs{
    Context::gpu(0), Context::gpu(1), Context::gpu(2), Context::gpu(3)};

  std::vector<std::string> train_paths, val_paths;
  std::tie(train_paths, val_paths) = ListFolder("/home/data/xlidc/png", 0.9);

  auto train_data = ReadImages<level>(train_paths, patch_size, batch_size);
  auto val_data = ReadImages<level>(val_paths, patch_size, batch_size);

  KVStore::SetType("local");

  std::vector<std::thread> threads;
  num_train_threads = ctxs.size();
  for (size_t i = 0; i < ctxs.size(); ++i) {
    int piece = (train_data.size() + ctxs.size() - 1) / ctxs.size();
    auto begin = train_data.cbegin() + i * piece;
    auto end = (i + 1 == ctxs.size()) ? train_data.cend() : begin + piece;
    threads.emplace_back(train_thread, threads.size(), ctxs[i],
        begin, end, val_data, argc, argv);
  }

  for (auto& thread : threads) {
    thread.join();
  }

  MXNotifyShutdown();
  return 0;
}
