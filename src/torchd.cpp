#include <torch/script.h>
#include <torch/torch.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "SignalHandler.h"
#include "httplib.h"
#include "simdjson.h"

const int MAX_INPUT_SIZE = 100 * 1024 * 1024; // 100 MiB

using namespace httplib;
using namespace simdjson;
using namespace torch::jit;

void log(const std::string &msg) {
  const auto time =
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  auto pid = getpid();
  auto tid = std::this_thread::get_id();
  // XXX locking
  std::cout << std::put_time(std::localtime(&time), "%F %T") << " torchd["
            << pid << " " << tid << "] " << msg << std::endl
            << std::flush;
}

inline torch::Tensor parse_json_tensor(ondemand::array array) {
  size_t size = array.count_elements();
  ondemand::array_iterator elem_it = array.begin();
  ondemand::value first_elem = *elem_it;
  if (first_elem.type() == ondemand::json_type::array) {
    // handle nested arrays recursively
    std::vector<torch::Tensor> rows;
    rows.emplace_back(parse_json_tensor(first_elem.get_array()));
    ++elem_it;
    for (std::size_t index = 1; index < size; ++index) {
      rows.emplace_back(parse_json_tensor((*elem_it).get_array()));
      ++elem_it;
    }
    return torch::stack(rows, 0);
  } else if (first_elem.type() == ondemand::json_type::number) {
    ondemand::number first_number = first_elem.get_number();
    ++elem_it;
    torch::Tensor tensor;
    // infer tensor data type from first element
    if (first_number.is_double()) {
      tensor = torch::empty(size, torch::dtype(torch::kFloat32));
      auto accessor = tensor.accessor<float, 1>();
      accessor[0] = first_number.get_double();
      for (std::size_t index = 1; index < size; ++index) {
        accessor[index] = (*elem_it).get_double();
        ++elem_it;
      }
    } else if (first_number.is_int64()) {
      tensor = torch::empty(size, torch::dtype(torch::kInt64));
      auto accessor = tensor.accessor<int64_t, 1>();
      accessor[0] = first_number.get_int64();
      for (std::size_t index = 1; index < size; ++index) {
        accessor[index] = (*elem_it).get_int64();
        ++elem_it;
      }
    } else {
      std::stringstream buf;
      buf << first_number.get_number_type();
      throw std::runtime_error(
          "tensors must contain float or int values, but got " + buf.str());
    }

    return tensor;
  } else {
    std::stringstream buf;
    buf << first_elem.type();
    throw std::runtime_error(
        "tensors must contain float or int values, but got " + buf.str());
  }
}

std::vector<IValue> parse_json_inputs(const std::string &json_input,
                                      ondemand::parser &parser) {
  ondemand::document doc = parser.iterate(json_input);
  std::vector<IValue> inputs;

  try {
    ondemand::array inputs_array = doc.find_field("inputs").get_array();
    for (ondemand::value input : inputs_array) {
      torch::Tensor tensor = parse_json_tensor(input);
      inputs.emplace_back(tensor);
    }
    return inputs;
  } catch (simdjson_error &err) {
    std::string location(doc.current_location());
    std::string msg = (std::string("JSON parse error: ") + err.what() +
                       " near '" + location.substr(0, 20) + "'");
    throw std::runtime_error(msg);
  }
}

inline void dump_json_double(double value, std::string &json_output) {
  std::ostringstream buf;
  buf << std::setprecision(6) << value; // 6 digits is enough for float32
  json_output.append(buf.str());
}

inline void dump_json_tensor(const torch::Tensor &tensor,
                             std::string &json_output) {

  auto options = tensor.options();

  if (tensor.ndimension() == 0) {
    // tensor.item();
    throw std::runtime_error("not implemented");
  } else if (tensor.ndimension() == 1) {
    json_output.append("[");
    if (tensor.dtype() == torch::kFloat32) {
      auto accessor = tensor.accessor<float, 1>();
      bool comma = false;
      for (int i = 0; i < tensor.size(0); ++i) {
        if (comma) {
          json_output.append(",");
        }
        dump_json_double(accessor[i], json_output);
        comma = true;
      }
    } else if (tensor.dtype() == torch::kInt64) {
      auto accessor = tensor.accessor<int64_t, 1>();
      bool comma = false;
      for (int i = 0; i < tensor.size(0); ++i) {
        if (comma) {
          json_output.append(",");
        }
        json_output.append(std::to_string(accessor[i]));
        comma = true;
      }
    } else {
      std::ostringstream buf;
      buf << tensor.dtype();
      throw std::runtime_error(
          "tensor dtype is not supported (expecting float32 or int64): " +
          buf.str());
    }
    json_output.append("]");
  } else {
    bool comma = false;
    json_output.append("[");
    for (int i = 0; i < tensor.size(0); ++i) {
      if (comma) {
        json_output.append(",");
      }
      dump_json_tensor(tensor[i], json_output);
      comma = true;
    }
    json_output.append("]");
  }
}

inline void dump_json_ivalue(const IValue &value, std::string &json_output) {
  if (value.isTuple()) {
    auto tuple = value.toTuple();
    json_output.append("[");
    bool comma = false;
    for (const IValue &elem : tuple->elements()) {
      if (comma) {
        json_output.append(",");
      }
      dump_json_ivalue(elem, json_output);
      comma = true;
    }
    json_output.append("]");
  } else if (value.isTensor()) {
    dump_json_tensor(value.toTensor(), json_output);
  } else if (value.isDouble()) {
    dump_json_double(value.toDouble(), json_output);
  } else if (value.isInt()) {
    json_output.append(std::to_string(value.toInt()));
  }
  /*
  else if (value.isScalar()) {
    torch::Scalar scalar = value.toScalar();
  }
  */
  else {
    throw std::runtime_error(
        "value is not supported (expecting scalar, tensor, or tuple): " +
        value.tagKind());
  }
}

std::string forward(script::Module &model, const c10::Device &device,
                    std::mutex &device_mutex, const std::string &json_input) {
  ondemand::parser parser;
  std::string padded_input(json_input);
  padded_input.reserve(padded_input.size() + SIMDJSON_PADDING);
  std::vector<IValue> inputs = parse_json_inputs(padded_input, parser);
  IValue output;

  {
    // exclusive GPU access block
    std::lock_guard<std::mutex> guard(device_mutex);

    /* getting:
    RuntimeError: Inference tensors cannot be saved for backward. To work around
    you can make a clone to get a normal tensor and use it in autograd.
    https://github.com/pytorch/pytorch/issues/60333
    */
    // c10::InferenceMode guard;
    torch::NoGradGuard no_grad;

    // move inputs to GPU if needed
    for (auto iter = inputs.begin(); iter != inputs.end(); ++iter) {
      *iter = (*iter).toTensor().to(device);
    }

    try {
      output = model.forward(inputs);
    } catch (const c10::Error &error) {
      std::ofstream fin("/tmp/torchd_inputs.json");
      fin << json_input;
      fin.close();
      std::ofstream ferr("/tmp/torchd_error.txt");
      ferr << error.what();
      ferr.close();
      log("ERROR forward failed, see /tmp/torchd_inputs.json and "
          "/tmp/torchd_error.txt");
      throw;
    }

    // move output(s) back to CPU if needed
    if (output.isTuple()) {
      auto tuple = output.toTuple();
      std::vector<IValue> cpu_tensors;
      for (auto elem : tuple->elements()) {
        cpu_tensors.push_back(IValue(elem.toTensor().to(torch::kCPU)));
      }
      output = torch::ivalue::Tuple::create(cpu_tensors);
    } else if (output.isTensor()) {
      output = IValue(output.toTensor().to(torch::kCPU));
    } else {
      throw std::runtime_error(
          "invalid result type, expected tensor or tuple of tensors, but got " +
          output.tagKind());
    }
  }

  std::string json_output;
  json_output.append("{\"output\":");
  dump_json_ivalue(output, json_output);
  json_output.append("}\n");
  return json_output;
}

class Args {
public:
  Args() {}
  virtual ~Args() {}

  std::string model_path;
  std::string device_name;
  std::string host;
  int port;
  std::string pidfile_path;

  static bool parse_flag(int argc, char **argv, const std::string &option) {
    char **end = argv + argc;
    return std::find(argv, end, option) != end;
  }

  static std::string parse_option(int argc, char **argv,
                                  const std::string &option,
                                  const std::string &default_) {
    char **end = argv + argc;
    char **match = std::find(argv, end, option);
    if (match != end) {
      match++;
      if (match != end) {
        return *match;
      } else {
        return "";
      }
    }
    return default_;
  }

  void parse(int argc, char *argv[]) {
    model_path = parse_option(argc, argv, "--model", "");
    device_name = parse_option(argc, argv, "--device", "cpu");
    host = parse_option(argc, argv, "--host", "127.0.0.1");
    std::string port_string = parse_option(argc, argv, "--port", "7000");
    pidfile_path = parse_option(argc, argv, "--pidfile", "");
    try {
      port = std::stoi(port_string);
    } catch (std::exception &error) {
      port = -1;
    }
    bool print_help =
        parse_flag(argc, argv, "-h") || parse_flag(argc, argv, "--help");

    if (print_help) {
      std::cerr
          << "Usage: torchd [OPTIONS]\n\n"
          << "Options\n"
          << "  --model PATH     Load TorchScript model from PATH (required)\n"
          << "  --device STRING  Run model on device (default cpu)\n"
          << "  --host STRING    Bind at host (default 127.0.0.1)\n"
          << "  --port INTEGER   Listen at port (default 7000)\n"
          << "  --pidfile PATH   Print PID to file\n"
          << "  --help           Show this message and exit\n";
      exit(1);
    }

    if (model_path == "") {
      std::cerr << "Error: --model is required\n";
      exit(1);
    }
    if (port <= 0) {
      std::cerr << "Error: --port is not an integer\n";
      exit(1);
    }
  }
};

int main(int argc, char *argv[]) {

  Args args;
  args.parse(argc, argv);

  // Block signals in this thread.
  // Threads spawned later will inherit the signal mask.
  sgnl::SignalHandler signal_handler({SIGINT, SIGTERM});

  try {
    c10::Device d(args.device_name);
  } catch (const c10::Error &error) {
    log("ERROR Unknown device: \"" + args.device_name + "\": " + error.what());
    return EXIT_FAILURE;
  }

  script::Module model;
  c10::Device device(args.device_name);
  std::mutex device_mutex;

  try {
    model = torch::jit::load(args.model_path.c_str(), device);
    model.to(device);
  } catch (const c10::Error &error) {
    log("ERROR Failed to load model: \"" + args.model_path +
        "\": " + error.what());
    return EXIT_FAILURE;
  }

  Server svr;
  svr.set_payload_max_length(MAX_INPUT_SIZE);

  svr.Get("/", [](const Request & /*req*/, Response &res) {
    res.set_content("<html><h1>torchd ready</h1></html>\n", "text/html");
  });

  svr.Get("/ping", [](const Request & /*req*/, Response &res) {
    res.set_content("", "text/plain");
  });

  svr.Post("/forward", [&](const Request &req, Response &res) {
    std::string content_type = req.get_header_value("Content-Type");
    if (content_type != "application/json") {
      throw std::runtime_error(
          "wrong Content-Type, expecting \"application/json\", but got \"" +
          content_type + "\"");
    }
    std::string result = forward(model, device, device_mutex, req.body);
    res.set_content(result, "application/json");
  });

  svr.set_error_handler([](const Request &req, Response &res) {
    std::string json =
        "{\"error\": {\"status\": " + std::to_string(res.status) + "}}\n";
    res.set_content(json, "application/json");
  });

  svr.set_exception_handler(
      [](const Request &req, Response &res, std::exception &error) {
        log(std::string("ERROR ") + error.what());
        res.status = 500;
      });

  if (args.pidfile_path != "") {
    std::ofstream fpid(args.pidfile_path);
    fpid << getpid() << std::endl;
    fpid.close();
  }

  auto fut_stop_signal =
      std::async(std::launch::async, &sgnl::SignalHandler::sigwait_handler,
                 &signal_handler, [&svr](int /*signum*/) {
                   svr.stop();
                   return true;
                 });

  log("INFO listening at http://" + args.host + ":" +
      std::to_string(args.port));
  svr.listen(args.host.c_str(), args.port);
  fut_stop_signal.get();
  log("INFO exiting");
  return EXIT_SUCCESS;
}
