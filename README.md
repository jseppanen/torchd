
# `torchd` – call TorchScript models via HTTP

Simple C++ server to call `forward` on your TorchScript models from anywhere that can send HTTP requests.

## Installation

First, download and extract the [libtorch](https://pytorch.org/get-started/locally/) zip file.

Then, compile `torchd`:
```
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$PWD/../libtorch ..
make
```

## Usage example

Get a TorchScript model, e.g., [traced_bert.pt](https://huggingface.co/docs/transformers/master/en/serialization#torchscript)

Start `torchd` server
```
torchd --model traced_bert.pt
```

Make an HTTP request
```
curl -H 'Content-Type: application/json' --data '{"inputs": [[[101, 2040, 2001, 3958, 27227, 1029, 102, 3958, 103, 2001, 1037, 13997, 11510, 102]], [[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]]]}' http://127.0.0.1:7777/forward
```

Stop `torchd` server
```
kill -TERM <pid>
```

Show usage
```
$ torchd --help
Usage: torchd [OPTIONS]

Options
  --model PATH     Load TorchScript model from PATH (required)
  --device STRING  Run model on device (default cpu)
  --host STRING    Bind at host (default 127.0.0.1)
  --port INTEGER   Listen at port (default 7777)
  --pidfile PATH   Print PID to file
  --help           Show this message and exit
```

## Acknowledgements

Built with the excellent [libtorch](https://github.com/pytorch/pytorch), [cpp-httplib](https://github.com/yhirose/cpp-httplib), [simdjson](https://github.com/simdjson/simdjson), and [signal-wrangler](https://github.com/thomastrapp/signal-wrangler) libraries.
