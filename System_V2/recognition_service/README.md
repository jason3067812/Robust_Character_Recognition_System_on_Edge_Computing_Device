# Recognition Service

## Environment

- Python: 3.6
- Install according to the packages listed in `requirements.txt`.

## Start the service

To start the service, open a terminal in the same folder as `README.md`. And set `PYTHONPATH` with the following command.

```bash
export PYTHONPATH=$(pwd)
```
Then if you use Linux based on **ARCH64** CPU, run the following commands to set the library required by OPENCV.

```bash
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```
Then start the service and run the following command.

```bash
python3 app/grpc_server.py
```
The output `recognition service started.` indicates that the service has been started and can receive requests. A practical example is as follows:

```bash
2021-03-25 00:45:15.557939: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-03-25 00:45:15.557978: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge...
2021-03-25 00:45:15.557988: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      
2021-03-25 00:45:21,018 INFO: recognition service started.
```

## Testing function

- `test/test_recognize_one_image.py`: Enable to test the recognition process of a single image.
- `test/test_grpc_server.py`: You can read in images and use grpc client to test grpc server calls.
