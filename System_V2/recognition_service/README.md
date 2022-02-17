# Recognition Service

## 架設環境

Python 環境使用 3.6。

依照 `requirements.txt` 所列出的軟體包安裝即可。

## 啟動服務

如要啟動服務，請在 `README.md` 的同級資料夾開啟終端機。並使用以下指令設定 `PYTHONPATH`。

```bash
export PYTHONPATH=$(pwd)
```

接著如果使用基於 **ARCH64** CPU 的 Linux，要多運行以下指令設定 OPENCV 所需 Library。

```bash
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```

接著啟動服務運行以下指令。

```bash
python3 app/grpc_server.py
```

看到輸出有顯示 `recognition service started.` 字樣代表服務已經啟動完成，可以接收請求了。實際範例如下：

```bash
2021-03-25 00:45:15.557939: I tensorflow/compiler/jit/xla_gpu_device.cc:99] Not creating XLA devices, tf_xla_enable_xla_devices not set
2021-03-25 00:45:15.557978: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1261] Device interconnect StreamExecutor with strength 1 edge...
2021-03-25 00:45:15.557988: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1267]      
2021-03-25 00:45:21,018 INFO: recognition service started.
```

## 測試功能

- `test/test_recognize_one_image.py`: 可以測試單張影像的辨識流程。
- `test/test_grpc_server.py`: 可以讀入影像並利用 grpc client 測試 grpc server 的調用。
