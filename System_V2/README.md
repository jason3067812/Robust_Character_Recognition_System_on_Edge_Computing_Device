# UI and recognition

本功能有兩個模組（兩個資料夾）：

- recognition_service: 為基於 Tensorflow 的辨識服務 ，請先設置完成，並運行。
- cam_ui: 為相機讀取與顯示結果的 UI 界面。

兩個模組透過 gRPC 傳送圖片與辨識結果。
