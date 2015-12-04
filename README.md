# CUDA-GMM

About:
Gaussian Mixture Model accelerated by CUDA

This is a simple example on how to use cuda to accelerate the GMM. You can use this program to extract the moving objects from webcam. I only use opencv3.0 for reading frames from camera and showing the image filtered by my GMM. That's all i use the extra library. It's easy to understand the algorithm in this example. 

這是一個狠簡單的例子，CUDA加速高斯混合模型可以濾出攝影機中會移動的物件，我只使用opencv3.0讀攝影機及顯示照片的結果，可以從這個例子清楚了解演算法的實作。

------------------------------------------
Compiling the CUDA Code (Linux)    
> $ make

My compiler:
> nvcc: NVIDIA (R) Cuda compiler driver Cuda compilation tools, release 7.5, V7.5.17

------------------------------------------
Running the Code (Linux)
> $ ./gmmWithcuda

------------------------------------------
License

The MIT License (MIT)

Copyright (c) <2015> CHIH-TE LU

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
