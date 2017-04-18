#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <iostream>

#define CUDA_CALL(x) do { cudaError_t err = x; if (( err ) != cudaSuccess ){ \
printf ("Error \"%s\" at %s :%d \n" , cudaGetErrorString(err), \
__FILE__ , __LINE__ ) ; return err;\
}} while (0);

void deBoorMakeTridiag(std::vector<float> x, std::vector<float> y, float d0, float dn, std::vector<float> &a, std::vector<float> &b, std::vector<float> &c, std::vector<float> &r)
{
	std::vector<float> dX(x.size() - 1);
	std::vector<float> dY(y.size() - 1);
	for (int i = 0; i < dX.size(); i++)
	{
		dX[i] = x[i + 1] - x[i];
		dY[i] = y[i + 1] - y[i];
	}
	for (int i = 0; i < a.size(); i++)
	{
		a[i] = dX[i + 1];
		b[i] = 2 * (dX[i] + dX[i + 1]);
		c[i] = dX[i];
		r[i] = 3 * ((dX[i] / dX[i + 1]) * dY[i + 1] + (dX[i + 1] / dX[i]) * dY[i]);
	}
	r[0] -= a[0] * d0;
	r[r.size() - 1] -= c[c.size() - 1] * dn;
}

void deBoorMakeEqui(std::vector<float> x, std::vector<float> y, float d0, float dn, std::vector<float> &a, std::vector<float> &b, std::vector<float> &c, std::vector<float> &r)
{
	float h = 3 / (x[1] - x[0]);
	for (int i = 0; i < r.size(); i++)
	{
		a[i] = 1;
		b[i] = 4;
		c[i] = 1;
		r[i] = h * (y[i + 2] - y[i]);
	}
	r[0] -= d0;
	r[r.size() - 1] -= dn;
}

//__global__ void LU_tridiag(float* a, float* b, float* c, float* r, int from, int to)
//{
//	for (int i = from + 1; i < to; i++)
//	{
//		a[i] = a[i] / b[i - 1];
//		b[i] = b[i] - (a[i] * c[i - 1]);
//		r[i] = r[i] - (a[i] * r[i - 1]);
//	}
//	r[to - 1] = r[to - 1] / b[to - 1];
//	for (int i = to - 2; i >= from; i--)
//	{
//		r[i] = (r[i] - (c[i] * r[i + 1])) / b[i];
//	}
//}

__device__ void LU_tridiag_device(float* a, float* b, float* c, float* r, int from, int to)
{
	for (int i = from + 1; i < to; i++)
	{
		a[i] = a[i] / b[i - 1];
		b[i] = b[i] - (a[i] * c[i - 1]);
		r[i] = r[i] - (a[i] * r[i - 1]);
	}
	r[to - 1] = r[to - 1] / b[to - 1];
	for (int i = to - 2; i >= from; i--)
	{
		r[i] = (r[i] - (c[i] * r[i + 1])) / b[i];
	}
}

// ================================================================================================================================================================ LU DEVICE EQUIDISTANCNE
__device__ void LU_tridiag_dev_equi(float* a, float* b, float* r, int from, int to)
{
	// float *a = new float[to - from];
	// float *b = new float[to - from];
	b[from] = -14;
	for (int i = from + 1; i < to; i++)
	{
		a[i] = 1 / b[i - 1];
		r[i] = r[i] - (a[i] * r[i - 1]);
		b[i] = -14 - a[i];
	}
	r[to - 1] = r[to - 1] / b[to - 1];
	for (int i = to - 2, j = to - from - 2; i >= from; i--, j--)
	{
		r[i] = (r[i] - r[i + 1]) / b[i];
	}
	// delete[] a;
	// delete[] b;
}

void LU_CPU_equi(std::vector<float> &r, int from, int to, bool even)
{
	std::vector<float> a(to - from);
	std::vector<float> b(to - from);
	b[0] = -14;
	for (int i = from + 1, j = 1; i < to; i++, j++)
	{
		a[j] = 1 / b[j - 1];
		r[i] = r[i] - (a[j] * r[i - 1]);
		if (i == to - 1 && even)
		{
			b[j] = -15 - a[j];
		}
		else
		{
			b[j] = -14 - a[j];
		}
	}
	r[to - 1] = r[to - 1] / b[to - from - 1];
	for (int i = to - 2, j = to - from - 2; i >= from; i--, j--)
	{
		r[i] = (r[i] - r[i + 1]) / b[j];
	}
}

void LU_CPU(std::vector<float> a, std::vector<float> b, std::vector<float> c, std::vector<float> &r, int from, int to)
{
	for (int i = from + 1; i < to; i++)
	{
		a[i] = a[i] / b[i - 1];
		b[i] = b[i] - (a[i] * c[i - 1]);
		r[i] = r[i] - (a[i] * r[i - 1]);
	}
	r[to - 1] = r[to - 1] / b[to - 1];
	for (int i = to - 2; i >= from; i--)
	{
		r[i] = (r[i] - (c[i] * r[i + 1])) / b[i];
	}
}

__global__ void partitioning(float* a, float* b, float* c, float* r, float* Va, float* Vb, float* Vc, float* Vr, int* Vindex, int pLength, int Vsize, int remainder) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int i = idx * 2 + 1;
	int myLength = pLength;
	int j = idx * myLength;
	if (i == Vsize - 1) // if this is the last processor
	{
		myLength += remainder;
	}
	if (i < Vsize)
	{
		/*float *dev_a = new float[myLength];
		float *dev_b = new float[myLength];
		float *dev_c = new float[myLength];
		float *dev_r = new float[myLength];
		for (int k = 0; k < myLength; k++)
		{
			dev_a[k] = a[j + k];
			dev_b[k] = b[j + k];
			dev_c[k] = c[j + k];
			dev_r[k] = r[j + k];
		}*/
		/*float Vai = dev_a[1];
		float Vbi = dev_b[1];
		float Vci = dev_c[1];
		float Vri = dev_r[1];*/
		float Vai = a[j + 1];
		float Vbi = b[j + 1];
		float Vci = c[j + 1];
		float Vri = r[j + 1];
		Vindex[i - 1] = j;
		Vindex[i] = j + myLength - 1;
		for (int k = 2; k < myLength; k++) /* && j < size*/
		{
			float alpha = a[j + k] / Vbi;
			Vai = -Vai * alpha;
			Vbi = b[j + k] - alpha * Vci;
			Vci = c[j + k];
			Vri = r[j + k] - alpha * Vri;
			// float alpha = Vbi / a[j + k];
			// Vri -= alpha * r[j + k];
			// Vbi = Vci - alpha * b[j + k];
			// Vci = -alpha * c[j + k];
		}
		Va[i] = Vai;
		Vb[i] = Vbi;
		Vc[i] = Vci;
		Vr[i] = Vri;
		i--;
		Vai = a[j + myLength - 2];
		Vbi = b[j + myLength - 2];
		Vci = c[j + myLength - 2];
		Vri = r[j + myLength - 2];
		for (int k = myLength - 3; k >= 0; k--)
		{
			float beta = c[j + k] / Vbi;
			Vri = r[j + k] - beta * Vri;
			Vbi = b[j + k] - beta * Vai;
			Vai = a[j + k];
			Vci = -Vci * beta;
		}
		Va[i] = Vai;
		Vb[i] = Vbi;
		Vc[i] = Vci;
		Vr[i] = Vri;
		/*delete[] dev_a;
		delete[] dev_b;
		delete[] dev_c;
		delete[] dev_r;*/
	}
}

// =============================================================================================================================================================================REDUCED PARTITIONING
__global__ void partitioning_reduced(float* r, float* Va, float* Vb, float* Vc, float* Vr, int* Vindex, int pLength, int Vsize, int remainder, bool even) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int i = idx * 2 + 1;
	int myLength = pLength;
	int j = idx * myLength;
	if (i == Vsize - 1) // if this is the last processor
	{
		myLength += remainder;
	}
	even = even && (i == Vsize - 1);
	if (i < Vsize)
	{
		float Vai = 1;
		float Vbi = -14;
		float Vri = r[j + 1];
		Vindex[i - 1] = j;
		Vindex[i] = j + myLength - 1;
		for (int k = 2; k < myLength; k++)
		{
			float alpha = 1 / Vbi;
			Vai = -Vai * alpha;
			if (k == myLength - 1 && even)
			{
				Vbi = -15 - alpha;
			}
			else
			{
				Vbi = -14 - alpha;
			}
			Vri = r[j + k] - alpha * Vri;
		}
		Va[i] = Vai;
		Vb[i] = Vbi;
		Vc[i] = 1;
		Vr[i] = Vri;
		i--;
		Vbi = -14;
		float Vci = 1;
		Vri = r[j + myLength - 2];
		for (int k = myLength - 3; k >= 0; k--)
		{
			float beta = 1 / Vbi;
			Vri = r[j + k] - beta * Vri;
			Vbi = -14 - beta;
			Vci = -Vci * beta;
		}
		Va[i] = 1;
		Vb[i] = Vbi;
		Vc[i] = Vci;
		Vr[i] = Vri;
	}
}

// =============================================================================================================================================================================REDUCED FINAL
__global__ void final_computations_reduced(float* a, float* b, float* r, float* Vr, int* Vindex, int Vsize)
{
	int i = (blockDim.x * blockIdx.x + threadIdx.x) * 2;
	if (i < Vsize)
	{
		/*int Vind = Vindex[i];
		int Vind1 = Vindex[i + 1];
		float Vri = Vr[i];
		float Vri1 = Vr[i + 1];*/

		r[Vindex[i]] = Vr[i];
		r[Vindex[i + 1]] = Vr[i + 1];

		int idx1 = Vindex[i] + 1;
		r[idx1] -= Vr[i];

		int idx2 = Vindex[i + 1] - 1;
		r[idx2] -= Vr[i + 1];

		LU_tridiag_dev_equi(a, b, r, idx1, idx2 + 1);
	}
}

__global__ void final_computations(float* a, float* b, float* c, float* r, float* Vr, int* Vindex, int Vsize)
{
	int i = (blockDim.x * blockIdx.x + threadIdx.x) * 2;
	if (i < Vsize)
	{
		/*int Vind = Vindex[i];
		int Vind1 = Vindex[i + 1];
		float Vri = Vr[i];
		float Vri1 = Vr[i + 1];*/

		r[Vindex[i]] = Vr[i];
		r[Vindex[i + 1]] = Vr[i + 1];

		int idx1 = Vindex[i] + 1;
		r[idx1] -= a[idx1] * Vr[i];

		int idx2 = Vindex[i + 1] - 1;
		r[idx2] -= c[idx2] * Vr[i + 1];

		LU_tridiag_device(a, b, c, r, idx1, idx2 + 1);
	}
}

// ============================================================================================================================================================================= GPU REST
__global__ void rest_gpu(float* d, float* r, float* y, int Dsize, int Rsize, float d0, float dn, float h) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < Rsize)
	{
		int idx = (i + 1) * 2;
		float Didx_prev = (i > 0) ? r[i - 1] : d0 ;
		float Didx_next = (idx < Dsize - 1) ? r[i] : dn;
		float mu1 = 3 / h, dp4 = 0.25;
		if (i == 0)
		{
			d[0] = d0;
		}
		d[idx] = Didx_next;
		d[idx - 1] = (mu1 * (y[idx] - y[idx - 2]) - Didx_prev - Didx_next) * dp4;
	}
}

std::vector<float> compute_rest(std::vector<float> r, std::vector<float> y, float d0, float dn, float h) {
	int size = y.size();
	std::vector<float> d(size);
	d[0] = d0;
	d[size - 1] = dn;
	float mu1 = 3 / h, dp4 = 0.25;
	for (int i = 1, k = 0; i < size - 3; i += 2, k++)
	{
		d[i + 1] = r[k];
		d[i] = (mu1 * (y[i + 1] - y[i - 1]) - d[i - 1] - d[i + 1]) * dp4;
	}
	d[size - 2] = (mu1 * (y[size - 1] - y[size - 3]) - d[size - 3] - d[size - 1]) * dp4;
	return d;
}

cudaError_t austin_berndt_moulton(std::vector<float> a, std::vector<float> b, std::vector<float> c, std::vector<float> &r, int nOfParts)
{
	int Vsize = nOfParts * 2;
	std::vector<float> Va(Vsize);
	std::vector<float> Vb(Vsize);
	std::vector<float> Vc(Vsize);
	std::vector<float> Vr(Vsize);
	std::vector<int> Vindex(Vsize);

	cudaEvent_t start, stop_malloc, stop_memcpy1, stop_partitioning, stop_seq, stop_final, stop_memcpy_final;
	float time1 = 0.0;
	float time2 = 0.0;
	float time3 = 0.0;
	float time4 = 0.0;
	float time5 = 0.0;
	float time6 = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop_malloc);
	cudaEventCreate(&stop_memcpy1);
	cudaEventCreate(&stop_partitioning);
	cudaEventCreate(&stop_seq);
	cudaEventCreate(&stop_final);
	cudaEventCreate(&stop_memcpy_final);

	cudaEventRecord(start);
	cudaEventSynchronize(start);

	float *dev_a = 0; CUDA_CALL(cudaMalloc((void**)&dev_a, a.size() * sizeof(float)));
	float *dev_b = 0; CUDA_CALL(cudaMalloc((void**)&dev_b, b.size() * sizeof(float)));
	float *dev_c = 0; CUDA_CALL(cudaMalloc((void**)&dev_c, c.size() * sizeof(float)));
	float *dev_r = 0; CUDA_CALL(cudaMalloc((void**)&dev_r, r.size() * sizeof(float)));
	float *dev_Va = 0; CUDA_CALL(cudaMalloc((void**)&dev_Va, Vsize * sizeof(float)));
	float *dev_Vb = 0; CUDA_CALL(cudaMalloc((void**)&dev_Vb, Vsize * sizeof(float)));
	float *dev_Vc = 0; CUDA_CALL(cudaMalloc((void**)&dev_Vc, Vsize * sizeof(float)));
	float *dev_Vr = 0; CUDA_CALL(cudaMalloc((void**)&dev_Vr, Vsize * sizeof(float)));
	int *dev_Vidx = 0; CUDA_CALL(cudaMalloc((void**)&dev_Vidx, Vsize * sizeof(int)));

	cudaEventRecord(stop_malloc);
	cudaEventSynchronize(stop_malloc);
	cudaEventElapsedTime(&time1, start, stop_malloc);

	CUDA_CALL(cudaMemcpy(dev_a, &a[0], a.size() * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_b, &b[0], b.size() * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_c, &c[0], c.size() * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_r, &r[0], r.size() * sizeof(float), cudaMemcpyHostToDevice));

	cudaEventRecord(stop_memcpy1);
	cudaEventSynchronize(stop_memcpy1);
	cudaEventElapsedTime(&time2, stop_malloc, stop_memcpy1);

	int pLength = r.size() / nOfParts;
	int remainder = r.size() - (pLength * nOfParts);
	int threadsPerBlock = 128;
	int numBlocks = (nOfParts + threadsPerBlock - 1) / threadsPerBlock;

	CUDA_CALL(cudaSetDevice(0));

	partitioning <<<numBlocks, threadsPerBlock>>> (dev_a, dev_b, dev_c, dev_r, dev_Va, dev_Vb, dev_Vc, dev_Vr, dev_Vidx, pLength, Vr.size(), remainder);
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());

	cudaEventRecord(stop_partitioning);
	cudaEventSynchronize(stop_partitioning);
	cudaEventElapsedTime(&time3, stop_memcpy1, stop_partitioning);

	CUDA_CALL(cudaMemcpy(&Va[0], dev_Va, Vsize * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(&Vb[0], dev_Vb, Vsize * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(&Vc[0], dev_Vc, Vsize * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(&Vr[0], dev_Vr, Vsize * sizeof(float), cudaMemcpyDeviceToHost));
	LU_CPU(Va, Vb, Vc, Vr, 0, Vsize);
	CUDA_CALL(cudaMemcpy(dev_Vr, &Vr[0], Vsize * sizeof(float), cudaMemcpyHostToDevice));
	/*LU_tridiag<<<1, 1>>>(dev_Va, dev_Vb, dev_Vc, dev_Vr, 0, Vsize);
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());*/

	cudaEventRecord(stop_seq);
	cudaEventSynchronize(stop_seq);
	cudaEventElapsedTime(&time4, stop_partitioning, stop_seq);

	final_computations <<<numBlocks, threadsPerBlock>>> (dev_a, dev_b, dev_c, dev_r, dev_Vr, dev_Vidx, Vr.size());
	CUDA_CALL(cudaGetLastError());
	cudaError_t err = cudaDeviceSynchronize();
	if ((err) != cudaSuccess) {
		printf("Error \"%s\" at %s :%d \n", cudaGetErrorString(err), __FILE__, __LINE__);
	}

	cudaEventRecord(stop_final);
	cudaEventSynchronize(stop_final);
	cudaEventElapsedTime(&time5, stop_seq, stop_final);

	CUDA_CALL(cudaMemcpy(&r[0], dev_r, r.size() * sizeof(float), cudaMemcpyDeviceToHost));

	cudaEventRecord(stop_memcpy_final);
	cudaEventSynchronize(stop_memcpy_final);
	cudaEventElapsedTime(&time6, stop_final, stop_memcpy_final);

	std::cout << "malloc time: " << time1 << " ms" << std::endl;
	std::cout << "memcpy time: " << time2 << " ms" << std::endl;
	std::cout << "partit time: " << time3 << " ms" << std::endl;
	std::cout << "sequen time: " << time4 << " ms" << std::endl;
	std::cout << "fiinal time: " << time5 << " ms" << std::endl;
	std::cout << "rescpy time: " << time6 << " ms" << std::endl;
	std::cout << "sum time: " << time3 + time4 + time5 << " ms" << std::endl;
	std::cout << "============================" << std::endl;

	return err;
}

// ============================================================================================================================================================================== ABM REDUCED
cudaError_t ABM_reduced(std::vector<float> &r, std::vector<float> &F, int nOfParts, float d1, float dr, float h)
{
	int Vsize = nOfParts * 2;
	std::vector<float> Va(Vsize);
	std::vector<float> Vb(Vsize);
	std::vector<float> Vc(Vsize);
	std::vector<float> Vr(Vsize);
	std::vector<float> d(F.size());
	std::vector<int> Vindex(Vsize);

	std::vector<float> r2(r.size());
	r2 = r;

	cudaEvent_t start, stop_malloc, stop_memcpy1, stop_partitioning, stop_seq, stop_final, stop_memcpy_final, stop_rest;
	float time1 = 0.0;
	float time2 = 0.0;
	float time3 = 0.0;
	float time4 = 0.0;
	float time5 = 0.0;
	float time6 = 0.0;
	float time7 = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop_malloc);
	cudaEventCreate(&stop_memcpy1);
	cudaEventCreate(&stop_partitioning);
	cudaEventCreate(&stop_seq);
	cudaEventCreate(&stop_final);
	cudaEventCreate(&stop_memcpy_final);
	cudaEventCreate(&stop_rest);

	cudaEventRecord(start);
	cudaEventSynchronize(start);

	float *dev_a = 0; CUDA_CALL(cudaMalloc((void**)&dev_a, r.size() * sizeof(float)));
	float *dev_b = 0; CUDA_CALL(cudaMalloc((void**)&dev_b, r.size() * sizeof(float)));
	float *dev_r = 0; CUDA_CALL(cudaMalloc((void**)&dev_r, r.size() * sizeof(float)));
	float *dev_d = 0; CUDA_CALL(cudaMalloc((void**)&dev_d, F.size() * sizeof(float)));
	float *dev_F = 0; CUDA_CALL(cudaMalloc((void**)&dev_F, F.size() * sizeof(float)));
	float *dev_Va = 0; CUDA_CALL(cudaMalloc((void**)&dev_Va, Vsize * sizeof(float)));
	float *dev_Vb = 0; CUDA_CALL(cudaMalloc((void**)&dev_Vb, Vsize * sizeof(float)));
	float *dev_Vc = 0; CUDA_CALL(cudaMalloc((void**)&dev_Vc, Vsize * sizeof(float)));
	float *dev_Vr = 0; CUDA_CALL(cudaMalloc((void**)&dev_Vr, Vsize * sizeof(float)));
	int *dev_Vidx = 0; CUDA_CALL(cudaMalloc((void**)&dev_Vidx, Vsize * sizeof(int)));

	cudaEventRecord(stop_malloc);
	cudaEventSynchronize(stop_malloc);
	cudaEventElapsedTime(&time1, start, stop_malloc);

	CUDA_CALL(cudaMemcpy(dev_r, &r[0], r.size() * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(dev_F, &F[0], F.size() * sizeof(float), cudaMemcpyHostToDevice));

	cudaEventRecord(stop_memcpy1);
	cudaEventSynchronize(stop_memcpy1);
	cudaEventElapsedTime(&time2, stop_malloc, stop_memcpy1);

	int pLength = r.size() / nOfParts;
	int remainder = r.size() - (pLength * nOfParts);
	int threadsPerBlock = 128;
	int numBlocks = (nOfParts + threadsPerBlock - 1) / threadsPerBlock;

	CUDA_CALL(cudaSetDevice(0));

	partitioning_reduced <<<numBlocks, threadsPerBlock >>> (dev_r, dev_Va, dev_Vb, dev_Vc, dev_Vr, dev_Vidx, pLength, Vr.size(), remainder, (F.size() % 2 == 0));
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());

	cudaEventRecord(stop_partitioning);
	cudaEventSynchronize(stop_partitioning);
	cudaEventElapsedTime(&time3, stop_memcpy1, stop_partitioning);

	CUDA_CALL(cudaMemcpy(&Va[0], dev_Va, Vsize * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(&Vb[0], dev_Vb, Vsize * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(&Vc[0], dev_Vc, Vsize * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(&Vr[0], dev_Vr, Vsize * sizeof(float), cudaMemcpyDeviceToHost));
	LU_CPU(Va, Vb, Vc, Vr, 0, Vsize);
	CUDA_CALL(cudaMemcpy(dev_Vr, &Vr[0], Vsize * sizeof(float), cudaMemcpyHostToDevice));
	/*LU_tridiag<<<1, 1>>>(dev_Va, dev_Vb, dev_Vc, dev_Vr, 0, Vsize);
	CUDA_CALL(cudaGetLastError());
	CUDA_CALL(cudaDeviceSynchronize());*/

	cudaEventRecord(stop_seq);
	cudaEventSynchronize(stop_seq);
	cudaEventElapsedTime(&time4, stop_partitioning, stop_seq);

	final_computations_reduced <<<numBlocks, threadsPerBlock >>> (dev_a, dev_b, dev_r, dev_Vr, dev_Vidx, Vr.size());
	CUDA_CALL(cudaGetLastError());
	cudaError_t err = cudaDeviceSynchronize();
	if ((err) != cudaSuccess) {
		printf("Error \"%s\" at %s :%d \n", cudaGetErrorString(err), __FILE__, __LINE__);
	}

	cudaEventRecord(stop_final);
	cudaEventSynchronize(stop_final);
	cudaEventElapsedTime(&time5, stop_seq, stop_final);

	int Rsize = (F.size() % 2 == 0) ? r.size() : r.size() + 1;
	numBlocks = (Rsize + threadsPerBlock - 1) / threadsPerBlock;
	rest_gpu <<<numBlocks, threadsPerBlock >>> (dev_d, dev_r, dev_F, F.size(), Rsize, d1, dr, h);

	cudaEventRecord(stop_rest);
	cudaEventSynchronize(stop_rest);
	cudaEventElapsedTime(&time7, stop_final, stop_rest);

	CUDA_CALL(cudaMemcpy(&d[0], dev_d, d.size() * sizeof(float), cudaMemcpyDeviceToHost));
	r = d;
	r[r.size() - 1] = dr;

	cudaEventRecord(stop_memcpy_final);
	cudaEventSynchronize(stop_memcpy_final);
	cudaEventElapsedTime(&time6, stop_rest, stop_memcpy_final);

	std::cout << "malloc time: " << time1 << " ms" << std::endl;
	std::cout << "memcpy time: " << time2 << " ms" << std::endl;
	std::cout << "partit time: " << time3 << " ms" << std::endl;
	std::cout << "sequen time: " << time4 << " ms" << std::endl;
	std::cout << "fiinal time: " << time5 << " ms" << std::endl;
	std::cout << "rest,, time: " << time7 << " ms" << std::endl;
	std::cout << "rescpy time: " << time6 << " ms" << std::endl;
	std::cout << "sum time: " << time3 + time4 + time5 + time7 << " ms" << std::endl;
	std::cout << "============================" << std::endl;

	return err;
}

void ABM_on_CPU(std::vector<float> a, std::vector<float> b, std::vector<float> c, std::vector<float> &r, int nOfParts) {
	int Vsize = nOfParts * 2;
	std::vector<float> Va(Vsize);
	std::vector<float> Vb(Vsize);
	std::vector<float> Vc(Vsize);
	std::vector<float> Vr(Vsize);
	std::vector<int> Vindex(Vsize);
	int j = 1;
	int pLength = b.size() / nOfParts;
	int remainder = b.size() - (pLength * nOfParts);
	for (int i = 0; i < Vb.size(); i += 2)
	{
		i++;
		if (i == Vb.size() - 1)
		{
			pLength += remainder;
		}
		Va[i] = a[j];
		Vb[i] = b[j];
		Vc[i] = c[j];
		Vr[i] = r[j];
		Vindex[i - 1] = j - 1;
		int jInit = j - 1;
		j++;
		for (int k = 0; k < pLength - 2 && j < b.size(); k++, j++)
		{
			float alpha = Vb[i] / a[j];
			Vr[i] -= alpha * r[j];
			Vb[i] = Vc[i] - alpha * b[j];
			Vc[i] = -alpha * c[j];
		}
		i--;
		Va[i] = a[j - 2];
		Vb[i] = b[j - 2];
		Vc[i] = c[j - 2];
		Vr[i] = r[j - 2];
		Vindex[i + 1] = j - 1;
		for (int k = j - 3; k >= jInit; k--)
		{
			float beta = Vb[i] / c[k];
			Vr[i] = Vr[i] - beta * r[k];
			Vb[i] = Va[i] - beta * b[k];
			Va[i] = -beta * a[k];
		}
		j++;
	}
	LU_CPU(Va, Vb, Vc, Vr, 0, Vsize);
	for (int i = 0; i < Vr.size(); i++)
	{
		r[Vindex[i]] = Vr[i];
	}
	for (int i = 0; i < Vr.size(); i += 2)
	{
		int idx1 = Vindex[i] + 1;
		r[idx1] -= a[idx1] * Vr[i];
		int idx2 = Vindex[i + 1] - 1;
		r[idx2] -= c[idx2] * Vr[i + 1];

		LU_CPU(a, b, c, r, idx1, idx2 + 1);
	}
}

// ===================================================================================================================================================================== TOROK MAKE TRIDIAG
void TorokMakeTridiag(std::vector<float> y, float h, float d0, float dn, std::vector<float> &r)
{
	// std::vector<float> r((y.size() / 2) - 1);
	float mu1 = 3 / h;
	float mu2 = 4 * mu1;
	int j = 2;
	for (int i = 0; i < r.size() - 1; i++)
	{
		r[i] = mu1 * (y[j + 2] - y[j - 2]) - mu2 * (y[j + 1] - y[j - 1]);
		j += 2;
	}
	r[0] -= d0;
	j = r.size() * 2;
	int eta;
	int tau;
	if (y.size() % 2 == 0) {
		eta = -4;
		tau = 0;
	} else {
		eta = 1;
		tau = 2;
	}
	r[r.size() - 1] = mu1 * (y[j + tau] - y[j - 2]) - mu2 * (y[j + 1] - y[j - 1]) - eta * dn;
}

// =================================================================================================================================================================== MAIN
int main()
{
	const int matrixSize = 500 * 1024 + 1;
	std::vector<float> a(matrixSize);
	std::vector<float> b(matrixSize);
	std::vector<float> c(matrixSize);
	std::vector<float> r(matrixSize);
	float d1 = 1, dr = -1;
	float x1 = -4000, xr = 4000;
	std::vector<float> X(matrixSize + 2);
	std::vector<float> F(matrixSize + 2);
	std::vector<float> r3((F.size() / 2) - 1);
	float h = (xr - x1) / (X.size() - 1);
	// Data X, F:
	X[0] = x1;
	F[0] = 1 / (1 + 4 * X[0] * X[0]);
	for (int i = 1; i < X.size(); i++)
	{
		X[i] = X[i - 1] + h; //F[i] = 1 / (1 + 4 * X[i] * X[i]);
		F[i] = sin(X[i]);
	}

	// =================================================================================================================================================================== VOLAM MAKE TRIDIAG
	cudaEvent_t start_make, stop_makeT, stop_makedB;
	float timeT = 0.0;
	float timedB = 0.0;
	cudaEventCreate(&start_make);
	cudaEventCreate(&stop_makeT);
	cudaEventCreate(&stop_makedB);

	cudaEventRecord(start_make);
	cudaEventSynchronize(start_make);

	TorokMakeTridiag(F, h, d1, dr, r3);

	cudaEventRecord(stop_makeT);
	cudaEventSynchronize(stop_makeT);

	deBoorMakeEqui(X, F, d1, dr, a, b, c, r);

	cudaEventRecord(stop_makedB);
	cudaEventSynchronize(stop_makedB);
	cudaEventElapsedTime(&timeT, start_make, stop_makeT);
	cudaEventElapsedTime(&timedB, stop_makeT, stop_makedB);
	//srand(time(NULL));
	//for (size_t i = 0; i < matrixSize; i++)
	//{
	//	a[i] = rand() % 10 + 1;
	//	c[i] = rand() % 10 + 1;
	//	b[i] = a[i] + c[i] + 1 + rand() % 10; // musi byt diagonalne dominantna
	//	r[i] = rand() % 100;
	//}
	std::vector<float> a2(matrixSize);
	std::vector<float> b2(matrixSize);
	std::vector<float> c2(matrixSize);
	std::vector<float> r2(matrixSize);
	std::vector<float> r4((F.size() / 2) - 1);
	a2 = a;
	b2 = b;
	c2 = c;
	r2 = r;
	r4 = r3;
	//std::vector<float> b3((F.size() / 2) - 1);

	cudaEvent_t start, stop_CPU, stop_Torok, stop_GPU, stop_GPU_reduced;
	float time1 = 0.0;
	float time2 = 0.0;
	float time3 = 0.0;
	float time4 = 0.0;
	cudaEventCreate(&start);
	cudaEventCreate(&stop_CPU);
	cudaEventCreate(&stop_Torok);
	cudaEventCreate(&stop_GPU);
	cudaEventCreate(&stop_GPU_reduced);

	cudaEventRecord(start);
	cudaEventSynchronize(start);

	// computing on CPU
	LU_CPU(a2, b2, c2, r2, 0, r2.size());
	//ABM_on_CPU(a2, b2, c2, r2, 1024);

	cudaEventRecord(stop_CPU);
	cudaEventSynchronize(stop_CPU);
	cudaEventElapsedTime(&time1, start, stop_CPU);

	// reduced computing on CPU
	LU_CPU_equi(r3, 0, r3.size(), (F.size() % 2 == 0));
	r3 = compute_rest(r3, F, d1, dr, h);

	cudaEventRecord(stop_Torok);
	cudaEventSynchronize(stop_Torok);
	cudaEventElapsedTime(&time2, stop_CPU, stop_Torok);

	// computing on GPU
	cudaError_t cudaStatus = austin_berndt_moulton(a, b, c, r, 1024);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "GPU computing failed!\n");
		return 1;
	}

	cudaEventRecord(stop_GPU);
	cudaEventSynchronize(stop_GPU);
	cudaEventElapsedTime(&time3, stop_Torok, stop_GPU);

	// ======================================================================================================================================================== VOLAM METODU GPU REDUCED
	cudaStatus = ABM_reduced(r4, F, 1024, d1, dr, h);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "GPU computing failed!\n");
		return 1;
	}
	// r4 = compute_rest(r4, F, d1, dr, h);

	cudaEventRecord(stop_GPU_reduced);
	cudaEventSynchronize(stop_GPU_reduced);
	cudaEventElapsedTime(&time4, stop_GPU, stop_GPU_reduced);

	std::cout << "redu make time: " << timeT << " ms" << std::endl;
	std::cout << "full make time: " << timedB << " ms" << std::endl;
	std::cout << "CPU time: " << time1 << " ms" << std::endl;
	std::cout << "reduced CPU time: " << time2 << " ms" << std::endl;
	std::cout << "my GPU time: " << time3 << " ms" << std::endl;
	std::cout << "reduced GPU time: " << time4 << " ms" << std::endl;
	std::cout << "normal/reduced: " << time1 / time2 << std::endl << std::endl;
	// std::cout.precision(15);
	for (int i = 0; i < r.size(); i++)
	{
		if (r4[i + 1] != r4[i + 1])
		{
			std::cout << "hodnota " << i + 1 << " je " << r3[i + 1] << std::endl;
		}
		float diff = abs(r4[i + 1] - r[i]);
		if (diff > 0.00001) { // 10^-5
			std::cout << "BACHA! rozdiel v " << i << " je presne " << diff << std::endl;
		}
	}
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}