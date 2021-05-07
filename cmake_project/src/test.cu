#include "test.h"

__constant__ TensorEntry* deviceSparseGamma;
__constant__ int deviceSparseGammaSize;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


// CUDA kernel, performs conventional O(n^5) multiplicatoin of SH vectors
// A, B, C are pointers to SH coefficients in device memory
// layout: SH_0 [ at(0,0), at(1,-1), at(1,0), ... ], SH_1, ...

template <typename Out>
void split(const std::string &s, char delim, Out result) {
    std::istringstream iss(s);
    std::string item;
    while (std::getline(iss, item, delim)) {
        *result++ = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}

static std::vector<TensorEntry> readGamma(int n)
{
    std::vector<TensorEntry> sparsegamma;
    std::string line;
    std::ifstream sparsefile("./gamma/sparse" + std::to_string(n));
    TensorEntry entry;
    while(getline(sparsefile, line))
    {
        std::vector<std::string> tokens = split(line.substr(1, line.length() - 3), ',');
        entry.a = std::stoi(tokens[0]);
        entry.b = std::stoi(tokens[1]);
        entry.c = std::stoi(tokens[2]);
        entry.val = std::stof(tokens[3]);
        sparsegamma.push_back(entry);
    }
    sparsefile.close();
    return sparsegamma;
}


void initGamma()
{
    std::vector<TensorEntry> v,v1;
    int size = 0;
    TensorEntry* p;
    // load sparse gamma n
    v = readGamma(n);
    size = v.size();
    //console.log("sparse n size:", size);
    cudaMalloc((void**)&p, size * sizeof(TensorEntry));
    cudaMemcpy(p, &v[0], size * sizeof(TensorEntry), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(deviceSparseGamma, &p, sizeof(TensorEntry*));
    cudaMemcpyToSymbol(deviceSparseGammaSize, &size, sizeof(int));
    gpuErrchk( cudaPeekAtLastError() );
    // SparseGammaGPU = v;
}

void releaseGamma()
{
    TensorEntry* p;
    cudaMemcpyFromSymbol(&p, deviceSparseGamma, sizeof(TensorEntry*));
    cudaFree(p);
}

void traditional_method_gpu(float* A, float* B, float* C, int num){
    // set threading dimensions
    assert(num%blocksize==0);
    dim3 grid(num/blocksize,1);
    dim3 block(blocksize,1);
    shprod_conventional<<<grid, block>>>(A,B,C);
    cudaDeviceSynchronize();
}

__global__ void shprod_conventional(float* A, float* B, float* C)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int base = i*n*n;
    float Areg[n*n];
    float Breg[n*n];
    float Creg[n*n];
    memcpy(Areg, A+base, n*n*sizeof(float));
    memcpy(Breg, B+base, n*n*sizeof(float));
    memset(Creg, 0, n*n*sizeof(float));
#define e deviceSparseGamma[i]
    for (int i=0; i<deviceSparseGammaSize; ++i)
        Creg[e.c] += e.val * Areg[e.a] * Breg[e.b];
#undef e
    memcpy(C+base, Creg, n*n*sizeof(float));
}

__global__ void cu_sh2fs(float* SH, cufftComplex* FS)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int shbase = i*n*n;
    const int fsbase = i*N*N;
    // copy to register
    float SHreg[n*n];
//    cufftComplex FSreg[N*N];
    memcpy(SHreg, SH+shbase, n*n*sizeof(float));
//    memset(FSreg, 0, N*N*sizeof(cufftComplex));
    // execute
    #include "generated/sh2fs.cu"
    // copy back to global memory
//   	for (int j=0; j<N*N; ++j)
//   		FS[j+i*N*N] = FSreg[j];
}

// convert from coefficients of Fourier Series to SH vector
__global__ void cu_fs2sh(cufftComplex* FS, float* SH)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int shbase = i*n*n;
    const int fsbase = i*N*N;
    // copy to register
    float SHreg[n*n];
//    cufftComplex FSreg[N*N];
//    memset(SHreg, 0, n*n*sizeof(float));
//   	for (int j=0; j<N*N; ++j)
//   		FSreg[j] = FS[j+i*N*N];
    // execute
    #include "generated/fs2sh.cu"
    // copy back to global memory
    memcpy(SH+shbase, SHreg, n*n*sizeof(float));
}

// element-wise multiplication B_i *= A_i
__global__ void multiply(cufftComplex* A, cufftComplex* B)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
	float x = A[i].x * B[i].x - A[i].y * B[i].y;
	float y = A[i].y * B[i].x + A[i].x * B[i].y;
	B[i].x = x;
	B[i].y = y;
}

// A, B, C are pointers to SH coefficients in device memory
// layout: SH_0 [ at(0,0), at(1,-1), at(1,0), ... ], SH_1, ...
void shprod_many(float* A, float* B, float* C, int num,
    cufftComplex *pool0, cufftComplex *pool1, cufftComplex *pool2)
{
	const int blocksize = 32;
	assert(num%blocksize == 0);
	// mem alloc
	/*cufftComplex *pool0, *pool1, *pool2;
	cudaMalloc((void**)&pool0, sizeof(cufftComplex)*N*N*num);
	cudaMalloc((void**)&pool1, sizeof(cufftComplex)*N*N*num);
	cudaMalloc((void**)&pool2, sizeof(cufftComplex)*N*N*num);*/
	// plan DFT
	cufftHandle plan;
	int sizes[2] = {N,N};
	cufftPlanMany(&plan, 2, sizes, NULL, 1, N*N, NULL, 1, N*N, CUFFT_C2C, num);
    //console.time("exclude_planning " + std::to_string(num));
	// DFT on A
	cu_sh2fs<<<num/blocksize, blocksize>>>(A, pool0);
	cudaDeviceSynchronize();
    //console.time("fftexec " + std::to_string(num));
	cufftExecC2C(plan, pool0, pool1, CUFFT_FORWARD);
	// DFT on B
	cu_sh2fs<<<num/blocksize, blocksize>>>(B, pool0);
	cufftExecC2C(plan, pool0, pool2, CUFFT_FORWARD);
	// element-wise multiply
	multiply<<<num*N*N/blocksize, blocksize>>>(pool1, pool2);
	// IDFT & convert backs to SH
	cufftExecC2C(plan, pool2, pool1, CUFFT_INVERSE);
	cudaDeviceSynchronize();
    //console.timeEnd("fftexec " + std::to_string(num));
	cu_fs2sh<<<num/blocksize, blocksize>>>(pool1, C);
	// synchronize
	cudaDeviceSynchronize();
    //console.timeEnd("exclude_planning " + std::to_string(num));
}
