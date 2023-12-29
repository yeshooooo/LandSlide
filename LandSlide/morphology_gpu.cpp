#include "morphology_gpu.h"

using namespace cv;

morphology_gpu::morphology_gpu(int srcType)
{
	//cuda::printShortCudaDeviceInfo(cuda::getDevice());
	element = getStructuringElement(elementShape, Size(5, 5));
	openFilter = cuda::createMorphologyFilter(MORPH_OPEN, srcType, element);
}




