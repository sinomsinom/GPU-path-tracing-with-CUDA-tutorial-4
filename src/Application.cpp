//
// Created by simon on 08/01/2025.
//

#include "Application.h"

#include <cuda_gl_interop.h>
#include <filesystem>
#include <iostream>
#include <string_view>

#include <format>

#include "handle_error.h"
#include "HDRloader.h"
#include "linear_math.h"
#include "Util.h"
#include "tga_saver.h"

constexpr bool alwaysRebuild = false;

void Application::loadBVHfromCache(FILE* BVHcachefile)
{
	const auto readData = [BVHcachefile]<typename T>(T *in, size_t elementCount) {
		if (elementCount != fread(in, sizeof(T), elementCount, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
	};
	readData(&nodeSize,1);
	readData(&triangle_count,1);
	readData(&leafnode_count,1);
	readData(&triWoopSize,1);
	readData(&triDebugSize,1);
	readData(&triIndicesSize,1);

	std::cout << "Number of nodes:          " << nodeSize		<< "\n";
	std::cout << "Number of triangles:      " << triangle_count << "\n";
	std::cout << "Number of BVH leafnodes:  " << leafnode_count << "\n";
	std::cout << "Number of triWoops:       " << triWoopSize	<< "\n";
	std::cout << "Number of Debugtriangles: " << triDebugSize	<< "\n";
	std::cout << "Number of indices:        " << triIndicesSize << "\n";

	cpuNodePtr = static_cast<Vec4i *>(malloc(nodeSize * sizeof(Vec4i)));
	cpuTriWoopPtr = static_cast<Vec4i *>(malloc(triWoopSize * sizeof(Vec4i)));
	cpuTriDebugPtr = static_cast<Vec4i *>(malloc(triDebugSize * sizeof(Vec4i)));
	cpuTriIndicesPtr = static_cast<S32 *>(malloc(triIndicesSize * sizeof(S32)));

	readData(cpuNodePtr, nodeSize);
	readData(cpuTriWoopPtr, triWoopSize);
	readData(cpuTriDebugPtr, triDebugSize);
	readData(cpuTriIndicesPtr, triIndicesSize);

	fclose(BVHcachefile);
	std::cout << "Successfully loaded BVH from cache file!\n";
}

void Application::writeBVHcachefile(FILE* BVHcachefile, const std::string_view BVHcacheFilename) const {

	[[maybe_unused]] const auto error = fopen_s(&BVHcachefile, BVHcacheFilename.data(), "wb");

	if (!BVHcachefile) std::cout << "Error opening BVH cache file!\n";

	const auto writeData = [BVHcachefile]<typename T>(T const *out, size_t elementCount) {
		if (elementCount != fwrite(out, sizeof(T), elementCount, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
	};

	writeData(&nodeSize,1);
	writeData(&triangle_count,1);
	writeData(&leafnode_count,1);
	writeData(&triWoopSize,1);
	writeData(&triDebugSize,1);
	writeData(&triIndicesSize,1);
	writeData(cpuNodePtr,nodeSize);
	writeData(cpuTriWoopPtr,triWoopSize);
	writeData(cpuTriDebugPtr,triDebugSize);
	writeData(cpuTriIndicesPtr,triIndicesSize);

	fclose(BVHcachefile);
	std::cout << "Successfully created BVH cache file!\n";
}

void Application::createBVH(const std::string_view scenefile)
{

	load_object(scenefile.data());
	float maxi2 = processgeo();

	std::cout << "Scene geometry loaded and processed\n";

	// create arrays for the triangles and the vertices
	// Scene() constructor: Scene(const S32 numTris, const S32 numVerts, const Array<Triangle>& tris, const Array<Vec3f>& verts)

	Array<Mesh::Triangle> tris;
	Array<Vec3f> verts;
	tris.clear();
	verts.clear();

	// convert Triangle to Scene::Triangle
	for (unsigned int i = 0; i < trianglesNo; i++){
		Mesh::Triangle newtri;
		newtri.vertices = Vec3i(triangles[i]._idx1, triangles[i]._idx2, triangles[i]._idx3);
		tris.add(newtri);
	}

	// fill up Array of vertices
	for (unsigned int i = 0; i < verticesNo; i++) {
		verts.add(Vec3f(vertices[i].x, vertices[i].y, vertices[i].z));
	}

	std::cout << "Building a new scene\n";
	Mesh* scene = new Mesh(trianglesNo, verticesNo, tris, verts);

	std::cout << "Building BVH with spatial splits\n";
	// create a default platform
	Platform defaultplatform;
	BVH::BuildParams defaultparams;
	BVH::Stats stats;
	BVH myBVH(scene, defaultplatform, defaultparams);

	std::cout << "Building CudaBVH\n";
	// create CUDA friendly BVH datastructure
	gpuBVH = new CudaBVH(myBVH, BVHLayout_Compact2);  // BVH layout for Kepler kernel Compact2
	std::cout << "CudaBVH successfully created\n";

	cpuNodePtr = gpuBVH->getGpuNodes();
	cpuTriWoopPtr = gpuBVH->getGpuTriWoop();
	cpuTriDebugPtr = gpuBVH->getDebugTri();
	cpuTriIndicesPtr = gpuBVH->getGpuTriIndices();
	// cpuTriNormalPtr = m_triNormals;

	nodeSize = gpuBVH->getGpuNodesSize();
	triWoopSize = gpuBVH->getGpuTriWoopSize();
	triDebugSize = gpuBVH->getDebugTriSize();
	triIndicesSize = gpuBVH->getGpuTriIndicesSize();
	leafnode_count = gpuBVH->getLeafnodeCount();
	triangle_count = gpuBVH->getTriCount();
}

void Application::initCUDAscenedata()
{

	// allocate GPU memory for accumulation buffer
	cudaCheckError(cudaMalloc(&accumulatebuffer, bufwidth * bufheight * sizeof(Vec3f)));

	// allocate GPU memory for interactive camera
	cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&cudaRendercam), sizeof(Camera)));

	// allocate and copy scene databuffers to the GPU (BVH nodes, triangle vertices, triangle indices)
	cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&cudaNodePtr), nodeSize * sizeof(float4)));
	cudaCheckError(cudaMemcpy(cudaNodePtr, cpuNodePtr, nodeSize * sizeof(float4), cudaMemcpyHostToDevice));

	cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&cudaTriWoopPtr), triWoopSize * sizeof(float4)));
	cudaCheckError(cudaMemcpy(cudaTriWoopPtr, cpuTriWoopPtr, triWoopSize * sizeof(float4), cudaMemcpyHostToDevice));

	cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&cudaTriDebugPtr), triDebugSize * sizeof(float4)));
	cudaCheckError(cudaMemcpy(cudaTriDebugPtr, cpuTriDebugPtr, triDebugSize * sizeof(float4), cudaMemcpyHostToDevice));

	cudaCheckError(cudaMalloc(reinterpret_cast<void **>(&cudaTriIndicesPtr), triIndicesSize * sizeof(S32)));
	cudaCheckError(cudaMemcpy(cudaTriIndicesPtr, cpuTriIndicesPtr, triIndicesSize * sizeof(S32), cudaMemcpyHostToDevice));

	std::cout << "Scene data copied to CUDA\n";
}

void Application::initHDR(const std::string_view HDRfile){

	HDRImage HDRresult{};

	if (HDRLoader::load(HDRfile.data(), HDRresult))
		std::cout << std::format("HDR environment map loaded. Width: {} Height: {}\n", HDRresult.width, HDRresult.height);
	else{
		printf("HDR environment map not found\nAn HDR map is required as light source. Exiting now...\n");
		system("PAUSE");
		exit(0);
	}

	const int HDRwidth = HDRresult.width;
	const int HDRheight = HDRresult.height;
	cpuHDRenv.resize(HDRwidth * HDRheight);

	//_data = new RGBColor[width*height];

	for (int i = 0; i<HDRwidth; i++){
		for (int j = 0; j<HDRheight; j++){
			int idx = 3 * (HDRwidth*j + i);
			//int idx2 = width*(height-j-1)+i;
			int idx2 = HDRwidth*(j)+i;
			cpuHDRenv[idx2] = float4(HDRresult.colors[idx], HDRresult.colors[idx + 1], HDRresult.colors[idx + 2], 0.0f);
		}
	}

	// copy HDR map to CUDA
	gpuHDRenv = cpuHDRenv;
}


Application::Application(const std::string_view sceneFile, const std::string_view hdrFile, const bool benchmarkMode, const int benchmarkFrames, const std::string_view outFile):
	vbo(0),
	isBenchmark(benchmarkMode),
	numBenchmarkFrames(benchmarkFrames),
	outFileName(outFile)
{
	// create a CPU camera
	hostRendercam = std::make_unique<Camera>();
	// initialise an interactive camera on the CPU side
	initCamera(interactiveCamera, scrwidth, scrheight);
	interactiveCamera.buildRenderCamera(*hostRendercam);

	// create the BVH structure:
	std::string BVHcacheFilename(sceneFile);
	BVHcacheFilename += ".bvh";
	FILE* BVHcachefile = nullptr;
	const errno_t error = fopen_s(&BVHcachefile, BVHcacheFilename.c_str(), "rb");
	std::cout << "Loading: " << BVHcachefile << ", " << BVHcacheFilename << "\n";
	if (!BVHcachefile || error){ nocachedBVH = true; }


	if constexpr (alwaysRebuild) {
		nocachedBVH = true;
	}
	if (nocachedBVH){
		std::cout << "No cached BVH file available\nCreating new BVH...\n";
		// initialise all data needed to start rendering (BVH data, triangles, vertices)
		createBVH(sceneFile);
		// store the BVH in a file
		writeBVHcachefile(BVHcachefile, BVHcacheFilename);
	}

	else { // cached BVH available
		std::cout << "Cached BVH available\nReading " << BVHcacheFilename << "...\n";
		loadBVHfromCache(BVHcachefile);
	}
	initCUDAscenedata(); // copy scene data to the GPU, ready to be used by CUDA
	initHDR(hdrFile); // initialise the HDR environment map

	}
Application::~Application()
{
	// free CUDA memory
	cudaFree(cudaNodePtr);
	cudaFree(cudaTriWoopPtr);
	cudaFree(cudaTriDebugPtr);
	cudaFree(cudaTriNormalPtr);
	cudaFree(cudaTriIndicesPtr);
	cudaFree(cudaRendercam);
	cudaFree(accumulatebuffer);
	cudaFree(finaloutputbuffer);

	// release CPU memory
	free(cpuNodePtr);
	free(cpuTriWoopPtr);
	free(cpuTriDebugPtr);
	free(cpuTriNormalPtr);
	free(cpuTriIndicesPtr);

	delete gpuBVH;
}

void Application::initOpenGL(int* argc, char** argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB); // specify the display mode to be RGB and single buffering
	glutInitWindowPosition(100, 100); // specify the initial window position
	glutInitWindowSize(scrwidth, scrheight); // specify the initial window size
	glutCreateWindow("MatchingSocks, CUDA path tracer using SplitBVH"); // create the window and set title

	cudaSetDevice(0);

	// initialise OpenGL:
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, scrwidth, 0.0, scrheight);
	fprintf(stderr, "OpenGL initialized \n");


	// Set this renderer to be the one used by the global handler
	// register callback function to display graphics
	// register callback for user interaction
	Handler::registerRenderer(this);

	// initialise GLEW
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 ")) {
		std::cerr << "ERROR: Support for necessary OpenGL extensions missing." << std::endl;
		exit(0);
	}
	std::cerr << "glew initialized  \n";

	// call Timer()
	Timer(0);
	createVBO(&vbo);
	std::cerr << "VBO created  \n";
	openGLInitialised = true;
}


void Application::startBenchmark() {
	benchStartTime = std::chrono::steady_clock::now();
	std::cout << std::format("Starting Benchmark...\n - Framecount: {}\n - Start Time: {:%T}\n", numBenchmarkFrames, std::chrono::system_clock::now());
}

void Application::stopBenchmark() const {
	const auto benchEndTime = std::chrono::steady_clock::now();
	const auto timeDiff = std::chrono::duration_cast<std::chrono::duration<double>>(benchEndTime - benchStartTime);
	std::cout << std::format("Ending Benchmark...\n - End Time: {}\n - Time: {}\n", std::chrono::system_clock::now(), timeDiff);
	doExit();
}


void Application::display()
{
	if (isBenchmark && framenumber == 0) {
		startBenchmark();
	}
	// if camera has moved, reset the accumulation buffer
	if (buffer_reset){ cudaCheckError(cudaMemset(accumulatebuffer, 1, bufwidth * bufheight * sizeof(Vec3f))); framenumber = 0; }

	buffer_reset = false;
	framenumber++;

	// build a new camera for each frame on the CPU
	interactiveCamera.buildRenderCamera(*hostRendercam);
	// copy the CPU camera to a GPU camera
	cudaCheckError(cudaMemcpy(cudaRendercam, hostRendercam.get(), sizeof(Camera), cudaMemcpyHostToDevice));

	cudaCheckError(cudaDeviceSynchronize());
	cudaCheckError(cudaGraphicsMapResources(1,&cudaResource, nullptr));

	size_t num_bytes = 0;

	cudaCheckError(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&finaloutputbuffer), &num_bytes, cudaResource)); // maps a buffer object for access by CUDA
	[[maybe_unused]] size_t num_elements = num_bytes / sizeof(Vec3f);
	glClear(GL_COLOR_BUFFER_BIT); //clear all pixels

	// calculate a new seed for the random number generator, based on the framenumber
	unsigned int hashedframes = WangHash(framenumber);

	if (isBenchmark && (framenumber >= numBenchmarkFrames)) {
		stopBenchmark();
	}

	// gateway from host to CUDA, passes all data needed to render frame (triangles, BVH tree, camera) to CUDA for execution
	cudaRender(cudaNodePtr, cudaTriWoopPtr, cudaTriDebugPtr, cudaTriIndicesPtr, finaloutputbuffer,
		accumulatebuffer, gpuHDRenv.data().get(), framenumber, hashedframes, nodeSize, leafnode_count, triangle_count, cudaRendercam, scrwidth, scrheight, bufwidth, bufheight);

	cudaDeviceSynchronize();
	cudaGraphicsUnmapResources(1,&cudaResource, nullptr);
	// cudaGLUnmapBufferObject(vbo);
	glFlush();
	glFinish();
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(2, GL_FLOAT, 12, nullptr);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, reinterpret_cast<GLvoid *>(8));

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, bufwidth * bufheight);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();
}

[[noreturn]] void Application::doExit() const {
	saveToTGA(outFileName, bufwidth, bufheight, finaloutputbuffer);
	std::exit(EXIT_SUCCESS);
}

inline int alignedTo16(const int value) {
	return (value + (16-1)) & ~(16-1);
}

void Application::resize(const int width, const int height) {
	scrwidth  = width;
	scrheight = height;
	bufwidth  = alignedTo16(width);
	bufheight = alignedTo16(height);
	interactiveCamera.setResolution(scrwidth, scrheight);
	cudaCheckError(cudaDeviceSynchronize());

	// free buffers
	if (accumulatebuffer) {
		cudaCheckError(cudaFree(accumulatebuffer));
		accumulatebuffer = nullptr;
	}
	if (finaloutputbuffer) {
		cudaCheckError(cudaGraphicsUnregisterResource(cudaResource));
		finaloutputbuffer = nullptr;
	}

	if (vbo) {
		glDeleteBuffers(1, &vbo);
		vbo = 0;
	}

	// Reallocate buffer
	const size_t buffer_size = bufwidth * bufheight * sizeof(Vec3f);
	cudaCheckError(cudaMalloc(&accumulatebuffer, buffer_size));
	cudaCheckError(cudaMemset(accumulatebuffer, 0, buffer_size));

	// Recreate VBO
	createVBO(&vbo);
	buffer_reset = true;
    glViewport(0, 0, scrwidth, scrheight);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0, scrwidth, 0.0, scrheight); // For orthographic projection
	glMatrixMode(GL_MODELVIEW);
}

void Application::start() const {
	// enter the main loop and start rendering

	if (!openGLInitialised) {
		std::cerr << " OpenGL has not been initialized.\n";
	}
	std::cerr << "Entering glutMainLoop...  \n";
	std::cout << "Rendering started...\n";
	glutMainLoop();
}

void Application::Timer(int) {
	glutPostRedisplay();
	glutTimerFunc(10, Timer, 0);
}


void Application::createVBO(GLuint* vbo)
{
	//Create vertex buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	//Initialize VBO
	unsigned int size = bufwidth * bufheight * sizeof(Vec3f);
	glBufferData(GL_ARRAY_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//Register VBO with CUDA
	cudaCheckError(cudaGraphicsGLRegisterBuffer(&cudaResource,*vbo,cudaGraphicsRegisterFlagsWriteDiscard));
}