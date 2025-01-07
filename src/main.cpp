#include <GL/glew.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <sstream>
#include <iostream>
#include "SceneLoader.h"
#include "Camera.h"
#include "Array.h"
#include "Scene.h"
#include "Util.h"
#include "BVH.h"
#include "CudaBVH.h"
#include "CudaRenderKernel.h"
#include "HDRloader.h"
#include "MouseKeyboardInput.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cxxopts.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// test scenes

//const char* scenefile = "data/icosahedron.obj";
//const char* scenefile = "data/dragon_vrip_res3.ply";  
// const char* scenefile = "data/dragon.obj";
//const char* scenefile = "data/happy_vrip_res2.ply";  
//const char* scenefile = "data/happy_vrip.ply"; 
//const char* scenefile = "data/bun_zipper.ply";  
//const char* scenefile = "data/trumpet.obj";      // minicooper.obj, cessna.obj
//const char* scenefile = "data/italianfromblender2.obj"; 
//const char* scenefile = "data/dragon.obj"; 
//const char* scenefile = "data/sponza_crytek.obj"; 

// HDR environment

//const char* HDRmapname = "data/ArboretumInBloom_Ref.hdr"; 
//const char* HDRmapname = "data/Topanga_Forest_B_3k.hdr";
//const char* HDRmapname = "data/Ditch-River_2k.hdr";
//const char* HDRmapname = "data/GCanyon_C_YumaPoint_3k.hdr";

Vec4i* cpuNodePtr = nullptr;
Vec4i* cpuTriWoopPtr = nullptr;
Vec4i* cpuTriDebugPtr = nullptr;
Vec4f* cpuTriNormalPtr = nullptr;
S32*   cpuTriIndicesPtr = nullptr;

float4* cudaNodePtr = nullptr;
float4* cudaTriWoopPtr = nullptr;
float4* cudaTriDebugPtr = nullptr;
float4* cudaTriNormalPtr = nullptr;
S32*    cudaTriIndicesPtr = nullptr;

Camera* cudaRendercam = nullptr;
Camera* hostRendercam = nullptr;
Vec3f* accumulatebuffer = nullptr; // image buffer storing accumulated pixel samples
Vec3f* finaloutputbuffer = nullptr; // stores averaged pixel samples
thrust::device_vector<float4> gpuHDRenv;
thrust::host_vector<float4> cpuHDRenv;
Vec4f* m_triNormals = nullptr;
CudaBVH* gpuBVH = nullptr;

cudaGraphicsResource* cudaResource = nullptr;

Clock watch;
GLuint vbo;

int framenumber = 0;
int nodeSize = 0;
int leafnode_count = 0;
int triangle_count = 0;
int triWoopSize = 0;
int triDebugSize = 0;
int triIndicesSize = 0;
float scalefactor = .2f;
__device__ float timer = 0.0f;
bool nocachedBVH = false;


void Timer(int unused) {
	glutPostRedisplay();
	glutTimerFunc(10, Timer, 0);
}


void createVBO(GLuint* vbo)
{
	//Create vertex buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);

	//Initialize VBO
	unsigned int size = scrwidth * scrheight * sizeof(Vec3f);
	glBufferData(GL_ARRAY_BUFFER, size, nullptr, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	//Register VBO with CUDA
	cudaGraphicsGLRegisterBuffer(&cudaResource,*vbo,cudaGraphicsRegisterFlagsWriteDiscard);
	// cudaGLRegisterBufferObject(*vbo);
}

// display function called by glutMainLoop(), gets executed every frame 
void disp()
{
	// if camera has moved, reset the accumulation buffer
	if (buffer_reset){ cudaMemset(accumulatebuffer, 1, scrwidth * scrheight * sizeof(Vec3f)); framenumber = 0; }

	buffer_reset = false;
	framenumber++;

	// build a new camera for each frame on the CPU
	interactiveCamera->buildRenderCamera(hostRendercam);

	// copy the CPU camera to a GPU camera
	cudaMemcpy(cudaRendercam, hostRendercam, sizeof(Camera), cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	cudaGraphicsMapResources(1,&cudaResource, nullptr);

	size_t num_bytes = 0;

	cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&finaloutputbuffer), &num_bytes, cudaResource); // maps a buffer object for access by CUDA
	[[maybe_unused]] size_t num_elements = num_bytes / sizeof(Vec3f);
	glClear(GL_COLOR_BUFFER_BIT); //clear all pixels

	// calculate a new seed for the random number generator, based on the framenumber
	unsigned int hashedframes = WangHash(framenumber);

	// gateway from host to CUDA, passes all data needed to render frame (triangles, BVH tree, camera) to CUDA for execution
	cudaRender(cudaNodePtr, cudaTriWoopPtr, cudaTriDebugPtr, cudaTriIndicesPtr, finaloutputbuffer,
		accumulatebuffer, gpuHDRenv.data().get(), framenumber, hashedframes, nodeSize, leafnode_count, triangle_count, cudaRendercam);
	
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
	glDrawArrays(GL_POINTS, 0, scrwidth * scrheight);
	glDisableClientState(GL_VERTEX_ARRAY);

	glutSwapBuffers();
}

void loadBVHfromCache(FILE* BVHcachefile, const std::string_view BVHcacheFilename)
{
	if (1 != fread(&nodeSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
	if (1 != fread(&triangle_count, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
	if (1 != fread(&leafnode_count, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
	if (1 != fread(&triWoopSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
	if (1 != fread(&triDebugSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
	if (1 != fread(&triIndicesSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";

	std::cout << "Number of nodes: " << nodeSize << "\n";
	std::cout << "Number of triangles: " << triangle_count << "\n";
	std::cout << "Number of BVH leafnodes: " << leafnode_count << "\n";

	cpuNodePtr = static_cast<Vec4i *>(malloc(nodeSize * sizeof(Vec4i)));
	cpuTriWoopPtr = static_cast<Vec4i *>(malloc(triWoopSize * sizeof(Vec4i)));
	cpuTriDebugPtr = static_cast<Vec4i *>(malloc(triDebugSize * sizeof(Vec4i)));
	cpuTriIndicesPtr = static_cast<S32 *>(malloc(triIndicesSize * sizeof(S32)));

	if (nodeSize != fread(cpuNodePtr, sizeof(Vec4i), nodeSize, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
	if (triWoopSize != fread(cpuTriWoopPtr, sizeof(Vec4i), triWoopSize, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
	if (triDebugSize != fread(cpuTriDebugPtr, sizeof(Vec4i), triDebugSize, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";
	if (triIndicesSize != fread(cpuTriIndicesPtr, sizeof(S32), triIndicesSize, BVHcachefile)) std::cout << "Error reading BVH cache file!\n";

	fclose(BVHcachefile);
	std::cout << "Successfully loaded BVH from cache file!\n";
}

void writeBVHcachefile(FILE* BVHcachefile, const std::string_view BVHcacheFilename){

	BVHcachefile = fopen(BVHcacheFilename.data(), "wb");
	if (!BVHcachefile) std::cout << "Error opening BVH cache file!\n";
	if (1 != fwrite(&nodeSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
	if (1 != fwrite(&triangle_count, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
	if (1 != fwrite(&leafnode_count, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
	if (1 != fwrite(&triWoopSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
	if (1 != fwrite(&triDebugSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
	if (1 != fwrite(&triIndicesSize, sizeof(unsigned), 1, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
	if (nodeSize != fwrite(cpuNodePtr, sizeof(Vec4i), nodeSize, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
	if (triWoopSize != fwrite(cpuTriWoopPtr, sizeof(Vec4i), triWoopSize, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
	if (triDebugSize != fwrite(cpuTriDebugPtr, sizeof(Vec4i), triDebugSize, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
	if (triIndicesSize != fwrite(cpuTriIndicesPtr, sizeof(S32), triIndicesSize, BVHcachefile)) std::cout << "Error writing BVH cache file!\n";
	
	fclose(BVHcachefile);
	std::cout << "Successfully created BVH cache file!\n";
}

// initialise HDR environment map
// from https://graphics.stanford.edu/wikis/cs148-11-summer/HDRIlluminator

void initHDR(const std::string_view HDRfile){
	
	HDRImage HDRresult{};

	if (HDRLoader::load(HDRfile.data(), HDRresult))
		printf("HDR environment map loaded. Width: %d Height: %d\n", HDRresult.width, HDRresult.height);
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

void initCUDAscenedata(){

	// allocate GPU memory for accumulation buffer
	cudaMalloc(&accumulatebuffer, scrwidth * scrheight * sizeof(Vec3f));
	
	// allocate GPU memory for interactive camera
	cudaMalloc(reinterpret_cast<void **>(&cudaRendercam), sizeof(Camera));

	// allocate and copy scene databuffers to the GPU (BVH nodes, triangle vertices, triangle indices)
	cudaMalloc(reinterpret_cast<void **>(&cudaNodePtr), nodeSize * sizeof(float4));
	cudaMemcpy(cudaNodePtr, cpuNodePtr, nodeSize * sizeof(float4), cudaMemcpyHostToDevice);

	cudaMalloc(reinterpret_cast<void **>(&cudaTriWoopPtr), triWoopSize * sizeof(float4));
	cudaMemcpy(cudaTriWoopPtr, cpuTriWoopPtr, triWoopSize * sizeof(float4), cudaMemcpyHostToDevice);

	cudaMalloc(reinterpret_cast<void **>(&cudaTriDebugPtr), triDebugSize * sizeof(float4));
	cudaMemcpy(cudaTriDebugPtr, cpuTriDebugPtr, triDebugSize * sizeof(float4), cudaMemcpyHostToDevice);

	cudaMalloc(reinterpret_cast<void **>(&cudaTriIndicesPtr), triIndicesSize * sizeof(S32));
	cudaMemcpy(cudaTriIndicesPtr, cpuTriIndicesPtr, triIndicesSize * sizeof(S32), cudaMemcpyHostToDevice);

	std::cout << "Scene data copied to CUDA\n";
}

void createBVH(const std::string_view scenefile){
	
	load_object(scenefile.data());
	float maxi2 = processgeo();

	std::cout << "Scene geometry loaded and processed\n";

	// create arrays for the triangles and the vertices
	// Scene() constructor: Scene(const S32 numTris, const S32 numVerts, const Array<Triangle>& tris, const Array<Vec3f>& verts)

	Array<Scene::Triangle> tris;
	Array<Vec3f> verts;
	tris.clear();
	verts.clear();

	// convert Triangle to Scene::Triangle
	for (unsigned int i = 0; i < trianglesNo; i++){
		Scene::Triangle newtri;
		newtri.vertices = Vec3i(triangles[i]._idx1, triangles[i]._idx2, triangles[i]._idx3);
		tris.add(newtri);
	}

	// fill up Array of vertices
	for (unsigned int i = 0; i < verticesNo; i++) { 
		verts.add(Vec3f(vertices[i].x, vertices[i].y, vertices[i].z));
	}

	std::cout << "Building a new scene\n";
	Scene* scene = new Scene(trianglesNo, verticesNo, tris, verts);

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

void deleteCudaAndCpuMemory(){
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

	delete hostRendercam;
	delete interactiveCamera;
	delete gpuBVH;
}


int main(int argc, char** argv){

	cxxopts::Options options("CudaPT", "Simple CUDA pathtracer");
	options.add_options()
		("f,scenefile", "Filename of Scene to load", cxxopts::value<std::string>()->default_value("../data/dragon.obj"))
		("hdr,hdrfile","Filename of HDR to load", cxxopts::value<std::string>()->default_value("../data/Topanga_Forest_B_3k.hdr"));
	const auto results = options.parse(argc, argv);
	const auto sceneFile = results["scenefile"].as<std::string>();
	const auto hdrFile = results["hdrfile"].as<std::string>();

	// create a CPU camera
	hostRendercam = new Camera();
	// initialise an interactive camera on the CPU side
	initCamera();
	interactiveCamera->buildRenderCamera(hostRendercam);

	std::string BVHcacheFilename(sceneFile);
	BVHcacheFilename += ".bvh";
	FILE* BVHcachefile = nullptr;
	errno_t error = fopen_s(&BVHcachefile, BVHcacheFilename.c_str(), "rb");
	if (!BVHcachefile || error){ nocachedBVH = true; }

	//if (true){ // overrule cache
	if (nocachedBVH){
		std::cout << "No cached BVH file available\nCreating new BVH...\n";
		// initialise all data needed to start rendering (BVH data, triangles, vertices)
		createBVH(sceneFile);
		// store the BVH in a file
		writeBVHcachefile(BVHcachefile, BVHcacheFilename);
	}

	else { // cached BVH available
		std::cout << "Cached BVH available\nReading " << BVHcacheFilename << "...\n";
		loadBVHfromCache(BVHcachefile, BVHcacheFilename);
	}

	initCUDAscenedata(); // copy scene data to the GPU, ready to be used by CUDA
	initHDR(hdrFile); // initialise the HDR environment map

	// initialise GLUT
	glutInit(&argc, argv);
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

	// register callback function to display graphics
	glutDisplayFunc(disp);

	// functions for user interaction
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(specialkeys);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);

	// initialise GLEW
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 ")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		exit(0);
	}
	fprintf(stderr, "glew initialized  \n");

	// call Timer()
	Timer(0);
	createVBO(&vbo);
	std::cerr << "VBO created  \n";
	// enter the main loop and start rendering
	std::cerr << "Entering glutMainLoop...  \n";
	std::cout << "Rendering started...\n";
	glutMainLoop();

	deleteCudaAndCpuMemory();
}
