//
// Created by simon on 08/01/2025.
//

#pragma once

#include <GL/glew.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "Camera.h"
#include "CudaBVH.h"
#include "MouseKeyboardInput.h"
#include "SceneLoader.h"
#include "Util.h"

class Application {
	friend class Handler;
public:
	explicit Application(std::string_view sceneFile, std::string_view hdrFile);
	Application(const Application& other) = delete;
	Application& operator=(const Application& other) = delete;
	Application(Application&& other) = delete;
	Application& operator=(Application&& other) = delete;
	~Application();

	void initOpenGL(int* argc, char** argv);
	void start() const;

	static void Timer(int);

	void createVBO(GLuint *vbo);

	void display();

	void resize(int width, int height);
private:
	// DATA
	Vec4i* cpuNodePtr = nullptr;
	Vec4i*cpuTriWoopPtr = nullptr;
	Vec4i* cpuTriDebugPtr = nullptr;
	Vec4f* cpuTriNormalPtr = nullptr;
	S32*   cpuTriIndicesPtr = nullptr;

	float4* cudaNodePtr = nullptr;
	float4* cudaTriWoopPtr = nullptr;
	float4* cudaTriDebugPtr = nullptr;
	float4* cudaTriNormalPtr = nullptr;
	S32*    cudaTriIndicesPtr = nullptr;

	Camera* cudaRendercam = nullptr;
	std::unique_ptr<Camera> hostRendercam;
	Vec3f* accumulatebuffer = nullptr; // image buffer storing accumulated pixel samples
	Vec3f* finaloutputbuffer = nullptr; // stores averaged pixel samples
	thrust::device_vector<float4> gpuHDRenv;
	thrust::host_vector<float4> cpuHDRenv;
	Vec4f* m_triNormals = nullptr;
	CudaBVH* gpuBVH = nullptr;
	InteractiveCamera interactiveCamera;

	cudaGraphicsResource* cudaResource = nullptr;

	Clock watch;
	GLuint vbo;

	int scrwidth = 1280;
	int scrheight = 720;
	int bufwidth = 1280;
	int bufheight = 720;

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
	bool openGLInitialised = false;

	//FUNCTIONS
	void createBVH(std::string_view scenefile);

	void initCUDAscenedata();

	void initHDR(std::string_view HDRfile);

	void loadBVHfromCache(FILE *BVHcachefile);

	void writeBVHcachefile(FILE *BVHcachefile, std::string_view BVHcacheFilename) const;
};

class Handler {
public:
	inline static Application* renderer;

	/**
	 * Registers all the glut handlers to the provided renderer
	 * @param renderer_
	 */
	static void registerRenderer(Application* renderer_) {
		if (renderer == nullptr) {
			std::cerr << "Renderer was not set!\n";
			std::exit(EXIT_FAILURE);
		}
		renderer = renderer_;

		// register callback function to display graphics
		glutDisplayFunc(Handler::handleDisplay);

		// functions for user interaction
		glutKeyboardFunc(Handler::handleKeyboard);
		glutSpecialFunc(Handler::handleSpecialKeys);
		glutMouseFunc(Handler::handleMouse);
		glutMotionFunc(Handler::handleMotion);
		glutReshapeFunc(Handler::handleReshape);
	}
private:
	static void handleReshape(int w, int h) {
		renderer->resize(w,h);
	}
	static void handleDisplay() {
		renderer->display();
	}
	static void handleKeyboard(unsigned char key, int x, int y) {
		keyboard(renderer->interactiveCamera, key, x, y, renderer->scrwidth, renderer->scrheight);
	}
	static void handleSpecialKeys(int key, int x, int y) {
		specialkeys(renderer->interactiveCamera, key,x,y);
	}
	static void handleMouse(int button, int state, int x, int y) {
		mouse(renderer->interactiveCamera, button, state, x, y);
	}
	static void handleMotion( int x, int y) {
		motion(renderer->interactiveCamera, x,y);
	}
};
