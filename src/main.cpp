#include <cxxopts.hpp>
#include "Application.h"

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


int main(int argc, char** argv) {
	cxxopts::Options options("CudaPT", "Simple CUDA pathtracer");
	options.add_options()
		("f,scenefile", "Filename of Scene to load", cxxopts::value<std::string>()->default_value("../data/dragon.obj"))
		("hdr,hdrfile","Filename of HDR to load", cxxopts::value<std::string>()->default_value("../data/Topanga_Forest_B_3k.hdr"));
	const auto results = options.parse(argc, argv);
	const auto sceneFile = results["scenefile"].as<std::string>();
	const auto hdrFile = results["hdrfile"].as<std::string>();

	Application renderer(sceneFile, hdrFile);
	renderer.initOpenGL(&argc, argv);
	renderer.start();

}