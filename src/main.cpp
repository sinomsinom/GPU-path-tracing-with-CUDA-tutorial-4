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
    cxxopts::Options options("SimpleCuPT", "Simple CUDA pathtracer");
    options.add_options()
        ("f,scenefile" , "Filename of Scene to load",        cxxopts::value<std::string>()->default_value("../data/dragon.obj"             ))
        ("hdr,hdrfile" , "Filename of HDR to load",          cxxopts::value<std::string>()->default_value("../data/Topanga_Forest_B_3k.hdr"))
        ("l,framelimit", "Number of frames to render in benchmark mode",cxxopts::value<int>        ()->default_value("100"                            ))
        ("b,benchmark" , "Enable Benchmark mode", cxxopts::value<bool>())
        ("o,out", "Benchmark out file", cxxopts::value<std::string>()->default_value("out.tga") )
        ("h,help", "Print usage")
    ;
    try {

        const auto results = options.parse(argc, argv);
        const std::string sceneFile  = results["scenefile" ].as<std::string>();
        const std::string hdrFile    = results["hdrfile"   ].as<std::string>();
        const bool        bmarkMode  = results["benchmark" ].as<bool>();
        const int         frameLimit = results["framelimit"].as<int>();
        const std::string outFile    = results["out"       ].as<std::string>();
        if (results.count("help")) {
            std::cout << options.help() << std::endl;
            return 0;
        }

        Application renderer(sceneFile, hdrFile, bmarkMode, frameLimit, outFile);
        renderer.initOpenGL(&argc, argv);
        renderer.start();
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        std::cerr << options.help() << std::endl;
    }
    return 0;
}