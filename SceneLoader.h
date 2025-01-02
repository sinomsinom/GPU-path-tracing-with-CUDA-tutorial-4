#ifndef __LOADER_H_
#define __LOADER_H_

#include "Geometry.h"
#include <string_view>

extern unsigned verticesNo;
extern Vertex* vertices;
extern unsigned int trianglesNo;
extern Triangle* triangles;

void panic(const char *fmt, ...);
void load_object(std::string_view filename);
float processgeo();

#endif
