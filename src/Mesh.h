//
// Created by simon on 08/01/2025.
//

#pragma once
#include <cstdint>
#include <iostream>

#include "Array.h"
#include "linear_math.h"


/**
* Contains a mesh's data
* Mainly used for the BVH
*/
class Mesh {
public:

    struct Triangle {
        Vec3i vertices;
    };

    Mesh(const uint32_t triangles_no, const uint32_t vertices_no, const Array<Mesh::Triangle> & tris, const Array<Vec3f> & verts):
        triangles_no(triangles_no),
        vertices_no(vertices_no),
        vertices(verts),
        triangles(tris) {
        std::cout << vertices.getSize() << std::endl;
        std::cout << triangles.getSize() << std::endl;
        std::cout << vertices_no << std::endl;
        std::cout << triangles_no << std::endl;
    };

    Mesh(const Mesh& other)             = delete;
    Mesh(Mesh&& other)                  = delete;
    Mesh& operator=(const Mesh& other)  = delete;
    Mesh& operator=(Mesh&& other)       = delete;

    ~Mesh();
    [[nodiscard]] int       getNumVertices  ()                  const { return vertices.getSize();  };
    [[nodiscard]] int       getNumTriangles ()                  const { return triangles.getSize(); };
    [[nodiscard]] Triangle* getTriangleData ()                        { return triangles.getPtr();  };
    [[nodiscard]] Vec3f*    getVertexData   ()                        { return vertices.getPtr();   };
    [[nodiscard]] Triangle& getTriangle     (const int i)             { return triangles.get(i);    };
    [[nodiscard]] Vec3f&    getVertex       (const int i)             { return vertices.get(i);     };


private:
    uint32_t          triangles_no;
    uint32_t          vertices_no;
    Array<Vec3f>      vertices;
    Array<Triangle>   triangles;

};
