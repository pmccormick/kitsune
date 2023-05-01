#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <math.h>
#include <stdint.h>
#include <float.h>
#include <limits.h>
#include <stdlib.h>

#include "hip/hip_runtime.h"


#define HIPCHECK(error)                          \
  if (error != hipSuccess) {                     \
    printf("error: '%s' (%d) at %s:%d\n",        \
    hipGetErrorString(error), error, __FILE__,   \
        __LINE__);                               \
    exit(1);                                     \
  }                                              \

struct Pixel {
  unsigned char r, g, b;
};

struct Vec {
  float x,y,z;
  __forceinline__ __device__ Vec(float v = 0.0f) {x = y = z = v;}
  __forceinline__ __device__ Vec(float a, float b, float c = 0.0f) {x = a;y = b;z = c;}
  __forceinline__ __device__ Vec operator+(const Vec r) const  { return Vec(x + r.x , y + r.y , z + r.z); }
  __forceinline__ __device__ Vec operator*(const Vec r) const { return  Vec(x * r.x , y * r.y , z * r.z); }
  __forceinline__ __device__ float operator%(const Vec r) const {return x * r.x + y * r.y + z * r.z;}
  __forceinline__ __device__ Vec operator!() { return *this * (1.0/sqrtf(*this % *this)); }
};

__forceinline__ __device__
float randomVal(unsigned int& x) {
  x = (214013*x+2531011);
  return ((x>>16)&0x7FFF) / (float)66635;
}

// Rectangle CSG equation. Returns minimum signed distance from
// space carved by lowerLeft vertex and opposite rectangle vertex 
// upperRight.
__forceinline__ __device__
float BoxTest(const Vec& position, Vec lowerLeft, Vec upperRight) {
  lowerLeft = position + lowerLeft * -1.0f;
  upperRight = upperRight + position * -1.0f;
  return -fminf(fminf(fminf(lowerLeft.x, upperRight.x), 
                      fminf(lowerLeft.y, upperRight.y)),
                fminf(lowerLeft.z, upperRight.z));
}

#define HIT_NONE 0
#define HIT_LETTER 1
#define HIT_WALL 2
#define HIT_SUN 3

// Sample the world using Signed Distance Fields.
__forceinline__ __device__
float QueryDatabase(const Vec& position, int &hitType) {
  float distance = 1e9;//FLT_MAX;
  Vec f = position; // Flattened position (z=0)
  f.z = 0;
  const float lines[10*4] = {
    -20.0f,  0.0f, -20.0f, 16.0f,
    -20.0f,  0.0f, -14.0f,  0.0f,
    -11.0f,  0.0f,  -7.0f, 16.0f, 
     -3.0f,  0.0f,  -7.0f, 16.0f,
     -5.5f,  5.0f,  -9.5f,  5.0f,
      0.0f,  0.0f,   0.0f, 16.0f,
      6.0f,  0.0f,   6.0f, 16.0f,
      0.0f, 16.0f,   6.0f,  0.0f,
      9.0f,  0.0f,   9.0f, 16.0f,
      9.0f,  0.0f,  15.0f,  0.0f
  };
  
  for (unsigned i = 0; i < sizeof(lines)/sizeof(float); i += sizeof(float)) {
    Vec begin = Vec(lines[i], lines[i + 1]) * 0.5f;
    Vec e = Vec(lines[i + 2], lines[i + 3]) * 0.5f + begin * -1.0f;
    Vec o = f + (begin + e * fminf(-fminf((((begin + f * -1) % e )/(e % e)), 0),1)) * -1.0f;
    distance = fminf(distance, o % o); // compare squared distance.
  }

  distance = sqrtf(distance); // Get real distance, not square distance.
  distance = powf(powf(distance, 8)+powf(position.z, 8), 0.125f) - 0.5f;
  hitType = HIT_LETTER;

  float roomDist;
  roomDist = fminf(-fminf(
      BoxTest(position, Vec(-30.0f, -0.5f, -30.0f), Vec(30.0f, 18.0f, 30.0f)),
      BoxTest(position, Vec(-25.0f, 17.0f, -25.0f), Vec(25.0f, 20.0f, 25.0f))),
                   BoxTest( // Ceiling "planks" spaced 8 units apart.
                       Vec(fmodf(fabsf(position.x), 8.0f), position.y, position.z),
                       Vec(1.5f, 18.5f, -25.0f),
                       Vec(6.5f, 20.0f,  25.0f)));
  if (roomDist < distance) {
    distance = roomDist;
    hitType = HIT_WALL;
  }
  float sun = 19.9f - position.y; // Everything above 19.9 is light source.
  if (sun < distance) {
    distance = sun;
    hitType = HIT_SUN;
  }
  return distance;
}

// Perform signed sphere marching
// Returns hitType 0, 1, 2, or 3 and update hit position/normal
__forceinline__ __device__
int RayMarching(const Vec& origin, const Vec& direction, Vec& hitPos, Vec& hitNorm) {
  int hitType = HIT_NONE;
  int noHitCount = 0;

  // Signed distance marching
  float d; // distance from closest object in world.
  for (float total_d = 0.0f; total_d < 100.0f; total_d += d) {
    d = QueryDatabase(hitPos = origin + direction * total_d, hitType);
    if (d < .01f || ++noHitCount > 99) {
      hitNorm =
          !Vec(QueryDatabase(hitPos + Vec(0.01f, 0.00f), noHitCount) - d,
               QueryDatabase(hitPos + Vec(0.00f, 0.01f), noHitCount) - d,
               QueryDatabase(hitPos + Vec(0.00f, 0.00f, 0.01f), noHitCount) - d);
      return hitType;
    }
  }
  return HIT_NONE;
}

__forceinline__ __device__
Vec Trace(Vec origin, Vec direction, unsigned int& rn) {
  Vec sampledPosition;
  Vec normal;
  Vec color(0.0f, 0.0f, 0.0f);
  Vec attenuation(1.0f);
  Vec lightDirection(!Vec(0.6f, 0.6f, 1.0f)); // Directional light

  for (int bounceCount = 8; bounceCount--;) {
    int hitType = RayMarching(origin, direction, sampledPosition, normal);
    if (hitType == HIT_NONE)
      break;                     // No hit. This is over, return color.
    else if (hitType == HIT_LETTER) { // Specular bounce on a letter. No color acc.
      direction = direction + normal * (normal % direction * -2);
      origin = sampledPosition + direction * 0.1f;
      attenuation = attenuation * 0.2f; // Attenuation via distance traveled.
    } else if (hitType == HIT_WALL) { // Wall hit uses color yellow?
      float incidence = normal % lightDirection;
      float p = 6.283185f * randomVal(rn);
      float c = randomVal(rn);
      float s = sqrtf(1.0f - c);
      float g = normal.z < 0.0f ? -1.0f : 1.0f;
      float u = (-1.0f / (g + normal.z));
      float v = normal.x * normal.y * u;
      float cosp;
      float sinp;
      sinp = sinf(p);
      cosp = cosf(p);
      //sincosf(p, &sinp, &cosp);
      direction = Vec(v, g + normal.y * normal.y * u, -normal.y) * (cosp * s) +
                  Vec(1 + g * normal.x * normal.x * u, g * v, -g * normal.x) *
                  (sinp * s) + normal * sqrtf(c);
      origin = sampledPosition + direction * 0.1f;
      attenuation = attenuation * 0.2f;

      if (incidence > 0.0f &&
          RayMarching(sampledPosition + normal * 0.1f, lightDirection,
                      sampledPosition, normal) == HIT_SUN)
        color = color + attenuation * Vec(500, 400, 100) * incidence;
    } else if (hitType == HIT_SUN) { //
      color = color + attenuation * Vec(50, 80, 100);
      break; // Sun Color
    }
  }
  return color;
}

__global__
void PathTracer(int sampleCount, Pixel *img, int totalPixels, 
                unsigned imgWidth, unsigned imgHeight) {
  int index  = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < totalPixels) {
    int x = index % imgWidth;
    int y = index / imgWidth;
    const Vec position(-12.0f, 5.0f, 25.0f);
    const Vec goal = !(Vec(-3.0f, 4.0f, 0.0f) + position * -1.0f);
    const Vec left = !Vec(goal.z, 0, -goal.x) * (1.0f / imgWidth);
    // Cross-product to get the up vector
    const Vec up(goal.y *left.z - goal.z * left.y,
               goal.z *left.x - goal.x * left.z,
               goal.x *left.y - goal.y * left.x);
    Vec color;
    for (unsigned int p = sampleCount, v = index; p--;) {
      Vec rand_left = Vec(randomVal(v), randomVal(v), randomVal(v))*.001;
      color = color + Trace(position,
                          !((goal+rand_left) + left *
                            ((x+randomVal(v)) - imgWidth / 2.0f + randomVal(v)) + up *
                            ((y+randomVal(v)) - imgHeight / 2.0f + randomVal(v))), v);
    }
    // Reinhard tone mapping
    color = color * (1.0f / sampleCount) + 14.0f / 241.0f;
    Vec o = color + 1.0f;
    color = Vec(color.x / o.x, color.y / o.y, color.z / o.z) * 255.0f;
    img[index].r = (unsigned char)color.x;
    img[index].g = (unsigned char)color.y;
    img[index].b = (unsigned char)color.z;
  }
}

int main(int argc, char **argv)
{
   using namespace std;

  unsigned int sampleCount = 1 << 7;
  unsigned int imageWidth = 1280;
  unsigned int imageHeight = 1024;

  if (argc > 1) {
    if (argc == 2)
      sampleCount = atoi(argv[1]);
    else if (argc == 4) {
      imageWidth = atoi(argv[2]);
      sampleCount = atoi(argv[1]);
      imageHeight = atoi(argv[3]);
    } else {
      cout << "usage: raytracer [#samples] [img-width img-height]\n";
      return 1;   
    }
  }

  cout << "\n";
  cout << "---- Raytracer benchmark (forall) ----\n"
       << "  Image size    : " << imageWidth << "x" << imageHeight << "\n"
       << "  Samples/pixel : " << sampleCount << "\n\n";

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);

  cout << "  Allocating image..." << std::flush;
  Pixel *img;
  unsigned int totalPixels = imageWidth * imageHeight;
  HIPCHECK(hipMallocManaged(&img, totalPixels * sizeof(Pixel)));
  cout << "  done.\n\n";

  cout << "  Starting benchmark..." << std::flush;
  auto start_time = chrono::steady_clock::now();
  int threadsPerBlock = 256;
  int blocksPerGrid = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

  hipMemPrefetchAsync(img, totalPixels * sizeof(Pixel), 0, 0);
  hipLaunchKernelGGL(PathTracer, dim3(blocksPerGrid), dim3(threadsPerBlock), 0, 0, sampleCount,
                     img,totalPixels, imageWidth, imageHeight);
  HIPCHECK(hipDeviceSynchronize());

  auto end_time = chrono::steady_clock::now();
  double elapsed_time = chrono::duration<double>(end_time-start_time).count();

  cout << "\n\n  Total time: " << elapsed_time << " seconds.\n";
  cout << "  Pixels/second: " << totalPixels / elapsed_time << ".\n\n";

  cout << "  Saving image..." << std::flush;
  std::ofstream img_file;
  img_file.open ("raytrace-hip.ppm");
  img_file << "P6 " << imageWidth << " " << imageHeight << " 255 ";
  for(int i = totalPixels-1; i >= 0; i--)
    img_file << img[i].r << img[i].g << img[i].b;
  img_file.close();  
  cout << "  done.\n\n"
       << "*** " << elapsed_time << ", " << elapsed_time << "\n"                
       << "----\n\n";
  hipFree(img);
  return 0;
}
