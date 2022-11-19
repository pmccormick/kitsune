extern "C" {
__device__
void __kit_sincos_f32(float p, float *sinp, float *cosp) {
  extern float __ocml_sin_32f(float);
  extern float __ocml_cos_32f(float);
  *sinp = sinf(p);
  *cosp = cosf(p);
}
}

extern "C" {
__device__
float __kit_fdividef(float x, float y) { 
  // Not really a fast version but mimics hip support... 
  return x / y; 
}
}
