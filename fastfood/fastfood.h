#pragma once
extern "C" {
void fastfood(
    float* gaussian,
    float* radamacher,
    float* chiSquared,
    float* patchMatrix,
    float* output,
    int outSize,
    int inSize,
    int numPatches); 
}
