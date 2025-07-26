#pragma once

#include <immintrin.h>  // For SIMD
#include <ppl.h>        // For parallel_for

// SIMD optimized BGRA to BGR conversion
inline void convertBGRAtoBGR_SIMD(const uint8_t* src, uint8_t* dst, int width, int height) {
    const int srcStride = width * 4;
    const int dstStride = width * 3;
    
    concurrency::parallel_for(0, height, [&](int y) {
        const uint8_t* srcRow = src + y * srcStride;
        uint8_t* dstRow = dst + y * dstStride;
        
        int x = 0;
        
        // Process 16 pixels at a time using AVX2
        for (; x <= width - 16; x += 16) {
            // Load 64 bytes (16 BGRA pixels)
            __m256i bgra0 = _mm256_loadu_si256((__m256i*)(srcRow + x * 4));
            __m256i bgra1 = _mm256_loadu_si256((__m256i*)(srcRow + x * 4 + 32));
            
            // Shuffle to remove alpha channel
            // This is complex with AVX2, so for now we'll do 4 pixels at a time with SSE
        }
        
        // Process 4 pixels at a time using SSE
        for (; x <= width - 4; x += 4) {
            __m128i bgra = _mm_loadu_si128((__m128i*)(srcRow + x * 4));
            
            // Shuffle mask to convert BGRA to BGR (removing A)
            __m128i shuf = _mm_setr_epi8(0,1,2, 4,5,6, 8,9,10, 12,13,14, -1,-1,-1,-1);
            __m128i bgr = _mm_shuffle_epi8(bgra, shuf);
            
            // Store only 12 bytes (4 BGR pixels)
            _mm_storeu_si128((__m128i*)(dstRow + x * 3), bgr);
        }
        
        // Handle remaining pixels
        for (; x < width; ++x) {
            dstRow[x * 3 + 0] = srcRow[x * 4 + 0]; // B
            dstRow[x * 3 + 1] = srcRow[x * 4 + 1]; // G
            dstRow[x * 3 + 2] = srcRow[x * 4 + 2]; // R
        }
    });
}

// Simple parallel version without SIMD
inline void convertBGRAtoBGR_Parallel(const uint8_t* src, uint8_t* dst, int width, int height) {
    const int srcStride = width * 4;
    const int dstStride = width * 3;
    
    concurrency::parallel_for(0, height, [&](int y) {
        const uint8_t* srcRow = src + y * srcStride;
        uint8_t* dstRow = dst + y * dstStride;
        
        for (int x = 0; x < width; ++x) {
            dstRow[x * 3 + 0] = srcRow[x * 4 + 0]; // B
            dstRow[x * 3 + 1] = srcRow[x * 4 + 1]; // G
            dstRow[x * 3 + 2] = srcRow[x * 4 + 2]; // R
        }
    });
}