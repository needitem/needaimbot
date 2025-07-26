#pragma once

#include <immintrin.h>
#include <ppl.h>
#include <cstdint>

// Optimized BGRA to BGR conversion using SIMD
inline void convertBGRAtoBGR_Optimized(const uint8_t* src, uint8_t* dst, int width, int height) {
    const int srcStride = width * 4;
    const int dstStride = width * 3;
    
    concurrency::parallel_for(0, height, [&](int y) {
        const uint8_t* srcRow = src + y * srcStride;
        uint8_t* dstRow = dst + y * dstStride;
        int x = 0;
        
        // Process 4 pixels at a time using SSE (12 output bytes from 16 input bytes)
        #ifdef __SSE3__
        const __m128i shuffleMask = _mm_setr_epi8(
            0, 1, 2,    // First pixel BGR
            4, 5, 6,    // Second pixel BGR
            8, 9, 10,   // Third pixel BGR
            12, 13, 14, // Fourth pixel BGR
            -1, -1      // Padding (ignored)
        );
        
        for (; x <= width - 4; x += 4) {
            // Load 16 bytes (4 BGRA pixels)
            __m128i bgra = _mm_loadu_si128((__m128i*)(srcRow + x * 4));
            
            // Shuffle to extract BGR components
            __m128i bgr = _mm_shuffle_epi8(bgra, shuffleMask);
            
            // Store 12 bytes (4 BGR pixels)
            _mm_storeu_si128((__m128i*)(dstRow + x * 3), bgr);
        }
        #endif
        
        // Handle remaining pixels
        for (; x < width; ++x) {
            dstRow[x * 3 + 0] = srcRow[x * 4 + 0]; // B
            dstRow[x * 3 + 1] = srcRow[x * 4 + 1]; // G
            dstRow[x * 3 + 2] = srcRow[x * 4 + 2]; // R
        }
    });
}

// Alternative version that processes 8 pixels at once with AVX2
inline void convertBGRAtoBGR_AVX2(const uint8_t* src, uint8_t* dst, int width, int height) {
    const int srcStride = width * 4;
    const int dstStride = width * 3;
    
    concurrency::parallel_for(0, height, [&](int y) {
        const uint8_t* srcRow = src + y * srcStride;
        uint8_t* dstRow = dst + y * dstStride;
        int x = 0;
        
        #ifdef __AVX2__
        // Process 8 pixels at a time
        for (; x <= width - 8; x += 8) {
            // Load 32 bytes (8 BGRA pixels)
            __m256i bgra = _mm256_loadu_si256((__m256i*)(srcRow + x * 4));
            
            // This is complex with AVX2 due to lane crossing, so we'll use two SSE operations
            __m128i bgra_lo = _mm256_extracti128_si256(bgra, 0);
            __m128i bgra_hi = _mm256_extracti128_si256(bgra, 1);
            
            // Process each half separately
            const __m128i shuffleMask = _mm_setr_epi8(0,1,2, 4,5,6, 8,9,10, 12,13,14, -1,-1,-1,-1);
            __m128i bgr_lo = _mm_shuffle_epi8(bgra_lo, shuffleMask);
            __m128i bgr_hi = _mm_shuffle_epi8(bgra_hi, shuffleMask);
            
            // Store results
            _mm_storeu_si128((__m128i*)(dstRow + x * 3), bgr_lo);
            _mm_storeu_si128((__m128i*)(dstRow + x * 3 + 12), bgr_hi);
        }
        #endif
        
        // Handle remaining pixels
        for (; x < width; ++x) {
            dstRow[x * 3 + 0] = srcRow[x * 4 + 0]; // B
            dstRow[x * 3 + 1] = srcRow[x * 4 + 1]; // G
            dstRow[x * 3 + 2] = srcRow[x * 4 + 2]; // R
        }
    });
}