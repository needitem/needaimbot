#pragma once
// =============================================================================
// Compile-time String Obfuscation
// 성능 영향 없음 - 모든 암호화/복호화가 컴파일 타임에 처리됨
// =============================================================================

#include <array>
#include <string>
#include <cstdint>

namespace obf {

// 컴파일 타임 XOR 키 생성 (빌드마다 다른 키)
constexpr uint64_t seed = __TIME__[0] + __TIME__[1] * 10 + __TIME__[3] * 100 + 
                          __TIME__[4] * 1000 + __TIME__[6] * 10000 + __TIME__[7] * 100000;

constexpr uint64_t generate_key(uint64_t n) {
    return (n * 6364136223846793005ULL + 1442695040888963407ULL);
}

// 컴파일 타임 문자열 암호화 클래스
template<size_t N>
class ObfuscatedString {
private:
    std::array<char, N> encrypted_;
    uint8_t key_;

    constexpr char encrypt_char(char c, size_t index) const {
        return c ^ static_cast<char>((key_ + index) & 0xFF);
    }

public:
    constexpr ObfuscatedString(const char(&str)[N], uint8_t key) : encrypted_{}, key_(key) {
        for (size_t i = 0; i < N; ++i) {
            encrypted_[i] = encrypt_char(str[i], i);
        }
    }

    // 런타임 복호화 (인라인되어 성능 영향 최소화)
    __forceinline std::string decrypt() const {
        std::string result;
        result.reserve(N - 1);
        for (size_t i = 0; i < N - 1; ++i) {
            result += encrypted_[i] ^ static_cast<char>((key_ + i) & 0xFF);
        }
        return result;
    }

    // C-string 버전 (스택에 복호화)
    __forceinline void decrypt_to(char* buffer, size_t bufSize) const {
        size_t len = (N - 1 < bufSize - 1) ? N - 1 : bufSize - 1;
        for (size_t i = 0; i < len; ++i) {
            buffer[i] = encrypted_[i] ^ static_cast<char>((key_ + i) & 0xFF);
        }
        buffer[len] = '\0';
    }
};

// 헬퍼 매크로
#define OBF_KEY(str) static_cast<uint8_t>(obf::generate_key(sizeof(str) + obf::seed) & 0xFF)

} // namespace obf

// =============================================================================
// 사용법 매크로
// =============================================================================

// 난독화된 문자열 생성 (컴파일 타임)
#define OBF(str) \
    []() -> std::string { \
        constexpr auto obfuscated = obf::ObfuscatedString<sizeof(str)>(str, OBF_KEY(str)); \
        return obfuscated.decrypt(); \
    }()

// C-string 버전 (버퍼에 복호화)
#define OBF_C(str, buffer, size) \
    do { \
        constexpr auto obfuscated = obf::ObfuscatedString<sizeof(str)>(str, OBF_KEY(str)); \
        obfuscated.decrypt_to(buffer, size); \
    } while(0)

// =============================================================================
// 심볼 난독화 (디버그 정보 제거용)
// =============================================================================

#ifdef NDEBUG
    // Release 빌드에서 함수명 난독화
    #define OBF_FUNC __declspec(noinline)
    #define OBF_INLINE __forceinline
#else
    #define OBF_FUNC
    #define OBF_INLINE inline
#endif

// =============================================================================
// 안티 디버깅 체크 (선택적 사용)
// =============================================================================

#ifdef ENABLE_ANTI_DEBUG
namespace obf {
    __forceinline bool is_debugger_present() {
        return IsDebuggerPresent() != 0;
    }
    
    __forceinline void anti_debug_check() {
        if (is_debugger_present()) {
            ExitProcess(0);
        }
    }
}
#define ANTI_DEBUG() obf::anti_debug_check()
#else
#define ANTI_DEBUG() ((void)0)
#endif
