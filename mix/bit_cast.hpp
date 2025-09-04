#include <type_traits>

namespace utils {

template <typename T, typename U>
inline T bit_cast(const U &u) {
  static_assert(sizeof(T) == sizeof(U), "Bit-casting must preserve size.");
  static_assert(std::is_trivial<T>::value, "T must be trivial copyable");
  static_assert(std::is_trivial<U>::value, "T must be trivial copyable");

  T t;
  uint8_t *t_ptr = reinterpret_cast<uint8_t*>(&t);
  const uint8_t *u_ptr = reinterpret_cast<const uint8_t*>(&u);

  for (size_t i=0; i < sizeof(U); ++i) {
    t_ptr[i] = u_ptr[i];
  }
  return t;
}

} // utils

inline int float2int(float x) {
    return utils::bit_cast<int>(x);
}

inline float int2float(int x) {
    return utils::bit_cast<float>(x);
}
