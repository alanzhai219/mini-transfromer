#include <iostream>

#include "float4.hpp"

int main() {
  float va = 12.34f;
  auto va_f4 = float_to_e2m1(va);
  std::cout << va_f4 << "\n";

  float vb = 1.2f;
  auto vb_f4 = float_to_e2m1(vb);
  std::cout << vb_f4 << "\n";
  return 0;
}
