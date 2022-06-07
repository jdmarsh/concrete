// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <iosfwd>
#include <iostream>
#include <stdlib.h>

extern "C" {
#include "concrete-ffi.h"
}

#include "concretelang/ClientLib/PublicArguments.h"
#include "concretelang/ClientLib/Serializers.h"

namespace concretelang {
namespace clientlib {

template <typename Result>
Result read_deser(std::istream &istream, Result (*deser)(BufferView)) {
  size_t length;
  readSize(istream, length);
  // buffer is too big to be allocated on stack
  // vector ensures everything is deallocated w.r.t. new
  std::vector<uint8_t> buffer(length);
  istream.read((char *)buffer.data(), length);
  assert(istream.good());
  return deser({buffer.data(), length});
}

template <typename BufferLike>
std::ostream &writeBufferLike(std::ostream &ostream, BufferLike &buffer) {
  writeSize(ostream, buffer.length);
  ostream.write((const char *)buffer.pointer, buffer.length);
  assert(ostream.good());
  return ostream;
}

std::ostream &operator<<(std::ostream &ostream,
                         const LweKeyswitchKey_u64 *key) {
  Buffer b = serialize_lwe_keyswitching_key_u64(key);
  writeBufferLike(ostream, b);
  free((void *)b.pointer);
  b.pointer = nullptr;
  return ostream;
}

std::ostream &operator<<(std::ostream &ostream,
                         const LweBootstrapKey_u64 *key) {
  Buffer b = serialize_lwe_bootstrap_key_u64(key);
  writeBufferLike(ostream, b);
  free((void *)b.pointer);
  b.pointer = nullptr;
  return ostream;
}

std::istream &operator>>(std::istream &istream, LweKeyswitchKey_u64 *&key) {
  key = read_deser(istream, deserialize_lwe_keyswitching_key_u64);
  return istream;
}

std::istream &operator>>(std::istream &istream, LweBootstrapKey_u64 *&key) {
  key = read_deser(istream, deserialize_lwe_bootstrap_key_u64);
  return istream;
}

std::istream &operator>>(std::istream &istream,
                         RuntimeContext &runtimeContext) {
  istream >> runtimeContext.evaluationKeys;
  assert(istream.good());
  return istream;
}

std::ostream &operator<<(std::ostream &ostream,
                         const RuntimeContext &runtimeContext) {
  ostream << runtimeContext.evaluationKeys;
  assert(ostream.good());
  return ostream;
}

std::ostream &serializeTensorData(uint64_t *values, size_t length,
                                  std::ostream &ostream) {
  if (incorrectMode(ostream)) {
    return ostream;
  }
  writeSize(ostream, length);
  for (size_t i = 0; i < length; i++) {
    writeWord(ostream, values[i]);
  }
  return ostream;
}

std::ostream &serializeTensorData(std::vector<int64_t> &sizes, uint64_t *values,
                                  std::ostream &ostream) {
  size_t length = 1;
  for (auto size : sizes) {
    length *= size;
    writeSize(ostream, size);
  }
  serializeTensorData(values, length, ostream);
  assert(ostream.good());
  return ostream;
}

std::ostream &serializeTensorData(TensorData &values_and_sizes,
                                  std::ostream &ostream) {
  std::vector<int64_t> &sizes = values_and_sizes.sizes;
  encrypted_scalars_t values = values_and_sizes.values.data();
  return serializeTensorData(sizes, values, ostream);
}

TensorData unserializeTensorData(
    std::vector<int64_t> &expectedSizes, // includes lweSize, unsigned to
                                         // accomodate non static sizes
    std::istream &istream) {
  TensorData result;
  if (incorrectMode(istream)) {
    return result;
  }
  for (auto expectedSize : expectedSizes) {
    size_t actualSize;
    readSize(istream, actualSize);
    if ((size_t)expectedSize != actualSize) {
      istream.setstate(std::ios::badbit);
    }
    assert(actualSize > 0);
    result.sizes.push_back(actualSize);
    assert(result.sizes.back() > 0);
  }
  size_t expectedLen = result.length();
  assert(expectedLen > 0);
  // TODO: full read in one step
  size_t actualLen;
  readSize(istream, actualLen);
  if (expectedLen != actualLen) {
    istream.setstate(std::ios::badbit);
  }
  assert(actualLen == expectedLen);
  result.values.resize(actualLen);
  for (uint64_t &value : result.values) {
    value = 0;
    readWord(istream, value);
  }
  return result;
}

std::ostream &operator<<(std::ostream &ostream,
                         const LweKeyswitchKey &wrappedKsk) {
  ostream << wrappedKsk.ksk;
  assert(ostream.good());
  return ostream;
}
std::istream &operator>>(std::istream &istream, LweKeyswitchKey &wrappedKsk) {
  istream >> wrappedKsk.ksk;
  assert(istream.good());
  return istream;
}

std::ostream &operator<<(std::ostream &ostream,
                         const LweBootstrapKey &wrappedBsk) {
  ostream << wrappedBsk.bsk;
  assert(ostream.good());
  return ostream;
}
std::istream &operator>>(std::istream &istream, LweBootstrapKey &wrappedBsk) {
  istream >> wrappedBsk.bsk;
  assert(istream.good());
  return istream;
}

std::ostream &operator<<(std::ostream &ostream,
                         const EvaluationKeys &evaluationKeys) {
  ostream << *evaluationKeys.sharedKsk;
  ostream << *evaluationKeys.sharedBsk;
  assert(ostream.good());
  return ostream;
}

std::istream &operator>>(std::istream &istream,
                         EvaluationKeys &evaluationKeys) {
  auto sharedKsk = LweKeyswitchKey(nullptr);
  auto sharedBsk = LweBootstrapKey(nullptr);

  istream >> sharedKsk;
  istream >> sharedBsk;

  evaluationKeys.sharedKsk =
      std::make_shared<LweKeyswitchKey>(std::move(sharedKsk));
  evaluationKeys.sharedBsk =
      std::make_shared<LweBootstrapKey>(std::move(sharedBsk));

  assert(istream.good());
  return istream;
}

} // namespace clientlib
} // namespace concretelang
