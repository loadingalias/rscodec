// The shim of simdjson for the fuzz targets. It reads one document with the DOM parser and writes
// its tree in the form that `fuzz_targets/read.rs` reads:
//
//   n, t, f            null, true, false
//   i <8 bytes>        an int64, little-endian
//   u <8 bytes>        a uint64 that int64 does not hold
//   d <8 bytes>        the bits of a double
//   Z <len> <bytes>    an integer that 64 bits do not hold, as its text
//   s <len> <bytes>    a string; <len> is 4 bytes, little-endian
//   k <len> <bytes>    the key of the next member
//   [ ... ]  { ... }   an array, an object
#include "simdjson.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>

namespace {

// The deepest document of the targets has 127 containers. simdjson does not count an empty
// container, so it accepts each document of that depth, and some documents that are deeper.
constexpr size_t DEPTH_MAX = 128;

std::string out;

void put_word(uint64_t word) {
  char bytes[8];
  std::memcpy(bytes, &word, 8);
  out.append(bytes, 8);
}

void put_text(char tag, std::string_view text) {
  out += tag;
  auto len = static_cast<uint32_t>(text.size());
  char bytes[4];
  std::memcpy(bytes, &len, 4);
  out.append(bytes, 4);
  out.append(text.data(), text.size());
}

void walk(simdjson::dom::element element) {
  using simdjson::dom::element_type;
  switch (element.type()) {
  case element_type::ARRAY:
    out += '[';
    for (simdjson::dom::element child : element.get_array().value_unsafe()) {
      walk(child);
    }
    out += ']';
    break;
  case element_type::OBJECT:
    out += '{';
    for (auto member : element.get_object().value_unsafe()) {
      put_text('k', member.key);
      walk(member.value);
    }
    out += '}';
    break;
  case element_type::INT64:
    out += 'i';
    put_word(static_cast<uint64_t>(element.get_int64().value_unsafe()));
    break;
  case element_type::UINT64:
    out += 'u';
    put_word(element.get_uint64().value_unsafe());
    break;
  case element_type::DOUBLE: {
    double value = element.get_double().value_unsafe();
    uint64_t bits;
    std::memcpy(&bits, &value, 8);
    out += 'd';
    put_word(bits);
    break;
  }
  case element_type::BIGINT:
    put_text('Z', element.get_bigint().value_unsafe());
    break;
  case element_type::STRING:
    put_text('s', element.get_string().value_unsafe());
    break;
  case element_type::BOOL:
    out += element.get_bool().value_unsafe() ? 't' : 'f';
    break;
  case element_type::NULL_VALUE:
    out += 'n';
    break;
  }
}

} // namespace

// Reads `input`. Returns 0 and the tree in `*tree` and `*tree_len`, valid until the next call, or
// the simdjson error code.
extern "C" int simdjson_oracle(const uint8_t *input, size_t len, const uint8_t **tree, size_t *tree_len) {
  static simdjson::dom::parser parser;
  static bool ready = false;
  if (!ready) {
    if (parser.allocate(1 << 16, DEPTH_MAX) != simdjson::SUCCESS) {
      return -1;
    }
    parser.number_as_string(true);
    ready = true;
  }
  simdjson::dom::element root;
  auto error = parser.parse(input, len).get(root);
  if (error != simdjson::SUCCESS) {
    return static_cast<int>(error);
  }
  out.clear();
  walk(root);
  *tree = reinterpret_cast<const uint8_t *>(out.data());
  *tree_len = out.size();
  return 0;
}
