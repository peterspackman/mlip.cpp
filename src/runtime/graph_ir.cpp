#include "graph_ir.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

// Simple JSON parsing - we'll use nlohmann/json if available, otherwise basic
// parsing For now, implement a basic parser that handles our specific format

namespace mlipcpp::runtime {

namespace {

// Skip whitespace
void skip_ws(const std::string &s, size_t &pos) {
  while (pos < s.size() && std::isspace(s[pos])) {
    ++pos;
  }
}

// Parse a JSON string
std::string parse_string(const std::string &s, size_t &pos) {
  skip_ws(s, pos);
  if (pos >= s.size() || s[pos] != '"') {
    throw std::runtime_error("Expected string at position " +
                             std::to_string(pos));
  }
  ++pos;
  std::string result;
  while (pos < s.size() && s[pos] != '"') {
    if (s[pos] == '\\' && pos + 1 < s.size()) {
      ++pos;
      switch (s[pos]) {
      case '"':
        result += '"';
        break;
      case '\\':
        result += '\\';
        break;
      case 'n':
        result += '\n';
        break;
      case 't':
        result += '\t';
        break;
      default:
        result += s[pos];
        break;
      }
    } else {
      result += s[pos];
    }
    ++pos;
  }
  if (pos >= s.size()) {
    throw std::runtime_error("Unterminated string");
  }
  ++pos; // Skip closing quote
  return result;
}

// Parse a JSON number
double parse_number(const std::string &s, size_t &pos) {
  skip_ws(s, pos);
  size_t start = pos;
  if (pos < s.size() && (s[pos] == '-' || s[pos] == '+')) {
    ++pos;
  }
  while (pos < s.size() && (std::isdigit(s[pos]) || s[pos] == '.' ||
                            s[pos] == 'e' || s[pos] == 'E' || s[pos] == '-')) {
    ++pos;
  }
  return std::stod(s.substr(start, pos - start));
}

// Expect a character
void expect_char(const std::string &s, size_t &pos, char c) {
  skip_ws(s, pos);
  if (pos >= s.size() || s[pos] != c) {
    throw std::runtime_error("Expected '" + std::string(1, c) +
                             "' at position " + std::to_string(pos));
  }
  ++pos;
}

// Forward declarations
GIRParam parse_value(const std::string &s, size_t &pos);

// Parse an array of values
std::vector<GIRParam> parse_array(const std::string &s, size_t &pos) {
  expect_char(s, pos, '[');
  std::vector<GIRParam> result;

  skip_ws(s, pos);
  if (pos < s.size() && s[pos] == ']') {
    ++pos;
    return result;
  }

  while (true) {
    result.push_back(parse_value(s, pos));
    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == ']') {
      ++pos;
      break;
    }
    expect_char(s, pos, ',');
  }
  return result;
}

// Special values for symbolic dimensions
constexpr int64_t DIM_N_ATOMS = -1000001;
constexpr int64_t DIM_MAX_NEIGHBORS = -1000002;
constexpr int64_t DIM_SEQ_LEN = -1000003;       // n_atoms * (max_neighbors + 1)
constexpr int64_t DIM_N_EDGES = -1000004;       // n_atoms * max_neighbors
constexpr int64_t DIM_MN_PLUS_ONE = -1000005;   // max_neighbors + 1

// Convert symbolic dimension name to special value
int64_t symbolic_dim_to_value(const std::string &name) {
  if (name == "n_atoms") return DIM_N_ATOMS;
  if (name == "max_neighbors") return DIM_MAX_NEIGHBORS;
  if (name == "seq_len") return DIM_SEQ_LEN;
  if (name == "n_edges") return DIM_N_EDGES;
  if (name == "max_neighbors_plus_one") return DIM_MN_PLUS_ONE;
  // Unknown symbolic name - return -1
  return -1;
}

// Parse an array that may contain integers or symbolic dimension strings
std::vector<int64_t> parse_int_array(const std::string &s, size_t &pos) {
  expect_char(s, pos, '[');
  std::vector<int64_t> result;

  skip_ws(s, pos);
  if (pos < s.size() && s[pos] == ']') {
    ++pos;
    return result;
  }

  while (true) {
    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == '"') {
      // Symbolic dimension name
      std::string sym = parse_string(s, pos);
      result.push_back(symbolic_dim_to_value(sym));
    } else {
      // Numeric value
      result.push_back(static_cast<int64_t>(parse_number(s, pos)));
    }
    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == ']') {
      ++pos;
      break;
    }
    expect_char(s, pos, ',');
  }
  return result;
}

// Parse an array of strings
std::vector<std::string> parse_string_array(const std::string &s, size_t &pos) {
  expect_char(s, pos, '[');
  std::vector<std::string> result;

  skip_ws(s, pos);
  if (pos < s.size() && s[pos] == ']') {
    ++pos;
    return result;
  }

  while (true) {
    result.push_back(parse_string(s, pos));
    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == ']') {
      ++pos;
      break;
    }
    expect_char(s, pos, ',');
  }
  return result;
}

// Parse a generic value
GIRParam parse_value(const std::string &s, size_t &pos) {
  skip_ws(s, pos);
  if (pos >= s.size()) {
    throw std::runtime_error("Unexpected end of input");
  }

  if (s[pos] == '"') {
    return parse_string(s, pos);
  } else if (s[pos] == '[') {
    // Try to determine if it's an int array or mixed
    auto arr = parse_array(s, pos);
    if (arr.empty()) {
      return std::vector<int64_t>{};
    }
    // Check if all elements are numbers
    bool all_ints = true;
    for (const auto &v : arr) {
      if (!std::holds_alternative<double>(v) &&
          !std::holds_alternative<int64_t>(v)) {
        all_ints = false;
        break;
      }
    }
    if (all_ints) {
      std::vector<int64_t> int_arr;
      for (const auto &v : arr) {
        if (std::holds_alternative<double>(v)) {
          int_arr.push_back(static_cast<int64_t>(std::get<double>(v)));
        } else {
          int_arr.push_back(std::get<int64_t>(v));
        }
      }
      return int_arr;
    }
    // Mixed array - just return first element or empty
    return std::vector<int64_t>{};
  } else if (s.compare(pos, 4, "true") == 0) {
    pos += 4;
    return true;
  } else if (s.compare(pos, 5, "false") == 0) {
    pos += 5;
    return false;
  } else if (s.compare(pos, 4, "null") == 0) {
    pos += 4;
    return std::string("null");
  } else if (s[pos] == '-' || s[pos] == '+' || std::isdigit(s[pos])) {
    double num = parse_number(s, pos);
    // Check if it's an integer
    if (num == static_cast<int64_t>(num)) {
      return static_cast<int64_t>(num);
    }
    return num;
  } else {
    throw std::runtime_error("Unexpected character at position " +
                             std::to_string(pos));
  }
}

// Parse GIRDtype from string
GIRDtype parse_dtype(const std::string &s) {
  if (s == "f32")
    return GIRDtype::F32;
  if (s == "f16")
    return GIRDtype::F16;
  if (s == "i32")
    return GIRDtype::I32;
  if (s == "i16")
    return GIRDtype::I16;
  if (s == "i8")
    return GIRDtype::I8;
  if (s == "bool")
    return GIRDtype::BOOL;
  throw std::runtime_error("Unknown dtype: " + s);
}

// Skip a JSON object (for ignored fields)
void skip_object(const std::string &s, size_t &pos) {
  expect_char(s, pos, '{');
  int depth = 1;
  while (pos < s.size() && depth > 0) {
    if (s[pos] == '"') {
      parse_string(s, pos);
    } else if (s[pos] == '{') {
      ++depth;
      ++pos;
    } else if (s[pos] == '}') {
      --depth;
      ++pos;
    } else if (s[pos] == '[') {
      // Skip array
      int arr_depth = 1;
      ++pos;
      while (pos < s.size() && arr_depth > 0) {
        if (s[pos] == '"') {
          parse_string(s, pos);
        } else if (s[pos] == '[') {
          ++arr_depth;
          ++pos;
        } else if (s[pos] == ']') {
          --arr_depth;
          ++pos;
        } else {
          ++pos;
        }
      }
    } else {
      ++pos;
    }
  }
}

// Parse params object
std::map<std::string, GIRParam> parse_params(const std::string &s,
                                             size_t &pos) {
  std::map<std::string, GIRParam> result;
  expect_char(s, pos, '{');

  skip_ws(s, pos);
  if (pos < s.size() && s[pos] == '}') {
    ++pos;
    return result;
  }

  while (true) {
    std::string key = parse_string(s, pos);
    expect_char(s, pos, ':');
    result[key] = parse_value(s, pos);

    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == '}') {
      ++pos;
      break;
    }
    expect_char(s, pos, ',');
  }
  return result;
}

// Parse an input specification
GIRInput parse_input(const std::string &s, size_t &pos) {
  GIRInput input;
  expect_char(s, pos, '{');

  while (true) {
    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == '}') {
      ++pos;
      break;
    }

    std::string key = parse_string(s, pos);
    expect_char(s, pos, ':');

    if (key == "name") {
      input.name = parse_string(s, pos);
    } else if (key == "dtype") {
      input.dtype = parse_dtype(parse_string(s, pos));
    } else if (key == "shape") {
      input.shape = parse_int_array(s, pos);
    } else if (key == "dynamic_dims") {
      auto dims = parse_int_array(s, pos);
      input.dynamic_dims.assign(dims.begin(), dims.end());
    } else {
      // Skip unknown field
      parse_value(s, pos);
    }

    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == ',') {
      ++pos;
    }
  }
  return input;
}

// Parse an output specification
GIROutput parse_output(const std::string &s, size_t &pos) {
  GIROutput output;
  expect_char(s, pos, '{');

  while (true) {
    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == '}') {
      ++pos;
      break;
    }

    std::string key = parse_string(s, pos);
    expect_char(s, pos, ':');

    if (key == "name") {
      output.name = parse_string(s, pos);
    } else if (key == "node_ref") {
      output.node_ref = parse_string(s, pos);
    } else if (key == "dtype") {
      output.dtype = parse_dtype(parse_string(s, pos));
    } else if (key == "shape") {
      output.shape = parse_int_array(s, pos);
    } else {
      parse_value(s, pos);
    }

    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == ',') {
      ++pos;
    }
  }
  return output;
}

// Parse a node
GIRNode parse_node(const std::string &s, size_t &pos) {
  GIRNode node;
  expect_char(s, pos, '{');

  while (true) {
    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == '}') {
      ++pos;
      break;
    }

    std::string key = parse_string(s, pos);
    expect_char(s, pos, ':');

    if (key == "id") {
      node.id = static_cast<int>(parse_number(s, pos));
    } else if (key == "op") {
      node.op = parse_string(s, pos);
    } else if (key == "name") {
      node.name = parse_string(s, pos);
    } else if (key == "inputs") {
      node.inputs = parse_string_array(s, pos);
    } else if (key == "output_shape") {
      node.output_shape = parse_int_array(s, pos);
    } else if (key == "output_dtype") {
      node.output_dtype = parse_dtype(parse_string(s, pos));
    } else if (key == "params") {
      node.params = parse_params(s, pos);
    } else {
      parse_value(s, pos);
    }

    skip_ws(s, pos);
    if (pos < s.size() && s[pos] == ',') {
      ++pos;
    }
  }
  return node;
}

} // namespace

GIRGraph parse_gir_json(const std::string &json_str) {
  GIRGraph graph;
  size_t pos = 0;

  expect_char(json_str, pos, '{');

  while (true) {
    skip_ws(json_str, pos);
    if (pos >= json_str.size() || json_str[pos] == '}') {
      break;
    }

    std::string key = parse_string(json_str, pos);
    expect_char(json_str, pos, ':');

    if (key == "$schema") {
      parse_string(json_str, pos); // Ignore
    } else if (key == "version") {
      graph.version = parse_string(json_str, pos);
    } else if (key == "model_type") {
      graph.model_type = parse_string(json_str, pos);
    } else if (key == "metadata") {
      skip_object(json_str, pos); // Skip for now
    } else if (key == "constants") {
      skip_object(json_str, pos); // Skip for now
    } else if (key == "inputs") {
      expect_char(json_str, pos, '[');
      skip_ws(json_str, pos);
      while (pos < json_str.size() && json_str[pos] != ']') {
        graph.inputs.push_back(parse_input(json_str, pos));
        skip_ws(json_str, pos);
        if (json_str[pos] == ',')
          ++pos;
      }
      expect_char(json_str, pos, ']');
    } else if (key == "outputs") {
      expect_char(json_str, pos, '[');
      skip_ws(json_str, pos);
      while (pos < json_str.size() && json_str[pos] != ']') {
        graph.outputs.push_back(parse_output(json_str, pos));
        skip_ws(json_str, pos);
        if (json_str[pos] == ',')
          ++pos;
      }
      expect_char(json_str, pos, ']');
    } else if (key == "nodes") {
      expect_char(json_str, pos, '[');
      skip_ws(json_str, pos);
      while (pos < json_str.size() && json_str[pos] != ']') {
        graph.nodes.push_back(parse_node(json_str, pos));
        skip_ws(json_str, pos);
        if (json_str[pos] == ',')
          ++pos;
      }
      expect_char(json_str, pos, ']');
    } else {
      // Skip unknown field
      parse_value(json_str, pos);
    }

    skip_ws(json_str, pos);
    if (json_str[pos] == ',') {
      ++pos;
    }
  }

  return graph;
}

GIRGraph load_gir_file(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + path);
  }

  std::stringstream buffer;
  buffer << file.rdbuf();
  return parse_gir_json(buffer.str());
}

} // namespace mlipcpp::runtime
