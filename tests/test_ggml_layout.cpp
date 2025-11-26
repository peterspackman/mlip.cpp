#include <ggml.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  // Create context
  struct ggml_init_params params = {
      .mem_size = 1024 * 1024, .mem_buffer = NULL, .no_alloc = false};
  struct ggml_context *ctx = ggml_init(params);

  // Create a 2D tensor [3, 4]
  // In GGML: ne[0]=3, ne[1]=4
  struct ggml_tensor *t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 4);

  printf("Created tensor with ggml_new_tensor_2d(ctx, F32, 3, 4)\n");
  printf("  ne[0]=%lld, ne[1]=%lld\n", t->ne[0], t->ne[1]);
  printf("  nb[0]=%zu bytes, nb[1]=%zu bytes\n", t->nb[0], t->nb[1]);
  printf("  Total elements: %lld\n", ggml_nelements(t));

  // Fill sequentially
  float *data = (float *)t->data;
  for (int i = 0; i < 12; i++) {
    data[i] = (float)i;
  }

  printf("\nSequential data: [");
  for (int i = 0; i < 12; i++) {
    printf("%.0f%s", data[i], i < 11 ? ", " : "");
  }
  printf("]\n");

  // Test different indexing patterns
  printf("\nIf this is PyTorch [4, 3] (4 rows, 3 cols), element [row, col] "
         "should be:\n");
  printf("Pattern 1 - row + col * ne[1]: \n");
  for (int row = 0; row < 4; row++) {
    for (int col = 0; col < 3; col++) {
      int idx = row + col * 4;
      printf("  [%d,%d] -> data[%d] = %.0f\n", row, col, idx, data[idx]);
    }
  }

  printf("\nPattern 2 - col + row * ne[0]: \n");
  for (int row = 0; row < 4; row++) {
    for (int col = 0; col < 3; col++) {
      int idx = col + row * 3;
      printf("  [%d,%d] -> data[%d] = %.0f\n", row, col, idx, data[idx]);
    }
  }

  ggml_free(ctx);
  return 0;
}
