#include <cstdio>
#include <kitsune.h>

int main(int argc, char *argv[]) {

  int nx = 3, ny = 6;
  const int n_cells = nx * ny, n_nodes = (nx + 1) * (ny + 1);

  double *source_coordinates_kit = alloc<double>(2*(n_nodes + n_cells));
  size_t *source_node_offsets_kit = alloc<size_t>(nx * ny + 1);
  
  spawn source_coordinates
  {
    forall(size_t i=0; i<2*n_nodes; ++i) source_coordinates_kit[i]=0.;
  }

  spawn source_node_offsets
  {
    forall (size_t i = 0; i < nx * ny + 1; ++i)
        source_node_offsets_kit[i] = 4 * i;
  }
  sync source_coordinates;
  sync source_node_offsets;
 
  spawn cuda_centroids
  {
    [[tapir::target("cuda")]]
    forall(size_t i = 0; i < n_cells; ++i) 
        source_coordinates_kit[2*n_nodes + 2*i]=source_coordinates_kit[2*n_nodes + 2*i+1]=1.;
  }
  sync cuda_centroids;
  
  forall (size_t i=0; i<2*n_cells; ++i) if (source_coordinates_kit[2*n_nodes +i]==10.) printf("dumb test\n");

  return 0;
}

