/* Polygon mesh generator
//POSIBLE BUG: el algoritmo no viaja por todos los halfedges dentro de un poligono, 
    //por lo que pueden haber semillas que no se borren y tener poligonos repetidos de output
*/

#ifndef POLYLLA_HPP
#define POLYLLA_HPP


#include <array>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include "triangulation.cu"
#include <chrono>
#include <iomanip>

#define print_e(eddddge) eddddge<<" ( "<<mesh_input->origin(eddddge)<<" - "<<mesh_input->target(eddddge)<<") "

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define kernelCallCheck() \
	{ gpuErrchk( cudaPeekAtLastError() ); \
        gpuErrchk( cudaDeviceSynchronize() ); } 

#include <assert.h>
#include <cub/cub.cuh> 
#define BSIZE 1024


// Compute the distante of edge e
__device__ double distance_d(halfEdge *HalfEdges, vertex *Vertices, int e){
    double x1 = Vertices[origin_d(HalfEdges, e)].x;
    double y1 = Vertices[origin_d(HalfEdges, e)].y;
    double x2 = Vertices[target_d(HalfEdges, e)].x;
    double y2 = Vertices[target_d(HalfEdges, e)].y;

    //printf ("distance_d: %i %f %f %f %f\n", e, (float) x1, (float) y1, (float) x2, (float) y2);
    //printf ("origin_d: %i %f %f and target_d: %i %f %f\n", origin_d(HalfEdges, e), (float) x1, (float) y1, target_d(HalfEdges, e), (float) x2, (float) y2);

    return powf(x1-x2,2) + powf(y1-y2,2);
}

__device__ int compute_max_edge_d(halfEdge *HalfEdges, vertex *Vertices, int e){
    double dist0 = distance_d(HalfEdges, Vertices, e); //min
    double dist1 = distance_d(HalfEdges, Vertices, next_d(HalfEdges, e)); //mid
    double dist2 = distance_d(HalfEdges, Vertices, prev_d(HalfEdges, e)); //max

    double m1 = fmaxf(dist2, dist0);
    double m2 = fmaxf(m1, dist1);    

    //__syncthreads();
    //printf ("off: %i dist0: %f, dist1: %f, dist2: %f, m1: %f, m2: %f\n", e, (float) dist0, (float) dist1, (float) dist2, (float) m1, (float) m2);

    //if (m1 == m2)
    //    printf ("off: %i dist0: %f, dist1: %f, dist2: %f, m1: %f, m2: %f\n", e, (float) dist0, (float) dist1, (float) dist2, (float) m1, (float) m2);

    if(m2 == dist0)
        return e;
    else if(m2 == dist1)
        return next_d(HalfEdges, e);
    else
        return prev_d(HalfEdges, e);
    return -1;
}

__device__ bool is_frontier_edge_d(halfEdge *halfedges, bit_vector_d *max_edges, const int e)
{
    int twin = twin_d(halfedges, e);
    bool is_border_edge = is_border_face_d(halfedges, e) || is_border_face_d(halfedges, twin);
    bool is_not_max_edge = !(max_edges[e] || max_edges[twin]);
    if(is_border_edge || is_not_max_edge)
        return true;
    else
        return false;
}

__global__ void label_phase(halfEdge *halfedges, bit_vector_d *max_edges, bit_vector_d *frontier_edges, int n){
    int off = threadIdx.x + blockDim.x*blockIdx.x;
    if (off < n){
        frontier_edges[off] = false;
        if(is_frontier_edge_d(halfedges, max_edges, off))
            frontier_edges[off] = true;
    }
}

__global__ void label_edges_max_d(bit_vector_d *output, vertex *Vertices, halfEdge *HalfEdges, int n)
{
    int off = (blockIdx.x * blockDim.x + threadIdx.x);
    if(off < n)
    {
        int edge_max_index = compute_max_edge_d(HalfEdges, Vertices, incident_halfedge_d(off));
        __syncthreads();
        //output[off] = max;
        //output[off] = 0;
        //printf("thread: %i, edge_max_index: %i\n", off, (int)edge_max_index);
        //atomicAdd(output+edge_max_index, 1);
        output[edge_max_index] = 1;
    }
}

__device__ bool is_seed_edge_d(halfEdge *HalfEdges, bit_vector_d *max_edges, int e){
    int twin = twin_d(HalfEdges, e);

    bool is_terminal_edge = (is_interior_face_d(HalfEdges, twin) &&  (max_edges[e] && max_edges[twin]) );
    bool is_terminal_border_edge = (is_border_face_d(HalfEdges, twin) && max_edges[e]);

    if( (is_terminal_edge && e < twin ) || is_terminal_border_edge){
        return true;
    }

    return false;
}

__global__ void seed_phase_d(halfEdge *HalfEdges, bit_vector_d *max_edges, half *seed_edges, int n){
    int off = threadIdx.x + blockDim.x*blockIdx.x;
    if (off < n){
        //seed_edges[off] = 0;
        if(is_interior_face_d(HalfEdges, off) && is_seed_edge_d(HalfEdges, max_edges, off))
            seed_edges[off] = 1;
        }
}


__global__ void compaction_d(int *output, int *input, half *condition, int n){
    int off = threadIdx.x + blockDim.x*blockIdx.x;
    //printf("hola %i %i %i\n", off, input[off], condition[off]);
    if (off < n){
        if ((int)condition[off] ==  1)
            output[input[off]] = off;//*/
        //printf("hola %i %i %i %i\n", off, output[input[off]], input[off], condition[off]);
    }
}

int scan(int *d_out, int *d_in, int num_items){
    int *len = new int[1];
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    /*int *d_scan, *d_out;
    cudaMalloc(&d_out, sizeof(int)*num_items);
    cudaMalloc(&d_scan, sizeof(int)*num_items);*/
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
    gpuErrchk( cudaDeviceSynchronize() );
    cudaMemcpy(len, d_out+num_items-1, sizeof(int), cudaMemcpyDeviceToHost);
    //gpuErrchk( cudaDeviceSynchronize() );
    return *len;
}

__device__ int CW_edge_to_vertex_d(halfEdge *HalfEdges, int e)
{
    int twn, nxt;
    twn = twin_d(HalfEdges, e);
    nxt = next_d(HalfEdges, twn);
    return nxt;
}    

__device__ int CCW_edge_to_vertex_d(halfEdge *HalfEdges, int e)
{
    int twn, nxt;
    if(is_border_face_d(HalfEdges, e)){
        nxt = HalfEdges[e].prev;
        twn = HalfEdges[nxt].twin;
        return twn;
    }
    nxt = HalfEdges[e].next;
    nxt = HalfEdges[nxt].next;
    twn = HalfEdges[nxt].twin;
    return twn;
}    
__device__ int search_next_frontier_edge_d(halfEdge *HalfEdges, bit_vector_d *frontier_edges, const int e)
{
    int nxt = e;
    while(!frontier_edges[nxt])
    {
        nxt = CW_edge_to_vertex_d(HalfEdges, nxt);
    }  
    return nxt;
}

__device__ int search_prev_frontier_edge_d(halfEdge *HalfEdges, bit_vector_d *frontier_edges, const int e)
{
    int prv = e;
    while(!frontier_edges[prv])
    {
        prv = CCW_edge_to_vertex_d(HalfEdges, prv);
    }  
    return prv;
}

__global__ void travel_phase_d(halfEdge *output, halfEdge *HalfEdges, bit_vector_d *max_edges, bit_vector_d *frontier_edges, int n){
    int off = threadIdx.x + blockIdx.x*blockDim.x;
    if (off < n){
        output[off] = HalfEdges[off];
        //printf ("aca-gpu %i\n", off);
        if (!is_frontier_edge_d(HalfEdges,max_edges,off)){
            output[off].next = search_next_frontier_edge_d(HalfEdges,frontier_edges,off);
            output[off].prev = search_prev_frontier_edge_d(HalfEdges,frontier_edges,off);
        }
        else{
            output[off].next = search_next_frontier_edge_d(HalfEdges,frontier_edges,next_d(HalfEdges,off));
            output[off].prev = search_prev_frontier_edge_d(HalfEdges,frontier_edges,prev_d(HalfEdges,off));
        }
    }
}

//Travel in CCW order around the edges of vertex v from the edge e looking for the next frontier edge
__global__ void search_frontier_edge_d(int *output, halfEdge *HalfEdges, bit_vector_d *frontier_edges, int *seed_edges, int n)
{
    int off = threadIdx.x + blockIdx.x*blockDim.x;
    if (off < n){
        int nxt = seed_edges[off];
        //printf("%i %i\n",off,seed_edges[off]);
        while(!frontier_edges[nxt])
        {
            nxt = CW_edge_to_vertex_d(HalfEdges, nxt);
        }  
        output[off] = nxt;
    }
}


// cub version of the scan parallel
template <typename T>
void scan_parallel_cub(T *out, T *in, int n) {
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, in, out, n); //kernelCallCheck();
    // Allocate temporary storage*/
    cudaMalloc(&d_temp_storage, temp_storage_bytes); //kernelCallCheck();
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, in, out, n); //kernelCallCheck();
}

template <typename T>
static __device__ T one() {
  	return T{1};
}

template <typename T>
static __device__ T zero() {
  	return T{0};
}

#define WARPSIZE 32
#define WARP_PER_BLOCK 32
#define SEGMENT_SIZE 256 * WARP_PER_BLOCK
#define BLOCK_DIM WARP_PER_BLOCK * WARPSIZE

#include <cuda.h>
#include <mma.h>
#include <cub/cub.cuh> 
using namespace nvcuda;
static const int M              = 16;
static const int N              = 16;
static const int K              = 16;
static const int WMMA_TILE_SIZE = (M * N);

template <typename T>
__global__ void add_partial_sums_2(T *output, half *d_in, T *sums_warp, T *sums_block, int num_elements) {
	const int offset = threadIdx.x + blockIdx.x * blockDim.x;
	//const int globalWarpIdx = (threadIdx.x + blockDim.x * blockIdx.x)/WARPSIZE;
	const int globalSegmentIdx = offset >> 13; // /8192;
	const int globalWarpIdx = offset >> 8; // /256
	if (offset < num_elements) {
		output[offset+1] = (T)d_in[offset] + sums_warp[globalWarpIdx] + sums_block[globalSegmentIdx];
		//printf("%i %i %i %i\n",offset,(int)partial_sums[offset],globalSegmentIdx,(int)segmented_partial_sums[globalSegmentIdx]);
	}
}

// Compute scan using Tensor Cores
//template <int SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
template <typename T, typename V>
static __global__ void compute_wmma_segmented_prefixsum_256n_block_ps_2(V *d_out, T *sums_warp, T *sums_block, V *d_in, int num_segments) {
	
	T acc = 0;
	//__shared__ T partial_acc;
	__shared__ half u_frag_s[WMMA_TILE_SIZE];
	__shared__ half l_frag_s[WMMA_TILE_SIZE];
	__shared__ half la_mat_s[SEGMENT_SIZE];
	//int acc = 0; // only use the first 16

	//__shared__ V l_sum[WARP_PER_BLOCK];
	//__shared__ V l_out[SEGMENT_SIZE];
	
	const int localWarpIdx = threadIdx.x >> 5;// WARPSIZE;
	const int local_offset = localWarpIdx * WMMA_TILE_SIZE;
	//const int laneid = threadIdx.x % WARPSIZE;
	//const int globalSegmentIdx = (threadIdx.x + blockDim.x * blockIdx.x)/SEGMENT_SIZE;
	const int globalWarpIdx = (threadIdx.x + blockDim.x * blockIdx.x) >> 5; //WARPSIZE;
	const int offset = local_offset + blockIdx.x*SEGMENT_SIZE; //global_offset + (+ localWarpIdx) * WMMA_TILE_SIZE;
	
	#pragma unroll
	for (int idx = threadIdx.x; idx < WMMA_TILE_SIZE; idx += BLOCK_DIM) {
		const auto ii = idx / N;
		const auto jj = idx % N;
		u_frag_s[idx] = ii <= jj ? one<half>() : zero<half>();
		l_frag_s[idx] = ii <= jj ? zero<half>() : one<half>();
	}
	
	__syncthreads();
	
	wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
	wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> u_frag;
	wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> l_frag;
	wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> o_frag;
	wmma::fragment<wmma::accumulator, M, N, K, half> la_frag;
	wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> la_mat_frag;
	wmma::fragment<wmma::accumulator, M, N, K, half> au_frag;
	wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;
	
	wmma::load_matrix_sync(u_frag, u_frag_s, 16);
	wmma::load_matrix_sync(l_frag, l_frag_s, 16);
	wmma::fill_fragment(o_frag, one<half>());
	wmma::fill_fragment(out_frag, zero<half>());

	wmma::fill_fragment(out_frag, zero<half>());
	wmma::fill_fragment(la_frag, zero<half>());
	wmma::load_matrix_sync(a_frag, d_in + offset, 16);
	wmma::load_matrix_sync(b_frag, d_in + offset, 16);

	wmma::mma_sync(au_frag, a_frag, u_frag, out_frag);
	wmma::mma_sync(la_frag, l_frag, b_frag, la_frag);

	// store accumulator la_frag into shared memory and load it into
	// matrix_a
	// fragment la_mat_frag
	wmma::store_matrix_sync(la_mat_s + local_offset, la_frag, 16, wmma::mem_row_major);
	wmma::load_matrix_sync(la_mat_frag, la_mat_s + local_offset, 16);

	wmma::mma_sync(out_frag, la_mat_frag, o_frag, au_frag);

	wmma::store_matrix_sync(d_out + offset, out_frag, 16, wmma::mem_row_major);
	//wmma::store_matrix_sync(l_out + local_offset, out_frag, 16, wmma::mem_row_major);

	__syncthreads();
	// then, do the scan on the warp accumulation
	if (threadIdx.x < WARP_PER_BLOCK) {
		acc = d_out[threadIdx.x*WMMA_TILE_SIZE + blockIdx.x*SEGMENT_SIZE + WMMA_TILE_SIZE - 1];
		//printf("-> %i %i %i\n",threadIdx.x, threadIdx.x*256 + blockIdx.x*8192 + 255, (int)acc);
	}
	__syncthreads();
    // Specialize WarpScan for type T
    typedef cub::WarpScan<T> WarpScan;
    // Allocate WarpScan shared memory for WARP_PER_BLOCK warps
    __shared__ typename WarpScan::TempStorage temp_storage[WARP_PER_BLOCK];
    // Obtain one input item per thread
    // Compute inclusive warp-wide prefix sums
	//__syncthreads();
    WarpScan(temp_storage[globalWarpIdx]).InclusiveSum(acc, acc);
	__syncthreads(); 

	if (threadIdx.x < WARP_PER_BLOCK - 1) {
		sums_warp[threadIdx.x+blockIdx.x*WARP_PER_BLOCK + 1] = acc;
	}
	__syncthreads();

	// store the partial sum of the current warp in shared memory
	if(threadIdx.x == WARP_PER_BLOCK - 1) {
		sums_block[blockIdx.x] = acc; //(T) sums_warp[WARP_PER_BLOCK];// acc;//temp_storage[threadIdx.x];
		if (blockIdx.x == 0) {
			sums_warp[0] = 0;
		}
	}
}

// scan parallel calcule the prefix sum using CUDA-TC programming from the array in and write the result in out
template <typename T>
void scan_parallel_tc_2(T *out, half *in, int n) {
	int num_segments = (n + 255) >> 8; //256;
	int num_block = (n + 8191) >> 13; //8192; //(n + SEGMENT_SIZE - 1) / SEGMENT_SIZE; //(n + 32 - 1)/32 + 1; //
    dim3 blockDim(BLOCK_DIM,1,1);
    dim3 gridDim(num_block,1,1);
	T *sums_block;
	T *sums_warp;
	half *sums_thread;
	cudaMalloc(&sums_block,sizeof(T)*num_block);
	cudaMalloc(&sums_warp,sizeof(T)*(num_segments));
	cudaMalloc(&sums_thread,sizeof(half)*n);
    compute_wmma_segmented_prefixsum_256n_block_ps_2<T,half><<<gridDim, blockDim>>>(sums_thread, sums_warp, sums_block, in, n); kernelCallCheck();
    cudaDeviceSynchronize();
	
	T *aux;
	cudaMalloc(&aux,sizeof(T)*num_block);
	scan_parallel_cub<T>(aux,sums_block,num_block); kernelCallCheck();
	add_partial_sums_2<T><<<(n+BSIZE-1)/BSIZE, BSIZE>>>(out, sums_thread, sums_warp, aux, n); kernelCallCheck(); //<256, SEGMENT_SIZE>*/
    cudaDeviceSynchronize();
}



class Polylla
{
private:
    typedef std::vector<int> _polygon; 
    typedef std::vector<char> bit_vector; 


    Triangulation *mesh_input; // Halfedge triangulation
    Triangulation *mesh_output;
    std::vector<int> output_seeds; //Seeds of the polygon

    //std::vector<int> triangles; //True if the edge generated a triangle CHANGE!!!!

    bit_vector max_edges; //True if the edge i is a max edge
    bit_vector frontier_edges; //True if the edge i is a frontier edge
    std::vector<int> seed_edges; //Seed edges that generate polygon simple and non-simple

    // Auxiliary array used during the barrier-edge elimination
    std::vector<int> triangle_list;
    bit_vector seed_bet_mark;

    //Statistics
    int m_polygons = 0; //Number of polygons
    int n_frontier_edges = 0; //Number of frontier edges
    int n_barrier_edge_tips = 0; //Number of barrier edge tips
    int n_polygons_to_repair = 0;
    int n_polygons_added_after_repair = 0;

    // Times
    double t_label_max_edges = 0;
    double t_label_frontier_edges = 0;
    double t_label_seed_edges = 0;
    double t_traversal_and_repair = 0;
    double t_traversal = 0;
    double t_repair = 0;
    
public:

    Polylla() {}; //Default constructor

    //Constructor with triangulation
    Polylla(Triangulation *input_mesh){
        this->mesh_input = input_mesh;
        construct_Polylla();
    }



    //Constructor from a OFF file
    Polylla(std::string off_file){

        this->mesh_input = new Triangulation(off_file);
        mesh_output = new Triangulation(*mesh_input);
        construct_Polylla();
    }

    //Constructor from a node_file, ele_file and neigh_file
    Polylla(std::string node_file, std::string ele_file, std::string neigh_file){
        this->mesh_input = new Triangulation(node_file, ele_file, neigh_file);
        //call copy constructor
        mesh_output = new Triangulation(*mesh_input);
        construct_Polylla();
    }

    ~Polylla() {
        //triangles.clear(); 
        max_edges.clear(); 
        frontier_edges.clear();
        seed_edges.clear(); 
        seed_bet_mark.clear();
        triangle_list.clear();
        delete mesh_input;
        delete mesh_output;
    }

    void construct_Polylla(){

        max_edges = bit_vector(mesh_input->halfEdges(), false);
        frontier_edges = bit_vector(mesh_input->halfEdges(), false);
        //triangles = mesh_input->get_Triangles(); //Change by triangle list
        seed_bet_mark = bit_vector(this->mesh_input->halfEdges(), false);

        //terminal_edges = bit_vector(mesh_input->halfEdges(), false);
        //seed_edges = bit_vector(mesh_input->halfEdges(), false);
        
        // declare and initialize device arrays
        //int *d_triangles;
        int n_triangle = mesh_input->faces();

        // copy halfedges to device
        int n_halfedges = mesh_input->halfEdges();
        halfEdge *halfedges_d, *halfedges_h = new halfEdge[n_halfedges];
        halfedges_h = mesh_input->HalfEdges.data();
        cudaMalloc( &halfedges_d, n_halfedges*sizeof(halfEdge) );
        cudaMemcpy( halfedges_d, halfedges_h, n_halfedges*sizeof(halfEdge), cudaMemcpyHostToDevice );

        // copy vertices to device
        int n_vertices = mesh_input->vertices();
        vertex *vertices_d, *vertices_h = new vertex[n_vertices];
        vertices_h = mesh_input->Vertices.data();
        cudaMalloc( &vertices_d, n_vertices*sizeof(vertex) );
        cudaMemcpy( vertices_d, vertices_h, n_vertices*sizeof(vertex), cudaMemcpyHostToDevice );

        bit_vector_d *max_edges_d;
        cudaMalloc( &max_edges_d, n_halfedges*sizeof(bit_vector_d) );
        cudaMemset(max_edges_d, 0, n_halfedges*sizeof(bit_vector_d));

        // DEFINE GRID AND BLOCK SIZE
        dim3 block, grid;
        block = dim3(BSIZE, 1, 1);    
        grid = dim3((n_triangle + BSIZE - 1)/BSIZE, 1, 1);

        gpuErrchk( cudaDeviceSynchronize() );
        label_edges_max_d<<<grid, block>>>(max_edges_d, vertices_d, halfedges_d, n_triangle);
        gpuErrchk( cudaDeviceSynchronize() );

        //Label max edges of each triangle
        //for (size_t t = 0; t < mesh_input->faces(); t++){
        auto t_start = std::chrono::high_resolution_clock::now();
        for(int i = 0; i < mesh_input->faces(); i++)
            max_edges[label_max_edge(mesh_input->incident_halfedge(i))] = true;

        
        // copy back to host
        bit_vector_d *h_max_edges = new bit_vector_d[n_halfedges];
        cudaMemcpy( h_max_edges, max_edges_d, n_halfedges*sizeof(bit_vector_d), cudaMemcpyDeviceToHost );
        //cudaMemcpy( max_edges, max_edges_d, n_halfedges*sizeof(bit_vector_d), cudaMemcpyDeviceToHost );
        gpuErrchk( cudaDeviceSynchronize() );

        int count = 0;
        for(int i = 0; i < n_halfedges; i++){
            //printf("%i %i %i\n", i, (int) h_max_edges[i], (int) max_edges[i]);
            //assert(max_edges[i] == h_max_edges[i]);
            if (max_edges[i] != h_max_edges[i]){
                //printf("%i %i %i\n", i, (int) h_max_edges[i], (int) max_edges[i]);
                //printf ("Distances: %f %f %f\n", mesh_input->distance(i), mesh_input->distance(mesh_input->next(i)), mesh_input->distance(mesh_input->prev(i)));
                count++;
            }
        }
        printf ("Number of errors: %i\n", count); //*/

        
        for(int i = 0; i < n_halfedges; i++)
            max_edges[i] = h_max_edges[i];

        
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"Labered max edges in "<<elapsed_time_ms<<" ms"<<std::endl;


        block = dim3(BSIZE, 1, 1);    
        grid = dim3((n_halfedges + BSIZE - 1)/BSIZE, 1, 1);    
        

        bit_vector_d *frontier_edges_d;
        cudaMalloc(&frontier_edges_d, sizeof(bit_vector_d)*n_halfedges);
        label_phase<<<grid,block>>>(halfedges_d, max_edges_d, frontier_edges_d, n_halfedges); 
        cudaDeviceSynchronize(); //*/


        t_start = std::chrono::high_resolution_clock::now();
        //Label frontier edges
        for (std::size_t e = 0; e < mesh_input->halfEdges(); e++){
            if(is_frontier_edge(e)){
                frontier_edges[e] = true;
                n_frontier_edges++;
            }
        }
        gpuErrchk( cudaDeviceSynchronize() );
        // copy back to host
        bit_vector_d *h_frontier_edges = new bit_vector_d[n_halfedges];
        cudaMemcpy( h_frontier_edges, frontier_edges_d, n_halfedges*sizeof(bit_vector_d), cudaMemcpyDeviceToHost );
        //cudaMemcpy( h_frontier_edges, frontier_edges_d, n_halfedges*sizeof(bit_vector_d), cudaMemcpyDeviceToHost );
        gpuErrchk( cudaDeviceSynchronize() );
        

        for(int i = 0; i < n_halfedges; i++){
            //printf("%i %i %i\n", i, (int) h_max_edges[i], (int) max_edges[i]);
            assert(max_edges[i] == h_max_edges[i]);
            //printf("%i %i %i\n", i, (int) h_frontier_edges[i], (int) frontier_edges[i]);
        }
        
        for(int i = 0; i < n_halfedges; i++)
            frontier_edges[i] = h_frontier_edges[i];

        t_end = std::chrono::high_resolution_clock::now();
        elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"Labeled frontier edges in "<<elapsed_time_ms<<" ms"<<std::endl;
        

        // GPU SEED PHASE
        half *seed_edges_ad;
        int *seed_edges_d;
        cudaMalloc(&seed_edges_ad, sizeof(half)*n_halfedges);
        cudaMemset(seed_edges_ad, 0, sizeof(half)*n_halfedges);
        cudaMalloc(&seed_edges_d, sizeof(int)*n_halfedges);
        gpuErrchk( cudaDeviceSynchronize() );
        seed_phase_d<<<grid,block>>>(halfedges_d, max_edges_d, seed_edges_ad, n_halfedges); 
        gpuErrchk( cudaDeviceSynchronize() );

        int seed_len;
        scan_parallel_tc_2<int>(seed_edges_d, seed_edges_ad, n_halfedges);
        cudaMemcpy( &seed_len, seed_edges_d + n_halfedges - 1, sizeof(int), cudaMemcpyDeviceToHost );
        //int seed_len = scan(seed_edges_d, seed_edges_ad, n_halfedges); // ESTO SE PUEDE MEJORAR!
        gpuErrchk( cudaDeviceSynchronize() );
        //printf ("-> %i %i %i %i\n", grid.x, grid.y, grid.z, (n_halfedges + BSIZE - 1)/BSIZE);
        compaction_d<<<(n_halfedges + BSIZE - 1)/BSIZE,BSIZE>>>(seed_edges_d, seed_edges_d, seed_edges_ad, n_halfedges);
        gpuErrchk( cudaDeviceSynchronize() );
        //compaction_cub(seed_edges_d, d_num, max_edges_d, seed_edges_ad, n_halfedges);
        //gpuErrchk( cudaDeviceSynchronize() );
        //printf("\ndone GPU seed phase....\n\n");//*/


        t_start = std::chrono::high_resolution_clock::now();
        //label seeds edges,
        for (std::size_t e = 0; e < mesh_input->halfEdges(); e++)
            if(mesh_input->is_interior_face(e) && is_seed_edge(e))
                seed_edges.push_back(e);
        t_end = std::chrono::high_resolution_clock::now();
        elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"Labeled seed edges in "<<elapsed_time_ms<<" ms"<<std::endl;

        // copy back to host
        int *h_seed_edges = new int[n_halfedges];
        gpuErrchk( cudaDeviceSynchronize() );
        cudaMemcpy( h_seed_edges, seed_edges_d, seed_len*sizeof(int), cudaMemcpyDeviceToHost );
        gpuErrchk( cudaDeviceSynchronize() );

        printf("Number of seeds: %i\n", seed_len);
        for (int i = 0; i < seed_len; i++){
            if (seed_edges[i] != h_seed_edges[i])
                printf ("%i %i %i\n", i, h_seed_edges[i], seed_edges[i]);
            //assert(h_seed_edges[i] == seed_edges[i]);
        }//*/

        for (int i = 0; i < seed_len; i++)
            seed_edges[i] = h_seed_edges[i];

        // TRAVEL PHASE IN GPU!!!!
        int *output_seed_d;
        halfEdge *output_HalfEdges_d;
        cudaMalloc(&output_HalfEdges_d, sizeof(halfEdge)*n_halfedges);
        travel_phase_d<<<(n_halfedges + BSIZE - 1)/BSIZE,block>>>(output_HalfEdges_d, halfedges_d, max_edges_d, frontier_edges_d, n_halfedges);
        gpuErrchk( cudaDeviceSynchronize() );
        cudaMalloc(&output_seed_d , sizeof(int)*seed_len);
        search_frontier_edge_d<<<(seed_len+BSIZE-1)/BSIZE,block>>>(output_seed_d, halfedges_d, frontier_edges_d, seed_edges_d, seed_len);
        gpuErrchk( cudaDeviceSynchronize() );

 

        //Travel phase: Generate polygon mesh
        int polygon_seed;
        //Foreach seed edge generate polygon
        t_start = std::chrono::high_resolution_clock::now();
        for(auto &e : seed_edges){
            polygon_seed = travel_triangles(e);
            if(!has_BarrierEdgeTip(polygon_seed)){ //If the polygon is a simple polygon then is part of the mesh
                output_seeds.push_back(polygon_seed);
            }else{ //Else, the polygon is send to reparation phase
                auto t_start_repair = std::chrono::high_resolution_clock::now();
                barrieredge_tip_reparation(polygon_seed);
                auto t_end_repair = std::chrono::high_resolution_clock::now();
                t_repair += std::chrono::duration<double, std::milli>(t_end_repair-t_start_repair).count();
            }         
        }    
        t_end = std::chrono::high_resolution_clock::now();
        t_traversal_and_repair = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        t_traversal = t_traversal_and_repair - t_repair;

        printf ("aca\n");
        // COPY OUTPUT_SEED TO HOST
        int *output_seed_h = new int[seed_len];
        cudaMemcpy(output_seed_h, output_seed_d, sizeof(int)*seed_len, cudaMemcpyDeviceToHost);
        for (uint i = 0; i < seed_len; i++) {
            printf("%i %i %i\n",i,output_seed_h[i],output_seeds[i]);
            output_seeds[i] == output_seed_h[i];
            //assert(output_seed_h[i] == output_seeds[i]);
        }

        this->m_polygons = output_seeds.size();

        std::cout<<"Mesh with "<<m_polygons<<" polygons "<<n_frontier_edges/2<<" edges and "<<n_barrier_edge_tips<<" barrier-edge tips."<<std::endl;
        //mesh_input->print_pg(std::to_string(mesh_input->vertices()) + ".pg");             
    }


    void print_stats(std::string filename){
        //Time
        std::cout<<"Time to generate Triangulation: "<<mesh_input->get_triangulation_generation_time()<<" ms"<<std::endl;
        std::cout<<"Time to label max edges "<<t_label_max_edges<<" ms"<<std::endl;
        std::cout<<"Time to label frontier edges "<<t_label_frontier_edges<<" ms"<<std::endl;
        std::cout<<"Time to label seed edges "<<t_label_seed_edges<<" ms"<<std::endl;
        std::cout<<"Time to label total"<<t_label_max_edges+t_label_frontier_edges+t_label_seed_edges<<" ms"<<std::endl;
        std::cout<<"Time to traversal and repair "<<t_traversal_and_repair<<" ms"<<std::endl;
        std::cout<<"Time to traversal "<<t_traversal<<" ms"<<std::endl;
        std::cout<<"Time to repair "<<t_repair<<" ms"<<std::endl;
        std::cout<<"Time to generate polygonal mesh "<<t_label_max_edges + t_label_frontier_edges + t_label_seed_edges + t_traversal_and_repair<<" ms"<<std::endl;

        //Memory
        long long m_max_edges =  sizeof(decltype(max_edges.back())) * max_edges.capacity();
        long long m_frontier_edge = sizeof(decltype(frontier_edges.back())) * frontier_edges.capacity();
        long long m_seed_edges = sizeof(decltype(seed_edges.back())) * seed_edges.capacity();
        long long m_seed_bet_mar = sizeof(decltype(seed_bet_mark.back())) * seed_bet_mark.capacity();
        long long m_triangle_list = sizeof(decltype(triangle_list.back())) * triangle_list.capacity();
        long long m_mesh_input = mesh_input->get_size_vertex_half_edge();
        long long m_mesh_output = mesh_output->get_size_vertex_half_edge();
        long long m_vertices_input = mesh_input->get_size_vertex_struct();
        long long m_vertices_output = mesh_output->get_size_vertex_struct();

        std::ofstream out(filename);
        std::cout<<"Printing JSON file as "<<filename<<std::endl;
        out<<"{"<<std::endl;
        out<<"\"n_polygons\": "<<m_polygons<<","<<std::endl;
        out<<"\"n_frontier_edges\": "<<n_frontier_edges/2<<","<<std::endl;
        out<<"\"n_barrier_edge_tips\": "<<n_barrier_edge_tips<<","<<std::endl;
        out<<"\"n_half_edges\": "<<mesh_input->halfEdges()<<","<<std::endl;
        out<<"\"n_faces\": "<<mesh_input->faces()<<","<<std::endl;
        out<<"\"n_vertices\": "<<mesh_input->vertices()<<std::endl;
        out<<"\"n_polygons_to_repair\": "<<n_polygons_to_repair<<","<<std::endl;
        out<<"\"n_polygons_added_after_repair\": "<<n_polygons_added_after_repair<<","<<std::endl;
        out<<"\"time_triangulation_generation\": "<<mesh_input->get_triangulation_generation_time()<<","<<std::endl;
        out<<"\"time_to_label_max_edges\": "<<t_label_max_edges<<","<<std::endl;
        out<<"\"time_to_label_frontier_edges\": "<<t_label_frontier_edges<<","<<std::endl;
        out<<"\"time_to_label_seed_edges\": "<<t_label_seed_edges<<","<<std::endl;
        out<<"\"time_to_label_total\": "<<t_label_max_edges+t_label_frontier_edges+t_label_seed_edges<<","<<std::endl;
        out<<"\"time_to_traversal_and_repair\": "<<t_traversal_and_repair<<","<<std::endl;
        out<<"\"time_to_traversal\": "<<t_traversal<<","<<std::endl;
        out<<"\"time_to_repair\": "<<t_repair<<","<<std::endl;
        out<<"\"time_to_generate_polygonal_mesh\": "<<t_label_max_edges + t_label_frontier_edges + t_label_seed_edges + t_traversal_and_repair<<","<<std::endl;
        out<<"\t\"memory_max_edges\": "<<m_max_edges<<","<<std::endl;
        out<<"\t\"memory_frontier_edge\": "<<m_frontier_edge<<","<<std::endl;
        out<<"\t\"memory_seed_edges\": "<<m_seed_edges<<","<<std::endl;
        out<<"\t\"memory_seed_bet_mar\": "<<m_seed_bet_mar<<","<<std::endl;
        out<<"\t\"memory_triangle_list\": "<<m_triangle_list<<","<<std::endl;
        out<<"\t\"memory_mesh_input\": "<<m_mesh_input<<","<<std::endl;
        out<<"\t\"memory_mesh_output\": "<<m_mesh_output<<","<<std::endl;
        out<<"\t\"memory_vertices_input\": "<<m_vertices_input<<","<<std::endl;
        out<<"\t\"memory_vertices_output\": "<<m_vertices_output<<","<<std::endl;
        out<<"\t\"memory_total\": "<<m_max_edges + m_frontier_edge + m_seed_edges + m_seed_bet_mar + m_triangle_list + m_mesh_input + m_mesh_output + m_vertices_input + m_vertices_output<<std::endl;
        out<<"}"<<std::endl;
        out.close();
    }




    //function whose input is a vector and print the elements of the vector
    void print_vector(std::vector<int> &vec){
        std::cout<<vec.size()<<" ";
        for (auto &v : vec){
            std::cout << v << " ";
        }
        std::cout << std::endl;
    }

    //Print ale file of the polylla mesh
    void print_ALE(std::string filename){
        std::ofstream out(filename);
        _polygon poly;
        out<<"# domain type\nCustom\n";
        out<<"# nodal coordinates: number of nodes followed by the coordinates \n";
        out<<mesh_input->vertices()<<std::endl;
        //print nodes
        for(std::size_t v = 0; v < mesh_input->vertices(); v++)
            out<<std::setprecision(15)<<mesh_input->get_PointX(v)<<" "<<mesh_input->get_PointY(v)<<std::endl; 
        out<<"# element connectivity: number of elements followed by the elements\n";
        out<<this->m_polygons<<std::endl;
        //print polygons
        int size_poly;
        int e_curr;
        for(auto &e_init : output_seeds){
            size_poly = 1;
            e_curr = mesh_output->next(e_init);
            while(e_init != e_curr){
                size_poly++;
                e_curr = mesh_output->next(e_curr);
            }
            out<<size_poly<<" ";            

            out<<mesh_output->origin(e_init)<<" ";
            e_curr = mesh_output->next(e_init);
            while(e_init != e_curr){
                out<<mesh_output->origin(e_curr)<<" ";
                e_curr = mesh_output->next(e_curr);
            }
            out<<std::endl; 
        }
        //Print borderedges
        out<<"# indices of nodes located on the Dirichlet boundary\n";
        ///Find borderedges
        int b_curr, b_init = 0;
        for(std::size_t i = mesh_input->halfEdges()-1; i != 0; i--){
            if(mesh_input->is_border_face(i)){
                b_init = i;
                break;
            }
        }
        out<<mesh_input->origin(b_init)<<" ";
        b_curr = mesh_input->prev(b_init);
        while(b_init != b_curr){
            out<<mesh_input->origin(b_curr)<<" ";
            b_curr = mesh_input->prev(b_curr);
        }
        out<<std::endl;
        out<<"# indices of nodes located on the Neumann boundary\n0\n";
        out<<"# xmin, xmax, ymin, ymax of the bounding box\n";
        double xmax = mesh_input->get_PointX(0);
        double xmin = mesh_input->get_PointX(0);
        double ymax = mesh_input->get_PointY(0);
        double ymin = mesh_input->get_PointY(0);
        //Search min and max coordinates
        for(std::size_t v = 0; v < mesh_input->vertices(); v++){
            //search range x
            if(mesh_input->get_PointX(v) > xmax )
                xmax = mesh_input->get_PointX(v);
            if(mesh_input->get_PointX(v) < xmin )
                xmin = mesh_input->get_PointX(v);
            //search range y
            if(mesh_input->get_PointY(v) > ymax )
                ymax = mesh_input->get_PointY(v);
            if(mesh_input->get_PointY(v) < ymin )
                ymin = mesh_input->get_PointY(v);
        }
        out<<xmin<<" "<<xmax<<" "<<ymin<<" "<<ymax<<std::endl;
        out.close();
    }

    //Print off file of the polylla mesh
    void print_OFF(std::string filename){
        std::ofstream out(filename);

      //  out<<"{ appearance  {+edge +face linewidth 2} LIST\n";
        out<<"OFF"<<std::endl;
        //num_vertices num_polygons 0
        out<<std::setprecision(15)<<mesh_input->vertices()<<" "<<m_polygons<<" 0"<<std::endl;
        //print nodes
        for(std::size_t v = 0; v < mesh_input->vertices(); v++)
            out<<mesh_input->get_PointX(v)<<" "<<mesh_input->get_PointY(v)<<" 0"<<std::endl; 
        //print polygons
        int size_poly;
        int e_curr;
        for(auto &e_init : output_seeds){
            size_poly = 1;
            e_curr = mesh_output->next(e_init);
            while(e_init != e_curr){
                size_poly++;
                e_curr = mesh_output->next(e_curr);
            }
            out<<size_poly<<" ";            

            out<<mesh_output->origin(e_init)<<" ";
            e_curr = mesh_output->next(e_init);
            while(e_init != e_curr){
                out<<mesh_output->origin(e_curr)<<" ";
                e_curr = mesh_output->next(e_curr);
            }
            out<<std::endl; 
        }
      //  out<<"}"<<std::endl;
        out.close();
    }

    //Print a halfedge file
    //The first line of the file is the number of halfedges
    //The rest of the lines are the halfedges with the following format:
    //origin target
    void print_hedge(std::string file_name){
        std::cout<<"Print halfedges"<<std::endl;
        std::ofstream file;
        file.open(file_name);
        int n_frontier_edges = 0;
        for(std::size_t i = 0; i < frontier_edges.size(); i++){
            if(frontier_edges[i] == true){
                n_frontier_edges++;
            }
        }
        file<<n_frontier_edges<<std::endl;
        for(std::size_t i = 0; i < mesh_input->halfEdges(); i++){
            if(frontier_edges[i] == true){
                file<<mesh_input->origin(i)<<" "<<mesh_input->target(i)<<"\n";
            }
        }
        file.close();
    }

    //Return a polygon generated from a seed edge
    _polygon generate_polygon(int e)
    {   
        _polygon poly;
        //search next frontier-edge
        int e_init = search_frontier_edge(e);
        int v_init = mesh_input->origin(e_init);
        int e_curr = mesh_input->next(e_init);
        int v_curr = mesh_input->origin(e_curr);
        poly.push_back(v_curr);
        while(e_curr != e_init && v_curr != v_init)
        {   
            e_curr = search_frontier_edge(e_curr);  
            //select triangle that contains v_curr as origin
            e_curr = mesh_input->next(e_curr);
            v_curr = mesh_input->origin(e_curr);
            poly.push_back(v_curr);
        }
        return poly;
    }


private:

    //Return true is the edge is terminal-edge or terminal border edge, 
    //but it only selects one halfedge as terminal-edge, the halfedge with lowest index is selected
    bool is_seed_edge(int e){
        int twin = mesh_input->twin(e);

        bool is_terminal_edge = (mesh_input->is_interior_face(twin) &&  (max_edges[e] && max_edges[twin]) );
        bool is_terminal_border_edge = (mesh_input->is_border_face(twin) && max_edges[e]);

        return (is_terminal_edge && e < twin ) || is_terminal_border_edge;
    }




    //Label max edges of all triangles in the triangulation
    //input: edge e indicent to a triangle t
    //output: position of edge e in max_edges[e] is labeled as true
    int label_max_edge(const int e)
    {
        //Calculates the size of each edge of a triangle 
        double dist0 = mesh_input->distance(e);
        double dist1 = mesh_input->distance(mesh_input->next(e));
        double dist2 = mesh_input->distance(mesh_input->prev(e));
        //Find the longest edge of the triangle
        if(std::max({dist0, dist1, dist2}) == dist0)
            return e;
        else if(std::max({dist0, dist1, dist2}) == dist1)
            return mesh_input->next(e);
        else
            return mesh_input->prev(e);
        return -1;
    }

 
    //Return true if the edge e is the lowest edge both triangles incident to e
    //in case of border edges, they are always labeled as frontier-edge
    bool is_frontier_edge(const int e)
    {
        int twin = mesh_input->twin(e);
        bool is_border_edge = mesh_input->is_border_face(e) || mesh_input->is_border_face(twin);
        bool is_not_max_edge = !(max_edges[e] || max_edges[twin]);
        return is_border_edge || is_not_max_edge;
    }

    //Travel in CCW order around the edges of vertex v from the edge e looking for the next frontier edge
    int search_frontier_edge(const int e)
    {
        int nxt = e;
        while(!frontier_edges[nxt])
            nxt = mesh_input->CW_edge_to_vertex(nxt);
        return nxt;
    }

    //return true if the polygon is not simple
    bool has_BarrierEdgeTip(int e_init){

        int e_curr = mesh_output->next(e_init);
        //travel inside frontier-edges of polygon
        while(e_curr != e_init){   
            //if the twin of the next halfedge is the current halfedge, then the polygon is not simple
            if( mesh_output->twin(mesh_output->next(e_curr)) == e_curr)
                return true;
            //travel to next half-edge
            e_curr = mesh_output->next(e_curr);
        }
        return false;
    }   

    //generate a polygon from a seed edge
    //input: Seed-edge
    //Output: seed frontier-edge of new popygon
    int travel_triangles(const int e)
    {   
        //search next frontier-edge
        int e_init = search_frontier_edge(e);
        //first frontier-edge is store to calculate the prev of next frontier-edfge
        int e_prev = e_init; 
        int v_init = mesh_input->origin(e_init);

        int e_curr = mesh_input->next(e_init);
        int v_curr = mesh_input->origin(e_curr);
        
        //travel inside frontier-edges of polygon
        while(e_curr != e_init && v_curr != v_init){   
            e_curr = search_frontier_edge(e_curr);

            //update next of previous frontier-edge
            mesh_output->set_next(e_prev, e_curr);  
            //update prev of current frontier-edge
            mesh_output->set_prev(e_curr, e_prev);

            //travel to next half-edge
            e_prev = e_curr;
            e_curr = mesh_input->next(e_curr);
            v_curr = mesh_input->origin(e_curr);
        }
        mesh_output->set_next(e_prev, e_init);
        mesh_output->set_prev(e_init, e_prev);
        return e_init;
    }
    
    //Given a barrier-edge tip v, return the middle edge incident to v
    //The function first calculate the degree of v - 1 and then divide it by 2, after travel to until the middle-edge
    //input: vertex v
    //output: edge incident to v
    int calculate_middle_edge(const int v){
        int frontieredge_with_bet = this->search_frontier_edge(mesh_input->edge_of_vertex(v));
        int internal_edges =mesh_input->degree(v) - 1; //internal-edges incident to v
        int adv = (internal_edges%2 == 0) ? internal_edges/2 - 1 : internal_edges/2 ;
        int nxt = mesh_input->CW_edge_to_vertex(frontieredge_with_bet);
        //back to traversing the edges of v_bet until select the middle-edge
        while (adv != 0){
            nxt = mesh_input->CW_edge_to_vertex(nxt);
            adv--;
        }
        return nxt;
    }

    //Given a seed edge e that generated polygon, split the polygon until remove al barrier-edge tips
    //input: seed edge e, polygon poly
    //output: polygon without barrier-edge tips
    void barrieredge_tip_reparation(const int e)
    {
        this->n_polygons_to_repair++;
        int x, y, i;
        int t1, t2;
        int middle_edge, v_bet;

        int e_init = e;
        int e_curr = mesh_output->next(e_init);
        //search by barrier-edge tips
        while(e_curr != e_init){   
            //if the twin of the next halfedge is the current halfedge, then the polygon is not simple
            if( mesh_output->twin(mesh_output->next(e_curr)) == e_curr){
                //std::cout<<"e_curr "<<e_curr<<" e_next "<<mesh_output->next(e_curr)<<" next del next "<<mesh_output->next(mesh_output->next(e_curr))<<" twin curr "<<mesh_output->twin(e_curr)<<" twin next "<<mesh_output->twin(mesh_output->next(e_curr))<<std::endl;

                n_barrier_edge_tips++;
                n_frontier_edges+=2;

                //select edge with bet
                v_bet = mesh_output->target(e_curr);
                middle_edge = calculate_middle_edge(v_bet);

                //middle edge that contains v_bet
                t1 = middle_edge;
                t2 = mesh_output->twin(middle_edge);
                
                //edges of middle-edge are labeled as frontier-edge
                this->frontier_edges[t1] = true;
                this->frontier_edges[t2] = true;

                //edges are use as seed edges and saves in a list
                triangle_list.push_back(t1);
                triangle_list.push_back(t2);

                seed_bet_mark[t1] = true;
                seed_bet_mark[t2] = true;
            }
                
            //travel to next half-edge
            e_curr = mesh_output->next(e_curr);
        }

        int t_curr;
        //generate polygons from seeds,
        //two seeds can generate the same polygon
        //so the bit_vector seed_bet_mark is used to label as false the edges that are already used
        int new_polygon_seed;
        while (!triangle_list.empty()){
            t_curr = triangle_list.back();
            triangle_list.pop_back();
            if(seed_bet_mark[t_curr]){
                this->n_polygons_added_after_repair++;
                seed_bet_mark[t_curr] = false;
                new_polygon_seed = generate_repaired_polygon(t_curr, seed_bet_mark);
                //Store the polygon in the as part of the mesh
                output_seeds.push_back(new_polygon_seed);
            }
        }

    }


    //Generate a polygon from a seed-edge and remove repeated seed from seed_list
    //POSIBLE BUG: el algoritmo no viaja por todos los halfedges dentro de un poligono, 
    //por lo que pueden haber semillas que no se borren y tener poligonos repetidos de output
    int generate_repaired_polygon(const int e, bit_vector &seed_list)
    {   
        int e_init = e;
        //search next frontier-edge
        while(!frontier_edges[e_init]){
            e_init = mesh_input->CW_edge_to_vertex(e_init);
            seed_list[e_init] = false; 
            //seed_list[mesh_input->twin(e_init)] = false;
        }        
        //first frontier-edge is store to calculate the prev of next frontier-edfge
        int e_prev = e_init; 
        int v_init = mesh_input->origin(e_init);

        int e_curr = mesh_input->next(e_init);
        int v_curr = mesh_input->origin(e_curr);
        seed_list[e_curr] = false;

        //travel inside frontier-edges of polygon
        while(e_curr != e_init && v_curr != v_init){   
            while(!frontier_edges[e_curr])
            {
                e_curr = mesh_input->CW_edge_to_vertex(e_curr);
                seed_list[e_curr] = false;
          //      seed_list[mesh_input->twin(e_curr)] = false;
            } 

            //update next of previous frontier-edge
            mesh_output->set_next(e_prev, e_curr);  
            //update prev of current frontier-edge
            mesh_output->set_prev(e_curr, e_prev);

            //travel to next half-edge
            e_prev = e_curr;        
            e_curr = mesh_input->next(e_curr);
            v_curr = mesh_input->origin(e_curr);
            seed_list[e_curr] = false;
            //seed_list[mesh_input->twin(e_curr)] = false;
        }
        mesh_output->set_next(e_prev, e_init);
        mesh_output->set_prev(e_init, e_prev);
        return e_init;
    }
};

#endif