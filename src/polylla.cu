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

typedef std::vector<int> _polygon; 
typedef std::vector<char> bit_vector; 

#include <assert.h>
#include <cub/cub.cuh> 
#define BSIZE 1024

__device__ int twin_d(halfEdge *HalfEdges, int e)
{
    return HalfEdges[e].twin;
}

__device__ int next_d(halfEdge *HalfEdges, int e)
{
    return HalfEdges[e].next;
}

__device__ int prev_d(halfEdge *HalfEdges, int e)
{
    return HalfEdges[e].prev;
}

__device__ bool is_border_face_d(halfEdge *HalfEdges, int e)
{
    return HalfEdges[e].is_border;
}

__device__ bool is_interior_face_d(halfEdge *HalfEdges, int e)
{
   return !is_border_face_d(HalfEdges, e);
}

__device__ int origin_d(halfEdge *HalfEdges, int e)
{
    return HalfEdges[e].origin;
}

__device__ int target_d(halfEdge *HalfEdges, int e)
{
    return HalfEdges[e].target;
}

    // Calculates the distante of edge e
__device__ double distance_d(halfEdge *HalfEdges, vertex *Vertices, int e){
        double x1 = Vertices[origin_d(HalfEdges, e)].x;
        double y1 = Vertices[origin_d(HalfEdges, e)].y;
        double x2 = Vertices[target_d(HalfEdges, e)].x;
        double y2 = Vertices[target_d(HalfEdges, e)].y;
        return powf(x1-x2,2) + powf(y1-y2,2);
    }

    __device__ int edges_max_d(halfEdge *HalfEdges, short max, int e){
        int init_vertex = origin_d(HalfEdges, e);
        int curr_vertex = -1;
        int nxt = e;
        while (curr_vertex != init_vertex){
            nxt = next_d(HalfEdges,nxt);
            curr_vertex = origin_d(HalfEdges,nxt);
            if(max == 0 && curr_vertex == origin_d(HalfEdges,e)){
                return nxt;
            }else if(max == 1 && curr_vertex == origin_d(HalfEdges,next_d(HalfEdges,e))){
                return nxt;
            }else if(max == 2 && curr_vertex == origin_d(HalfEdges,next_d(HalfEdges,next_d(HalfEdges,e)))){
                return nxt;
            }          
        }
        return -1;    
    }
    
__device__ short max_edge_kernel(halfEdge *HalfEdges, vertex *Vertices, int off){
    double dist0 = distance_d(HalfEdges, Vertices, off); //min
    double dist1 = distance_d(HalfEdges, Vertices, next_d(HalfEdges, off)); //mid
    double dist2 = distance_d(HalfEdges, Vertices, next_d(HalfEdges, next_d(HalfEdges, off))); //max

    short max;
    //Find the longest edge of the triangle
    if((dist0 >= dist1 && dist1 >= dist2) || (dist0 >= dist2 && dist2 >= dist1)){
        max = 0; //edge face[0]-face[1] is max
    }else if( (dist1 >= dist0 && dist0 >= dist2) || (dist1 >= dist2 && dist2 >= dist0)){
        max = 1; //edge face[1]-face[2] is max
    }else if( (dist2 >= dist1 && dist1 >= dist0) || (dist2 >= dist0 && dist0 >= dist1)){
        max = 2; //edge face[2]-face[0] is max
    }
    return max;
}

__global__ void label_edges_max_d(char *output, vertex *Vertices, halfEdge *HalfEdges, int n)
{
    int off = (blockIdx.x * blockDim.x + threadIdx.x);
    if(off < n)
    {
        short max = max_edge_kernel(HalfEdges, Vertices, off*3);
        //output[off] = max;
        output[off] = 0;
        output[edges_max_d(HalfEdges, max, off*3)] = 1;
    }
}

__device__ bool is_frontier_edge_d(halfEdge *HalfEdges, char *max_edges, const int e)
{
    int twin = twin_d(HalfEdges, e);
    bool is_border_edge = is_border_face_d(HalfEdges, e) || is_border_face_d(HalfEdges, twin);
    bool is_not_max_edge = !(max_edges[e] || max_edges[twin]);
    if(is_border_edge || is_not_max_edge)
        return true;
    else
        return false;
}

__global__ void label_phase(halfEdge *HalfEdges, char *max_edges, char *frontier_edges, int n){
    int off = threadIdx.x + blockDim.x*blockIdx.x;
    if (off < n){
        frontier_edges[off] = false;
        if(is_frontier_edge_d(HalfEdges, max_edges, off))
            frontier_edges[off] = true;
        }
    //printf("\nhola mundo  %i\n\n",(int)a[threadIdx.x].origin);
}//*/

__device__ bool is_seed_edge_d(halfEdge *HalfEdges, char *max_edges, int e){
    int twin = twin_d(HalfEdges, e);

    bool is_terminal_edge = (is_interior_face_d(HalfEdges, twin) &&  (max_edges[e] && max_edges[twin]) );
    bool is_terminal_border_edge = (is_border_face_d(HalfEdges, twin) && max_edges[e]);

    if( (is_terminal_edge && e < twin ) || is_terminal_border_edge){
        return true;
    }

    return false;
}

__global__ void seed_phase_d(halfEdge *HalfEdges, char *max_edges, int *seed_edges, int n){
    int off = threadIdx.x + blockDim.x*blockIdx.x;
    if (off < n){
        //seed_edges[off] = 0;
        if(is_interior_face_d(HalfEdges, off) && is_seed_edge_d(HalfEdges, max_edges, off))
            seed_edges[off] = 1;
        }
    //printf("\nhola mundo  %i\n\n",(int)a[threadIdx.x].origin);
}//*/

__global__ void compaction_d(int *output, int *input, int *condition, int n){
    int off = threadIdx.x + blockDim.x*blockIdx.x;
    //printf("hola %i %i %i\n", off, input[off], condition[off]);
    if (off < n){
        if (condition[off] == 1)
            output[input[off]] = off;//*/
        //printf("hola %i %i %i %i\n", off, output[input[off]], input[off], condition[off]);
    }
}

int scan(int *d_out, int *d_in, int num_items){
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
    int *len = new int[1];
    cudaMemcpy(len, d_out+num_items-1, sizeof(int), cudaMemcpyDeviceToHost);
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

//Travel in CCW order around the edges of vertex v from the edge e looking for the next frontier edge
__global__ void search_frontier_edge_d(int *output, halfEdge *HalfEdges, char *frontier_edges, int *seed_edges, int n)
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

__device__ int search_next_frontier_edge_d(halfEdge *HalfEdges, char *frontier_edges, const int e)
{
    int nxt = e;
    while(!frontier_edges[nxt])
    {
        nxt = CW_edge_to_vertex_d(HalfEdges, nxt);
    }  
    return nxt;
}

__device__ int search_prev_frontier_edge_d(halfEdge *HalfEdges, char *frontier_edges, const int e)
{
    int prv = e;
    while(!frontier_edges[prv])
    {
        prv = CCW_edge_to_vertex_d(HalfEdges, prv);
    }  
    return prv;
}


__global__ void travel_phase_d(halfEdge *output, halfEdge *HalfEdges, char *max_edges, char *frontier_edges, int n){
    int off = threadIdx.x + blockIdx.x*blockDim.x;
    if (off < n){
        output[off] = HalfEdges[off];
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

/*__global__ void kernel(std::vector<halfEdge> a){
    printf("\nhola mundo  %i\n\n",a.at(0).face);
}//*/

struct Polygon{
    int seed_edge; //Edge that generate the polygon
    std::vector<int> vertices; //Vertices of the polygon
    //std::vector<int> neighbors; //Neighbors of the polygon WIP
};

class Polylla
{
    public:
//private:

    Triangulation *mesh_input; // Halfedge triangulation
    Triangulation *mesh_output;
    std::vector<int> output_seeds; //Seeds of the polygon

    std::vector<Polygon> polygonal_mesh; //Vector of polygons generated by polygon
    std::vector<int> triangles; //True if the edge generated a triangle CHANGE!!!!

    bit_vector max_edges; //True if the edge i is a max edge
    bit_vector frontier_edges; //True if the edge i is a frontier edge
    std::vector<int> seed_edges; //Seed edges that generate polygon simple and non-simple

    int m_polygons = 0; //Number of polygons
    int n_frontier_edges = 0; //Number of frontier edges
    int n_barrier_edge_tips = 0; //Number of barrier edge tips
    
//public:

    Polylla() {}; //Default constructor

    //Constructor from a OFF file
    Polylla(std::string off_file){
        //std::cout<<"Generating Triangulization..."<<std::endl;
        auto t_start = std::chrono::high_resolution_clock::now();
        this->mesh_input = new Triangulation(off_file);
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"Triangulation generated "<<elapsed_time_ms<<" ms"<<std::endl;

        //call copy constructor
        mesh_output = new Triangulation(*mesh_input);
        construct_Polylla();
    }

    //Constructor from a node_file, ele_file and neigh_file
    Polylla(std::string node_file, std::string ele_file, std::string neigh_file){
        //std::cout<<"Generating Triangulization..."<<std::endl;
        auto t_start = std::chrono::high_resolution_clock::now();
        this->mesh_input = new Triangulation(node_file, ele_file, neigh_file);
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"Triangulation generated "<<elapsed_time_ms<<" ms"<<std::endl;

        //call copy constructor
        mesh_output = new Triangulation(*mesh_input);
        construct_Polylla();
    }

    ~Polylla() {
        delete mesh_input;
        delete mesh_output;
    }

    void construct_Polylla(){


        max_edges = bit_vector(mesh_input->halfEdges(), false);
        frontier_edges = bit_vector(mesh_input->halfEdges(), false);
        //seed_edges = bit_vector(mesh_input->halfEdges(), false);
        triangles = mesh_input->get_Triangles(); //Change by triangle list

        //DEVICE VARIABLES
        int n_triangle = triangles.size();
        int n = mesh_input->n_halfedges;

        char *max_edges_d; 
        vertex *vertices_d;
        halfEdge *HalfEdges_h = mesh_input->HalfEdges.data(); 
        halfEdge *HalfEdges_d;
        char *frontier_edges_d;
        char *frontier_edges_h = new char[n];
        int *seed_edges_ad;
        int *seed_edges_d;
        halfEdge *output_HalfEdges_d;
        int *output_seed_h;        
        int *output_seed_d;
       
        gpuErrchk( cudaDeviceSynchronize() );


        // DEFINE GRID AND BLOCK SIZE
        dim3 block, grid;
        block = dim3(BSIZE, 1, 1);    
        grid = dim3((n_triangle + BSIZE - 1)/BSIZE, 1, 1);  


        clock_t t = clock();
        auto t_start = std::chrono::high_resolution_clock::now();

        //MALLOC DEVICE MEMORY
        cudaMalloc(&max_edges_d, sizeof(char)*n);
        cudaMalloc(&HalfEdges_d, sizeof(halfEdge)*n);
        cudaMemcpy(HalfEdges_d, HalfEdges_h, sizeof(halfEdge)*n, cudaMemcpyHostToDevice); 
        cudaMalloc(&vertices_d, sizeof(vertex)*mesh_input->Vertices.size());
        cudaMemcpy(vertices_d, mesh_input->Vertices.data(), sizeof(vertex)*mesh_input->Vertices.size(), cudaMemcpyHostToDevice);
        cudaMalloc(&frontier_edges_d, sizeof(char)*n); 
        cudaMalloc(&seed_edges_ad, sizeof(int)*n);
        cudaMalloc(&seed_edges_d, sizeof(int)*n);
        cudaMalloc(&output_HalfEdges_d, sizeof(halfEdge)*n);
        //cudaMemcpy(output_HalfEdges_d, HalfEdges_h, sizeof(halfEdge)*n, cudaMemcpyHostToDevice);
       

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"Copy to device in "<<elapsed_time_ms<<" ms"<<std::endl;

        //Label max edges of each triangle
        //for (size_t t = 0; t < mesh_input->faces(); t++){

        
        t_start = std::chrono::high_resolution_clock::now();
        /*for(auto &t : triangles)
            max_edges[label_max_edge(t)] = true;   //*/

        
        // COPY HALFEDGES TO DEVICE

        // GPU LABEL MAX EDGES
        //char *max_edges_h = new char[n];
        //max_edges_h = max_edges.data();
        label_edges_max_d<<<grid, block>>>(max_edges_d, vertices_d, HalfEdges_d, n_triangle);
        gpuErrchk( cudaDeviceSynchronize() );//*/
        //cudaMemcpy(max_edges_d, max_edges.data(), sizeof(char)*mesh_input->n_halfedges, cudaMemcpyHostToDevice);  

        //cudaMemcpy(max_edges_h, max_edges_d, sizeof(char)*n, cudaMemcpyDeviceToHost);
        /*//check max edges...
        for (uint i = 0; i < n_triangle; i++) {
            printf("%i %i %i  ->  %i\n",i,max_edges_h[i],max_edges[i],triangles[i]);
            assert(max_edges_h[i] == max_edges[i]);
        }
        printf("-----> test passed GPU label max edges....\n"); //*/
        //cudaMemcpy(max_edges_d, max_edges.data(), sizeof(char)*n, cudaMemcpyHostToDevice);  


        t_end = std::chrono::high_resolution_clock::now();
        elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"Labered max edges in "<<elapsed_time_ms<<" ms"<<std::endl;

        block = dim3(BSIZE, 1, 1);    
        grid = dim3((n + BSIZE - 1)/BSIZE, 1, 1);    

        t_start = std::chrono::high_resolution_clock::now();

        // GPU VERSION!!!! IT's BETTER THAN CPU
        label_phase<<<grid,block>>>(HalfEdges_d, max_edges_d, frontier_edges_d, n); 
        gpuErrchk( cudaDeviceSynchronize() );//*/


        /*for (std::size_t e = 0; e < mesh_input->halfEdges(); e++){
            if(is_frontier_edge(e)){
                frontier_edges[e] = true;
                n_frontier_edges++;
            }
        }//*/
        //printf("\ndone GPU label phase....\n\n");

        /*cudaMemcpy(frontier_edges_h, frontier_edges_d, sizeof(char)*n, cudaMemcpyDeviceToHost);
        gpuErrchk( cudaDeviceSynchronize() );
        for (uint i = 0; i < n; i++) {
            //printf("%i %i %i\n",i,frontier_edges[i],frontier_edges_h[i]);
            assert(frontier_edges[i] == frontier_edges_h[i]);
        }
        printf("-----> test passed GPU label phase....\n"); //*/
        // */
       
        t_end = std::chrono::high_resolution_clock::now();
        elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"Labeled frontier edges in "<<elapsed_time_ms<<" ms"<<std::endl;
        
        // GPU SEED PHASE
        seed_phase_d<<<grid,block>>>(HalfEdges_d, max_edges_d, seed_edges_ad, n); 
        //kernel<<<1,1>>>();
        gpuErrchk( cudaDeviceSynchronize() );
        int seed_len = scan(seed_edges_d, seed_edges_ad, n); // ESTO SE PUEDE MEJORAR!
        //gpuErrchk( cudaDeviceSynchronize() );
        compaction_d<<<(n + BSIZE - 1)/BSIZE,BSIZE>>>(seed_edges_d, seed_edges_d, seed_edges_ad, n);
        gpuErrchk( cudaDeviceSynchronize() );
        //printf("\ndone GPU seed phase....\n\n");//*/

        t_start = std::chrono::high_resolution_clock::now();
        /*//label seeds edges,
        for (std::size_t e = 0; e < mesh_input->halfEdges(); e++)
            if(mesh_input->is_interior_face(e) && is_seed_edge(e))
                seed_edges.push_back(e);
        
        printf("\n---------------> %i     %i\n\n",seed_len,seed_edges.size());//*/
        t_end = std::chrono::high_resolution_clock::now();
        elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"Labeled seed edges in "<<elapsed_time_ms<<" ms"<<std::endl;

        /*int *seed_edges_h = new int[n];
        cudaMemcpy(seed_edges_h, seed_edges_d, sizeof(int)*seed_edges.size(), cudaMemcpyDeviceToHost);
        for (uint i = 0; i < seed_edges.size(); i++) {
            //printf("%i %i %i\n",i,seed_edges_h[i],seed_edges[i]);
            assert(seed_edges_h[i] == seed_edges[i]);
        }
        printf("-----> test passed GPU seed phase....\n"); //*/


        //Travel phase: Generate polygon mesh
        //int polygon_seed;
        //Foreach seed edge generate polygon
        t_start = std::chrono::high_resolution_clock::now();
        /*for(auto &e : seed_edges){
            polygon_seed = travel_triangles(e);

            //if(!has_BarrierEdgeTip(polygon_seed)){ //If the polygon is a simple polygon then is part of the mesh
                output_seeds.push_back(polygon_seed);
           /* }else{ //Else, the polygon is send to reparation phase
                barrieredge_tip_reparation(polygon_seed);
            }*/         
        //}    

        // TRAVEL PHASE IN GPU!!!!
        travel_phase_d<<<grid,block>>>(output_HalfEdges_d, HalfEdges_d, max_edges_d, frontier_edges_d, n);
        //gpuErrchk( cudaDeviceSynchronize() );
        cudaMalloc(&output_seed_d , sizeof(int)*seed_len);
        search_frontier_edge_d<<<(seed_len+BSIZE-1)/BSIZE,block>>>(output_seed_d, HalfEdges_d, frontier_edges_d, seed_edges_d, seed_len);
        gpuErrchk( cudaDeviceSynchronize() );



        t_end = std::chrono::high_resolution_clock::now();
        elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"Polygons generated/repaired in "<<elapsed_time_ms<<" ms"<<std::endl;
        
        t_start = std::chrono::high_resolution_clock::now();
        // COPY OUTPUT_SEED TO HOST
        output_seed_h = new int[seed_len];
        cudaMemcpy(output_seed_h, output_seed_d, sizeof(int)*seed_len, cudaMemcpyDeviceToHost);
        /*for (uint i = 0; i < seed_len; i++) {
            printf("%i %i %i\n",i,output_seed_h[i],output_seeds[i]);
            //assert(output_seed_h[i] == output_seeds[i]);
        }
        printf("-----> test passed GPU seed output ....\n"); //*/
        std::vector<int> nums(output_seed_h,output_seed_h+seed_len);

        //printf("-----------------------> %i %i %i\n",seed_len,output_seeds.size(),nums.size());
        output_seeds = nums;


        this->m_polygons = output_seeds.size(); // SEED_LEN

        // COPY HALFEDGES TO HOST
        halfEdge *output_HalfEdges_h = new halfEdge[n];
        cudaMemcpy(output_HalfEdges_h, output_HalfEdges_d, sizeof(halfEdge)*n, cudaMemcpyDeviceToHost);
        /*for (uint i = 0; i < n; i++) {
            printf("%i %i %i %i\n",i,(int)output_HalfEdges_h[i].next,(int)mesh_output->HalfEdges[i].next,(int)mesh_input->HalfEdges[i].next);
            //assert(output_HalfEdges_h[i].prev == mesh_output->HalfEdges[i].prev);
        }
        printf("-----> test passed GPU travel phase ....\n"); //*/
        std::vector<halfEdge> travel_he(output_HalfEdges_h,output_HalfEdges_h+n);
        mesh_output->HalfEdges = travel_he;
        t_end = std::chrono::high_resolution_clock::now();
        elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"Copy to device in "<<elapsed_time_ms<<" ms"<<std::endl;


        std::cout<<"Mesh with "<<m_polygons<<" polygons "<<n_frontier_edges/2<<" edges and "<<n_barrier_edge_tips<<" barrier-edge tips."<<std::endl;
        //mesh_input->print_pg(std::to_string(mesh_input->vertices()) + ".pg");             
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
        for(auto &i : this->polygonal_mesh){
            out<<i.vertices.size()<<" ";
            for(auto &v : i.vertices){
                out<<v + 1<<" ";
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
        out<<mesh_input->origin(b_init) + 1<<" ";
        b_curr = mesh_input->prev(b_init);
        while(b_init != b_curr){
            out<<mesh_input->origin(b_curr) + 1<<" ";
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

        out<<"{ appearance  {+edge +face linewidth 2} LIST\n";
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
        out<<"}"<<std::endl;
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


//private:

    //Return true is the edge is terminal-edge or terminal border edge, 
    //but it only selects one halfedge as terminal-edge, the halfedge with lowest index is selected
    bool is_seed_edge(int e){
        int twin = mesh_input->twin(e);

        bool is_terminal_edge = (mesh_input->is_interior_face(twin) &&  (max_edges[e] && max_edges[twin]) );
        bool is_terminal_border_edge = (mesh_input->is_border_face(twin) && max_edges[e]);

        if( (is_terminal_edge && e < twin ) || is_terminal_border_edge){
            return true;
        }

        return false;
    }




    //Label max edges of all triangles in the triangulation
    //input: edge e indicent to a triangle t
    //output: position of edge e in max_edges[e] is labeled as true
    int label_max_edge(const int e)
    {
        //Calculates the size of each edge of a triangle 

        double dist0 = mesh_input->distance(e); //min
        double dist1 = mesh_input->distance(mesh_input->next(e)); //mid
        double dist2 = mesh_input->distance(mesh_input->next(mesh_input->next(e))); //max

        short max;
        //Find the longest edge of the triangle
        if((dist0 >= dist1 && dist1 >= dist2) || (dist0 >= dist2 && dist2 >= dist1)){
            max = 0; //edge face[0]-face[1] is max
        }else if( (dist1 >= dist0 && dist0 >= dist2) || (dist1 >= dist2 && dist2 >= dist0)){
            max = 1; //edge face[1]-face[2] is max
        }else if( (dist2 >= dist1 && dist1 >= dist0) || (dist2 >= dist0 && dist0 >= dist1)){
            max = 2; //edge face[2]-face[0] is max
        }else{
            std::cout<<"ERROR: max edge not found"<<std::endl;
            exit(0);
        }
        int init_vertex = mesh_input->origin(e);
        int curr_vertex = -1;
        int nxt = e;
        // Return the index of the edge with the longest edge
        while (curr_vertex != init_vertex){
            nxt = mesh_input->next(nxt);
            curr_vertex = mesh_input->origin(nxt);
            if(max == 0 && curr_vertex == mesh_input->origin(e)){
                return nxt;
            }else if(max == 1 && curr_vertex == mesh_input->origin(mesh_input->next(e))){
                return nxt;
            }else if(max == 2 && curr_vertex == mesh_input->origin(mesh_input->next(mesh_input->next(e)))){
                return nxt;
            }          
        }
        return -1;
    }

 
    //Return true if the edge e is the lowest edge both triangles incident to e
    //in case of border edges, they are always labeled as frontier-edge
    bool is_frontier_edge(const int e)
    {
        int twin = mesh_input->twin(e);
        bool is_border_edge = mesh_input->is_border_face(e) || mesh_input->is_border_face(twin);
        bool is_not_max_edge = !(max_edges[e] || max_edges[twin]);
        if(is_border_edge || is_not_max_edge)
            return true;
        else
            return false;
    }

    //Travel in CCW order around the edges of vertex v from the edge e looking for the next frontier edge
    int search_frontier_edge(const int e)
    {
        int nxt = e;
        while(!frontier_edges[nxt])
        {
            nxt = mesh_input->CW_edge_to_vertex(nxt);
        }  
        return nxt;
    }

    //return true if the polygon is not simple
    bool has_BarrierEdgeTip(int e_init){

        int e_curr = mesh_output->next(e_init);
        
        //travel inside frontier-edges of polygon
        while(e_curr != e_init)
        {   
            //if the twin of the next halfedge is the current halfedge, then the polygon is not simple
            if( mesh_output->twin(mesh_output->next(e_curr)) == e_curr){
                return true;
            }
            
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
        while(e_curr != e_init && v_curr != v_init)
        {   
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
    //output: middle edge incident to v
    int search_middle_edge(const int v_bet, const int e_curr)
    {
        
        //select frontier-edge of barrier-edge tip
        int frontieredge_with_bet = mesh_input->twin(e_curr);
        int nxt = mesh_input->CW_edge_to_vertex(frontieredge_with_bet);
        int adv = 1; 
        //calculates the degree of v_bet
        while (nxt != frontieredge_with_bet)
        {
            nxt = mesh_input->CW_edge_to_vertex(nxt);
            adv++;
        }
        adv--; //last edge visited is the same with the frontier-edge so it is not counted
        if(adv%2 == 0){ //if the triangles surrounding the BET are even 
            adv = adv/2 - 1;
        }else{   
            //if the triangles surrounding the BET are odd, edges are even
            //Choose any edge of the triangle in the middle; prov is choose due to this always exists
            adv = adv/2;
        }   
        //back to traversing the edges of v_bet until select the middle-edge
        nxt = mesh_input->CW_edge_to_vertex(frontieredge_with_bet);
        //adv--;
        while (adv != 0)
        {
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
        int t1, t2;
        int middle_edge, v_bet;

        std::vector<int> triangle_list;
        bit_vector seed_bet_mark(this->mesh_input->halfEdges(), false);

        int e_init = e;
        int e_curr = mesh_output->next(e_init);
        //search by barrier-edge tips
        while(e_curr != e_init)
        {   
            //if the twin of the next halfedge is the current halfedge, then the polygon is not simple
            if( mesh_output->twin(mesh_output->next(e_curr)) == e_curr){
                //std::cout<<"e_curr "<<e_curr<<" e_next "<<mesh_output->next(e_curr)<<" next del next "<<mesh_output->next(mesh_output->next(e_curr))<<" twin curr "<<mesh_output->twin(e_curr)<<" twin next "<<mesh_output->twin(mesh_output->next(e_curr))<<std::endl;

                n_barrier_edge_tips++;
                n_frontier_edges+=2;

                //select edge with bet
                v_bet = mesh_output->target(e_curr);
                middle_edge = search_middle_edge(v_bet, e_curr);

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
        while (!triangle_list.empty())
        {
            t_curr = triangle_list.back();
            triangle_list.pop_back();
            if(seed_bet_mark[t_curr]){
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
        while(!frontier_edges[e_init])
        {
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
        while(e_curr != e_init && v_curr != v_init)
        {   
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