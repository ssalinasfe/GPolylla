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
#include "kernel.cu"
#include <chrono>
#include <iomanip>

#include <malloc_count-0.7.1/malloc_count.h>
#include <malloc_count-0.7.1/stack_count.h>

#define PARALLEL

#define print_e(eddddge) eddddge<<" ( "<<mesh_input->origin(eddddge)<<" - "<<mesh_input->target(eddddge)<<") "


class Polylla
{
private:
    typedef std::vector<int> _polygon; 
    typedef std::vector<int> bit_vector; 


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

    // Times Device
    double t_copy_to_device_d = 0;
    double t_label_max_edges_d = 0;
    double t_label_frontier_edges_d = 0;
    double t_label_seed_edges_d = 0;
    double t_label_extra_frontier_edge_d = 0;
    double t_label_seed_scan_d = 0;
    double t_label_seed_compaction_d = 0;
    double t_traversal_and_repair_d = 0;
    double t_traversal_1_d = 0;
    double t_traversal_2_d = 0;
    double t_overwrite_seed_d = 0;
    double t_back_to_host_d = 0;

    // Times Host
    double t_label_max_edges_h = 0;
    double t_label_frontier_edges_h = 0;
    double t_label_seed_edges_h = 0;
    double t_traversal_and_repair_h = 0;
    double t_traversal_h = 0;
    double t_repair_h = 0;

    int n_halfedges;
    int n_triangle;
    int n_vertices;
    int seed_len;

    
public:

    Polylla() {}; //Default constructor

    //Constructor random data construictor
    Polylla(int size){
        this->mesh_input = new Triangulation(size);
        std::cout<<"[Polylla] Input mesh generated"<<std::endl;
        mesh_output = new Triangulation(*mesh_input);
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

        std::cout<<"[Polylla] Constructing Polylla..."<<std::endl;

        bit_vector_d *max_edges_d;
        halfEdge *halfedges_d, *halfedges_h;
        vertex *vertices_d, *vertices_h;

        max_edges = bit_vector(mesh_input->halfEdges(), 0);
        frontier_edges = bit_vector(mesh_input->halfEdges(), 0);
        seed_bet_mark = bit_vector(this->mesh_input->halfEdges(), 0);


        // copy to device and initialize

        // declare and initialize device arrays
        n_triangle = mesh_input->faces();
        n_halfedges = mesh_input->halfEdges();


        // copy halfedges to device
        halfedges_h = new halfEdge[n_halfedges];
        halfedges_h = mesh_input->HalfEdges.data();

        // copy vertices to device
        n_vertices = mesh_input->vertices();
        vertices_h = new vertex[n_vertices];
        vertices_h = mesh_input->Vertices.data();


        //CUda MAllocs
        cudaMalloc(&halfedges_d, n_halfedges*sizeof(halfEdge) );
        cudaMalloc(&vertices_d, n_vertices*sizeof(vertex) );
        cudaMalloc(&max_edges_d, n_halfedges*sizeof(bit_vector_d) );

        gpuErrchk( cudaDeviceSynchronize() ); // clean gpu timers
        auto t_start = std::chrono::high_resolution_clock::now();

        cudaMemcpy(halfedges_d, halfedges_h, n_halfedges*sizeof(halfEdge), cudaMemcpyHostToDevice );
        cudaMemcpy(vertices_d, vertices_h, n_vertices*sizeof(vertex), cudaMemcpyHostToDevice );


        cudaMemcpy(max_edges_d, max_edges.data(), max_edges.size()*sizeof(bit_vector_d), cudaMemcpyHostToDevice );

        gpuErrchk( cudaDeviceSynchronize() );
        auto t_end = std::chrono::high_resolution_clock::now();
        t_copy_to_device_d = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"[GPU] Copy vectors to device in "<<t_copy_to_device_d<<" ms"<<std::endl;
            
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Label max edges
        gpuErrchk( cudaDeviceSynchronize() );
        t_start = std::chrono::high_resolution_clock::now();
        
        // copy max edges to device
        label_edges_max_d<<<(n_triangle + BSIZE - 1)/BSIZE, BSIZE>>>(max_edges_d, vertices_d, halfedges_d, n_triangle);
        gpuErrchk( cudaDeviceSynchronize() );

        t_end = std::chrono::high_resolution_clock::now();
        t_label_max_edges_d = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"[GPU] Labered max edges in "<<t_label_max_edges_d<<" ms"<<std::endl;
       
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Label frontier edges
            

        bit_vector_d *frontier_edges_d;
        cudaMalloc(&frontier_edges_d, sizeof(bit_vector_d)*n_halfedges);
        
        gpuErrchk( cudaDeviceSynchronize() );
        t_start = std::chrono::high_resolution_clock::now();
        label_phase<<<(n_halfedges + BSIZE - 1)/BSIZE, BSIZE>>>(halfedges_d, max_edges_d, frontier_edges_d, n_halfedges); 
        cudaDeviceSynchronize();
        
        t_end = std::chrono::high_resolution_clock::now();
        t_label_frontier_edges_d = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"[GPU] Labeled frontier edges in "<<t_label_frontier_edges_d<<" ms"<<std::endl;

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Seed phase


        // GPU SEED PHASE
        half *seed_edges_ad;
        int *seed_edges_d, *seed_edges_bd;
        cudaMalloc(&seed_edges_ad, sizeof(half)*n_halfedges);
        cudaMemset(seed_edges_ad, 0, sizeof(half)*n_halfedges);
        cudaMalloc(&seed_edges_bd, sizeof(int)*n_halfedges);
        cudaMalloc(&seed_edges_d, sizeof(int)*n_halfedges);


        //dim3 blockDim(1024,1024,1);
        //dim3 gridDim((n_halfedges+1024*1024-1)/(1024*1024),1,1);

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Label seed edges 1

        gpuErrchk( cudaDeviceSynchronize() );
        t_start = std::chrono::high_resolution_clock::now();

        seed_phase_d<<<(n_halfedges + BSIZE - 1)/BSIZE, BSIZE>>>(halfedges_d, max_edges_d, seed_edges_ad, n_halfedges); 
        gpuErrchk( cudaDeviceSynchronize() );

        t_end = std::chrono::high_resolution_clock::now();
        t_label_seed_edges_d = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"[GPU] Labeled seed edges in "<<t_label_seed_edges_d<<" ms"<<std::endl;

        //call print_all_halfedges kernel
       //print_all_halfedges<<<(n_halfedges + BSIZE - 1)/BSIZE, BSIZE>>>(halfedges_d, n_halfedges);
        
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Repair phase

        // print vertex 291
        //std::cout<<"Vertex 291: "<<mesh_input->get_PointX(291)<<" "<<mesh_input->get_PointY(291)<<std::endl;

        //std::cout<<"[GPU] label extra frontier edge kernel..."<<std::endl;

        gpuErrchk( cudaDeviceSynchronize() );
        t_start = std::chrono::high_resolution_clock::now();
        
        label_extra_frontier_edge_d<<<(n_vertices + BSIZE - 1)/BSIZE,BSIZE>>>(halfedges_d, frontier_edges_d, vertices_d, seed_edges_ad, n_vertices);
        gpuErrchk( cudaDeviceSynchronize() );
//
        t_end = std::chrono::high_resolution_clock::now();
        t_label_extra_frontier_edge_d = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"[GPU] Labeled extra frontier edge in "<<t_label_extra_frontier_edge_d<<" ms"<<std::endl;

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Label seed edges 2, Scan...

        gpuErrchk( cudaDeviceSynchronize() );
        t_start = std::chrono::high_resolution_clock::now();
        scan_parallel_tc_2<int>(seed_edges_bd, seed_edges_ad, n_halfedges);
        gpuErrchk( cudaDeviceSynchronize() );

        t_end = std::chrono::high_resolution_clock::now();
        t_label_seed_scan_d = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"[GPU] Labeled seed scan in "<<t_label_seed_scan_d<<" ms"<<std::endl;

        cudaMemcpy( &seed_len, seed_edges_bd + n_halfedges - 1, sizeof(int), cudaMemcpyDeviceToHost );

        //int seed_len = scan(seed_edges_d, seed_edges_ad, n_halfedges); // ESTO SE PUEDE MEJORAR!
        //gpuErrchk( cudaDeviceSynchronize() );
        //printf ("-> %i %i %i %i\n", grid.x, grid.y, grid.z, (n_halfedges + BSIZE - 1)/BSIZE);

     

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Label seed edges 3, Compaction

        gpuErrchk( cudaDeviceSynchronize() );
        t_start = std::chrono::high_resolution_clock::now();
        compaction_d<<<(n_halfedges + BSIZE - 1)/BSIZE,BSIZE>>>(seed_edges_d, seed_edges_bd, seed_edges_ad, n_halfedges);
        gpuErrchk( cudaDeviceSynchronize() );
        //compaction_cub(seed_edges_d, d_num, max_edges_d, seed_edges_ad, n_halfedges);
        //gpuErrchk( cudaDeviceSynchronize() );
        //printf("\ndone GPU seed phase....\n\n");//*/

        /*// copy seed edges to device and print
        int *seed_edges_ah = new int[n_halfedges];
        cudaMemcpy(seed_edges_ah, seed_edges_d, n_halfedges*sizeof(int), cudaMemcpyDeviceToHost );
        for (int i = 0; i < n_halfedges; i++) {
            if ((int)seed_edges_ah[i] != 0) {
                printf("%i ", (int)seed_edges_ah[i]);
            }
        }*/
     
        t_end = std::chrono::high_resolution_clock::now();
        t_label_seed_compaction_d = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"[GPU] Labeled seed compaction in "<<t_label_seed_compaction_d<<" ms"<<std::endl;




        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Travel 

        
        int *output_seed_d;
        halfEdge *output_HalfEdges_d;
        cudaMalloc(&output_HalfEdges_d, sizeof(halfEdge)*n_halfedges);

        gpuErrchk( cudaDeviceSynchronize() );
        t_start = std::chrono::high_resolution_clock::now();
        travel_phase_d<<<(n_halfedges + BSIZE - 1)/BSIZE,BSIZE>>>(output_HalfEdges_d, halfedges_d, max_edges_d, frontier_edges_d, n_halfedges);
        gpuErrchk( cudaDeviceSynchronize() );
        
        t_end = std::chrono::high_resolution_clock::now();
        t_traversal_1_d = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"[GPU] Traversal phase 1 in "<<t_traversal_1_d<<" ms"<<std::endl;

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Repair search frontier edge

        
        cudaMalloc(&output_seed_d , sizeof(int)*seed_len);        

        dim3 blockDim2(1024,1024,1);
        dim3 gridDim2((seed_len+1024*1024-1)/(1024*1024),1,1);


        gpuErrchk( cudaDeviceSynchronize() );
        t_start = std::chrono::high_resolution_clock::now();
        search_frontier_edge_d<<<blockDim2,gridDim2>>>(output_seed_d, halfedges_d, frontier_edges_d, seed_edges_d, seed_len);
        gpuErrchk( cudaDeviceSynchronize() );

        t_end = std::chrono::high_resolution_clock::now();
        t_traversal_2_d = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"[GPU] Traversal phase (search frontier edge) in "<<t_traversal_2_d<<" ms"<<std::endl;


        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Repair overwrite seed

        gpuErrchk( cudaDeviceSynchronize() );
        t_start = std::chrono::high_resolution_clock::now();
        
        overwrite_seed_d<<<(seed_len + BSIZE - 1)/BSIZE,BSIZE>>>(output_HalfEdges_d, output_seed_d, seed_len);
        gpuErrchk( cudaDeviceSynchronize() );

        t_end = std::chrono::high_resolution_clock::now();
        t_overwrite_seed_d = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"[GPU] Rapair overwrite seed in "<<t_overwrite_seed_d<<" ms"<<std::endl;

        // Back to host
        gpuErrchk( cudaDeviceSynchronize() );
        t_start = std::chrono::high_resolution_clock::now();
        bit_vector_d *h_max_edges = new bit_vector_d[n_halfedges];
        cudaMemcpy( h_max_edges, max_edges_d, n_halfedges*sizeof(bit_vector_d), cudaMemcpyDeviceToHost );
        bit_vector_d *h_frontier_edges = new bit_vector_d[n_halfedges];
        cudaMemcpy( h_frontier_edges, frontier_edges_d, n_halfedges*sizeof(bit_vector_d), cudaMemcpyDeviceToHost );
        int *h_seed_edges = new int[n_halfedges];
        cudaMemcpy( h_seed_edges, seed_edges_d, seed_len*sizeof(int), cudaMemcpyDeviceToHost );
        int *output_seed_h = new int[seed_len];
        cudaMemcpy(output_seed_h, output_seed_d, sizeof(int)*seed_len, cudaMemcpyDeviceToHost);
        gpuErrchk( cudaDeviceSynchronize() );

        t_end = std::chrono::high_resolution_clock::now();
        t_back_to_host_d = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"[GPU] Back to host in "<<t_back_to_host_d<<" ms"<<std::endl;
         

        // standard output, time not measured
        for (int i = 0; i < n_halfedges; i++)
            max_edges[i] = h_max_edges[i];
        for (int i = 0; i < n_halfedges; i++){
            frontier_edges[i] = h_frontier_edges[i];
            //std::cout<< i <<" "<<h_frontier_edges[i]<<"\n";
        }
        //for (int i = 0; i < seed_len; i++)
        //    seed_edges[i] = h_seed_edges[i];
        //for (int i = 0; i < seed_len; i++)
        //    output_seeds[i] = output_seed_h[i];
        //std::vector<int> aux_seed_edges(h_seed_edges, h_seed_edges + seed_len);
        //std::vector<int> aux_output(output_seed_h, output_seed_h + seed_len);
        //seed_edges = aux_seed_edges;
        //output_seeds = aux_output;
        gpuErrchk( cudaDeviceSynchronize() );

        // copy output_halfeget_h to halfedges
        halfEdge *h_halfedges = new halfEdge[n_halfedges];
        cudaMemcpy(h_halfedges, output_HalfEdges_d, sizeof(halfEdge)*n_halfedges, cudaMemcpyDeviceToHost);
        gpuErrchk( cudaDeviceSynchronize() );
        for (int i = 0; i < n_halfedges; i++)
            mesh_output->HalfEdges[i] = h_halfedges[i];


        //print output_seed
        for (int i = 0; i < seed_len; i++){
            //std::cout<<output_seed_h[i]<<" ";
            output_seeds.push_back(output_seed_h[i]);
        }

        this->m_polygons = output_seeds.size();
        std::cout<<"[GPU] Mesh with "<<m_polygons<<" polygons "<<n_frontier_edges/2<<" edges and "<<n_barrier_edge_tips<<" barrier-edge tips."<<std::endl;
        //mesh_input->print_pg(std::to_string(mesh_input->vertices()) + ".pg");    


        // cudaFree(max_edges_d);
        // cudaFree(frontier_edges_d);
        // cudaFree(seed_edges_d);
        // cudaFree(halfedges_d);
        // cudaFree(vertices_d);
        
       // delete h_seed_edges;
       // delete h_frontier_edges;
       // delete h_max_edges;
       // delete output_seed_h;
       // delete halfedges_h;
       // delete vertices_h;
    }


    void print_stats(std::string filename){
        //Time
        /*std::cout<<"Time to generate Triangulation: "<<mesh_input->get_triangulation_generation_time()<<" ms"<<std::endl;
        std::cout<<"Time to copy to device: "<<t_copy_to_device_d<<" ms"<<std::endl;
        std::cout<<"Time to label max edges "<<t_label_max_edges_d<<" ms"<<std::endl;
        std::cout<<"Time to label frontier edges "<<t_label_frontier_edges_d<<" ms"<<std::endl;
        std::cout<<"Time to label seed edges "<<t_label_seed_edges_d<<" ms"<<std::endl;
        std::cout<<"Time to label total"<<t_label_max_edges_d+t_label_frontier_edges_d+t_label_seed_edges_d<<" ms"<<std::endl;
        std::cout<<"Time to traversal and repair "<<t_traversal_and_repair_d<<" ms"<<std::endl;
        std::cout<<"Time to traversal "<<t_traversal_1_d<<" ms"<<std::endl;
        std::cout<<"Time to traversal "<<t_traversal_2_d<<" ms"<<std::endl;
        std::cout<<"Time to repair "<<t_repair_d<<" ms"<<std::endl;
        std::cout<<"Time to back to host: "<<t_back_to_host_d<<" ms"<<std::endl;
        std::cout<<"Time to generate polygonal mesh "<<t_label_max_edges_d + t_label_frontier_edges_d + t_label_seed_edges_d + t_traversal_and_repair_d<<" ms"<<std::endl;//*/

       

        //Memory
        long long m_max_edges =  n_halfedges*sizeof(bit_vector_d);
        long long m_frontier_edge =  sizeof(bit_vector_d)*n_halfedges;
        long long m_seed_edges = sizeof(int)*n_halfedges + sizeof(int)*n_halfedges;
        long long m_seed_bet_mar = sizeof(half)*n_halfedges;
        long long m_triangle_list = sizeof(int)*n_triangle;
        long long m_mesh_input = mesh_input->get_size_vertex_half_edge();
        long long m_mesh_output = mesh_output->get_size_vertex_half_edge();
        long long m_vertices_input = mesh_input->get_size_vertex_struct();
        long long m_vertices_output = mesh_output->get_size_vertex_struct();

        size_t free_byte;
        size_t total_byte;
        cudaMemGetInfo(&free_byte, &total_byte);
        size_t m_memory_total = total_byte - free_byte;

        
        std::ofstream out(filename);
        std::cout<<"Printing JSON file as "<<filename<<std::endl;
        out<<"{"<<std::endl;
        out<<"\"parallel\": "<< 1 <<","<<std::endl;
        out<<"\"n_polygons\": "<<m_polygons<<","<<std::endl;
        out<<"\"n_frontier_edges\": "<<n_frontier_edges/2<<","<<std::endl;
        out<<"\"n_barrier_edge_tips\": "<<n_barrier_edge_tips<<","<<std::endl;
        out<<"\"n_half_edges\": "<<mesh_input->halfEdges()<<","<<std::endl;
        out<<"\"n_faces\": "<<mesh_input->faces()<<","<<std::endl;
        out<<"\"n_vertices\": "<<mesh_input->vertices()<<","<<std::endl;
        out<<"\"n_polygons_to_repair\": "<<n_polygons_to_repair<<","<<std::endl;
        out<<"\"n_polygons_added_after_repair\": "<<n_polygons_added_after_repair<<","<<std::endl;
        out<<"\"time_triangulation_generation\": "<<mesh_input->get_triangulation_generation_time()<<","<<std::endl;

        out<<"\"d_time_copy_to_device\": "<<t_copy_to_device_d<<","<<std::endl;
        out<<"\"d_time_to_label_max_edges\": "<<t_label_max_edges_d<<","<<std::endl;
        out<<"\"d_time_to_label_frontier_edges\": "<<t_label_frontier_edges_d<<","<<std::endl;
        out<<"\"d_time_to_label_seed_edges\": "<<t_label_seed_edges_d<<","<<std::endl;
        out<<"\"d_time_to_label_extra_frontier_edge\": "<<t_label_extra_frontier_edge_d<<","<<std::endl;
        out<<"\"d_time_to_label_scan_edges\": "<<t_label_seed_scan_d<<","<<std::endl;
        out<<"\"d_time_to_label_compaction_edges\": "<<t_label_seed_compaction_d<<","<<std::endl;
        out<<"\"d_time_to_label_total\": "<<t_label_max_edges_d+t_label_frontier_edges_d+t_label_seed_edges_d<<","<<std::endl;
        out<<"\"d_time_to_traversal_and_repair\": "<<t_traversal_and_repair_d<<","<<std::endl;
        out<<"\"d_time_to_traversal\": "<<t_traversal_1_d<<","<<std::endl;
        out<<"\"d_time_to_traversal_search_frontier_edge\": "<<t_traversal_2_d<<","<<std::endl;
        out<<"\"d_time_to_back_to_host\": "<<t_back_to_host_d<<","<<std::endl;
        out<<"\"d_time_to_overwrite_seed\": "<<t_overwrite_seed_d<<","<<std::endl;
        out<<"\"d_time_to_generate_polygonal_mesh\": "<<t_label_max_edges_d + t_label_frontier_edges_d + t_label_seed_edges_d + t_traversal_and_repair_d<<","<<std::endl;
        
        out<<"\t\"memory_max_edges\": "<<m_max_edges<<","<<std::endl;
        out<<"\t\"memory_frontier_edge\": "<<m_frontier_edge<<","<<std::endl;
        out<<"\t\"memory_seed_edges\": "<<m_seed_edges<<","<<std::endl;
        out<<"\t\"memory_seed_bet_mar\": "<<m_seed_bet_mar<<","<<std::endl;
        out<<"\t\"memory_triangle_list\": "<<m_triangle_list<<","<<std::endl;
        out<<"\t\"memory_mesh_input\": "<<m_mesh_input<<","<<std::endl;
        out<<"\t\"memory_mesh_output\": "<<m_mesh_output<<","<<std::endl;
        out<<"\t\"memory_vertices_input\": "<<m_vertices_input<<","<<std::endl;
        out<<"\t\"memory_vertices_output\": "<<m_vertices_output<<","<<std::endl;
        out<<"\t\"memory_total\": "<<m_max_edges + m_frontier_edge + m_seed_edges + m_seed_bet_mar + m_triangle_list + m_mesh_input + m_mesh_output + m_vertices_input + m_vertices_output<<","<<std::endl;
        out<<"\t\"memory_total_gpu\": "<<m_memory_total<<std::endl;
        //out<<"\t\"memory_total_cpu\": "<<malloc_count()<<std::endl;
        out<<"}"<<std::endl;
        out.close();
    }



    //Print off file of the polylla mesh
    void print_OFF(std::string filename){
        std::ofstream out(filename);

        std::cout << "Printing OFF file" <<  mesh_input->vertices() << " " << m_polygons << std::endl;

      //  out<<"{ appearance  {+edge +face linewidth 2} LIST\n";
        out<<"OFF"<<std::endl;
        //num_vertices num_polygons 0
        out<<std::setprecision(15)<<mesh_input->vertices()<<" "<<m_polygons<<" 0"<<std::endl;
        //print nodes
        for(std::size_t v = 0; v < mesh_input->vertices(); v++)
            out<<mesh_input->get_PointX(v)<<" "<<mesh_input->get_PointY(v)<<" 0"<<std::endl; 
        //print polygons
        //printf("-------> 1\n");
        int size_poly;
        int e_curr;
        //std::cout<<"aca"<<std::endl;
        for(auto &e_init : output_seeds){
            size_poly = 1;
            e_curr = mesh_output->next(e_init);
            //std::cout<<"poly"<<"e_init"<<e_init<<std::endl;
            while(e_init != e_curr){
              //  std::cout<<"e_init "<<e_init<<" e_curr "<<e_curr<<std::endl;
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
            //count++;
            //printf("-------> 1.1 %d\n", count);
        }
        //printf("-------> 2\n");
      //  out<<"}"<<std::endl;
        out.close();
    }
};

#endif