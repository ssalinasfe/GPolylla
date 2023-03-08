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
    double t_traversal_and_repair_d = 0;
    double t_traversal_1_d = 0;
    double t_traversal_2_d = 0;
    double t_repair_d = 0;
    double t_back_to_host_d = 0;

    // Times Host
    double t_label_max_edges_h = 0;
    double t_label_frontier_edges_h = 0;
    double t_label_seed_edges_h = 0;
    double t_traversal_and_repair_h = 0;
    double t_traversal_h = 0;
    double t_repair_h = 0;
    
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

        max_edges = bit_vector(mesh_input->halfEdges(), 0);
        frontier_edges = bit_vector(mesh_input->halfEdges(), 0);
        //triangles = mesh_input->get_Triangles(); //Change by triangle list
        seed_bet_mark = bit_vector(this->mesh_input->halfEdges(), 0);

        //terminal_edges = bit_vector(mesh_input->halfEdges(), false);
        //seed_edges = bit_vector(mesh_input->halfEdges(), false);
        
        // copy to device and initialize


        gpuErrchk( cudaDeviceSynchronize() ); // clean gpu timers
        auto t_start = std::chrono::high_resolution_clock::now();

        // declare and initialize device arrays
        //int *d_triangles;
        int n_triangle = mesh_input->faces();
        // copy halfedges to device
        int n_halfedges = mesh_input->halfEdges();
        halfEdge *halfedges_d, *halfedges_h = new halfEdge[n_halfedges];
        halfedges_h = mesh_input->HalfEdges.data();
        cudaMalloc(&halfedges_d, n_halfedges*sizeof(halfEdge) );
        cudaMemcpy(halfedges_d, halfedges_h, n_halfedges*sizeof(halfEdge), cudaMemcpyHostToDevice );

        // copy vertices to device
        int n_vertices = mesh_input->vertices();
        vertex *vertices_d, *vertices_h = new vertex[n_vertices];
        vertices_h = mesh_input->Vertices.data();
        cudaMalloc(&vertices_d, n_vertices*sizeof(vertex) );
        cudaMemcpy(vertices_d, vertices_h, n_vertices*sizeof(vertex), cudaMemcpyHostToDevice );

        bit_vector_d *max_edges_d;
        cudaMalloc(&max_edges_d, n_halfedges*sizeof(bit_vector_d) );
        cudaMemset(max_edges_d, 0, n_halfedges*sizeof(bit_vector_d));

        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        //std::cout<<"[GPU] Copy vectors to device in "<<elapsed_time_ms<<" ms"<<std::endl;
        t_copy_to_device_d = elapsed_time_ms;
            
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Label max edges
        
        t_start = std::chrono::high_resolution_clock::now();

        /*std::vector<int> sort_max_edge(n_halfedges);
        for(int i = 0; i < n_halfedges; i++){
            sort_max_edge[i] = i;
        }
        // sort vector of halfedges
        std::sort(sort_max_edge.begin(), sort_max_edge.end(), [=](int a, int b) { return mesh_input->distance(a) < mesh_input->distance(b); });

        // print sorted vector

        std::vector<int> pos_sort_edge(n_halfedges);
        for(int i = 0; i < n_halfedges; i++){
            pos_sort_edge[sort_max_edge[i]] = i;
        }


        //dado un triangulo comapra aristas
        for (int i = 0; i < n_triangle; i++){
            // get halfedges of triangle
            int e1 = 3*i;
            int e2 = mesh_input->next(e1);
            int e3 = mesh_input->prev(e1);

            //compare e1, e2, e3 using vector sort_max_edge
            //int e1_pos = std::find(sort_max_edge.begin(), sort_max_edge.end(), e1) - sort_max_edge.begin();
            //int e2_pos = std::find(sort_max_edge.begin(), sort_max_edge.end(), e2) - sort_max_edge.begin();
            //int e3_pos = std::find(sort_max_edge.begin(), sort_max_edge.end(), e3) - sort_max_edge.begin();

            int e1_pos = pos_sort_edge[e1];
            int e2_pos = pos_sort_edge[e2];
            int e3_pos = pos_sort_edge[e3];
            


            int max_e = std::max({e1_pos, e2_pos, e3_pos});            
            //max_edges[max_e] = 1;
            max_edges[sort_max_edge[max_e]] = 1;
            //std::cout<<"max edge: "<<max_e<<std::endl;
        }
        
        // copy max edges to device
        cudaMemcpy(max_edges_d, max_edges.data(), max_edges.size()*sizeof(bit_vector_d), cudaMemcpyHostToDevice );*/

       // kernel<<<(max_edges.size()+1023)/1024, 1024>>>(max_edges_d, max_edges.size());

        label_edges_max_d<<<(n_triangle + BSIZE - 1)/BSIZE, BSIZE>>>(max_edges_d, vertices_d, halfedges_d, n_triangle);
        gpuErrchk( cudaDeviceSynchronize() );

        t_end = std::chrono::high_resolution_clock::now();
        elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        //std::cout<<"[GPU] Labered max edges in "<<elapsed_time_ms<<" ms"<<std::endl;
        t_label_max_edges_d = elapsed_time_ms;
       
        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Label frontier edges
            
        t_start = std::chrono::high_resolution_clock::now();

        bit_vector_d *frontier_edges_d;
        cudaMalloc(&frontier_edges_d, sizeof(bit_vector_d)*n_halfedges);
        label_phase<<<(n_halfedges + BSIZE - 1)/BSIZE, BSIZE>>>(halfedges_d, max_edges_d, frontier_edges_d, n_halfedges); 
        cudaDeviceSynchronize();
        
        t_end = std::chrono::high_resolution_clock::now();
        elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        //std::cout<<"[GPU] Labeled frontier edges in "<<elapsed_time_ms<<" ms"<<std::endl;
        t_label_frontier_edges_d = elapsed_time_ms;

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Seed phase

        t_start = std::chrono::high_resolution_clock::now();

        // GPU SEED PHASE
        half *seed_edges_ad;
        int *seed_edges_d;
        cudaMalloc(&seed_edges_ad, sizeof(half)*n_halfedges);
        cudaMemset(seed_edges_ad, 0, sizeof(half)*n_halfedges);
        cudaMalloc(&seed_edges_d, sizeof(int)*n_halfedges);
        seed_phase_d<<<(n_halfedges + BSIZE - 1)/BSIZE,BSIZE>>>(halfedges_d, max_edges_d, seed_edges_ad, n_halfedges); 
        gpuErrchk( cudaDeviceSynchronize() );

        int seed_len;
        scan_parallel_tc_2<int>(seed_edges_d, seed_edges_ad, n_halfedges);
        gpuErrchk( cudaDeviceSynchronize() );
        cudaMemcpy( &seed_len, seed_edges_d + n_halfedges - 1, sizeof(int), cudaMemcpyDeviceToHost );
        //int seed_len = scan(seed_edges_d, seed_edges_ad, n_halfedges); // ESTO SE PUEDE MEJORAR!
        //gpuErrchk( cudaDeviceSynchronize() );
        //printf ("-> %i %i %i %i\n", grid.x, grid.y, grid.z, (n_halfedges + BSIZE - 1)/BSIZE);
        compaction_d<<<(n_halfedges + BSIZE - 1)/BSIZE,BSIZE>>>(seed_edges_d, seed_edges_d, seed_edges_ad, n_halfedges);
        gpuErrchk( cudaDeviceSynchronize() );
        //compaction_cub(seed_edges_d, d_num, max_edges_d, seed_edges_ad, n_halfedges);
        //gpuErrchk( cudaDeviceSynchronize() );
        //printf("\ndone GPU seed phase....\n\n");//*/

        t_end = std::chrono::high_resolution_clock::now();
        elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        //std::cout<<"[GPU] Labeled seed edges in "<<elapsed_time_ms<<" ms"<<std::endl;
        t_label_seed_edges_d = elapsed_time_ms;

        ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        // Travel phase

        int polygon_seed;

        t_start = std::chrono::high_resolution_clock::now();
        
        int *output_seed_d;
        halfEdge *output_HalfEdges_d;
        cudaMalloc(&output_HalfEdges_d, sizeof(halfEdge)*n_halfedges);
        travel_phase_d<<<(n_halfedges + BSIZE - 1)/BSIZE,BSIZE>>>(output_HalfEdges_d, halfedges_d, max_edges_d, frontier_edges_d, n_halfedges);
        gpuErrchk( cudaDeviceSynchronize() );
        
        t_end = std::chrono::high_resolution_clock::now();
        t_traversal_1_d = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        //std::cout<<"[GPU] Traversal phase 1 in "<<t_traversal_1_d<<" ms"<<std::endl;

        t_start = std::chrono::high_resolution_clock::now();
        cudaMalloc(&output_seed_d , sizeof(int)*seed_len);
        search_frontier_edge_d<<<(seed_len+BSIZE-1)/BSIZE,BSIZE>>>(output_seed_d, halfedges_d, frontier_edges_d, seed_edges_d, seed_len);
        gpuErrchk( cudaDeviceSynchronize() );

        t_end = std::chrono::high_resolution_clock::now();
        t_traversal_2_d = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        //std::cout<<"[GPU] Traversal phase (search frontier edge) in "<<t_traversal_2_d<<" ms"<<std::endl;

        // Back to host
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
        elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        //std::cout<<"[GPU] Back to host in "<<elapsed_time_ms<<" ms"<<std::endl;
        t_back_to_host_d = elapsed_time_ms;

        // standard output, time not measured
        for (int i = 0; i < n_halfedges; i++)
            max_edges[i] = h_max_edges[i];
        for (int i = 0; i < n_halfedges; i++)
            frontier_edges[i] = h_frontier_edges[i];
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
        for (int i = 0; i < n_halfedges; i++)
        mesh_output->HalfEdges[i] = h_halfedges[i];


        //print output_seed
        for (int i = 0; i < seed_len; i++){
            //std::cout<<output_seed_h[i]<<" ";
            output_seeds.push_back(output_seed_h[i]);
            
        }

        this->m_polygons = output_seeds.size();
        //std::cout<<"[GPU] Mesh with "<<m_polygons<<" polygons "<<n_frontier_edges/2<<" edges and "<<n_barrier_edge_tips<<" barrier-edge tips."<<std::endl;
        //mesh_input->print_pg(std::to_string(mesh_input->vertices()) + ".pg");    
     
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
        out<<"\"d_time_to_label_total\": "<<t_label_max_edges_d+t_label_frontier_edges_d+t_label_seed_edges_d<<","<<std::endl;
        out<<"\"d_time_to_traversal_and_repair\": "<<t_traversal_and_repair_d<<","<<std::endl;
        out<<"\"d_time_to_traversal\": "<<t_traversal_1_d<<","<<std::endl;
        out<<"\"d_time_to_traversal_search_frontier_edge\": "<<t_traversal_2_d<<","<<std::endl;
        out<<"\"d_time_to_back_to_host\": "<<t_back_to_host_d<<","<<std::endl;
        out<<"\"d_time_to_repair\": "<<t_repair_d<<","<<std::endl;
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
        out<<"\t\"memory_total\": "<<m_max_edges + m_frontier_edge + m_seed_edges + m_seed_bet_mar + m_triangle_list + m_mesh_input + m_mesh_output + m_vertices_input + m_vertices_output<<std::endl;
        out<<"}"<<std::endl;
        out.close();
    }



    //Print off file of the polylla mesh
    void print_OFF(std::string filename){
        std::ofstream out(filename);

        std::cout << "Printing OFF file" <<  mesh_input->vertices() << " " << m_polygons << std::endl;
        int count = 0;

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
            //count++;
            //printf("-------> 1.1 %d\n", count);
        }
        //printf("-------> 2\n");
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