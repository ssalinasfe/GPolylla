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
#include <triangulation.hpp>
#include <chrono>
#include <iomanip>

#define print_e(eddddge) eddddge<<" ( "<<mesh_input->origin(eddddge)<<" - "<<mesh_input->target(eddddge)<<") "

struct Polygon{
    int seed_edge; //Edge that generate the polygon
    std::vector<int> vertices; //Vertices of the polygon
    //std::vector<int> neighbors; //Neighbors of the polygon WIP
};

class Polylla
{
private:
    typedef std::vector<int> _polygon; 
    typedef std::vector<char> bit_vector; 


    Triangulation *mesh_input; // Halfedge triangulation
    Triangulation *mesh_output;
    std::vector<int> output_seeds; //Seeds of the polygon

    std::vector<Polygon> polygonal_mesh; //Vector of polygons generated by polygon
    std::vector<int> triangles; //True if the edge generated a triangle CHANGE!!!!

    bit_vector max_edges; //True if the edge i is a max edge
    bit_vector frontier_edges; //True if the edge i is a frontier edge
    bit_vector terminal_edges; //True if the edge i is a terminal edge
    std::vector<int> seed_edges; //Seed edges that generate polygon simple and non-simple

    int m_polygons = 0; //Number of polygons
    int n_frontier_edges = 0; //Number of frontier edges
    int n_barrier_edge_tips = 0; //Number of barrier edge tips
    
public:

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
        terminal_edges = bit_vector(mesh_input->halfEdges(), false);
        //seed_edges = bit_vector(mesh_input->halfEdges(), false);
        triangles = mesh_input->get_Triangles(); //Change by triangle list

        //Label max edges of each triangle
        //for (size_t t = 0; t < mesh_input->faces(); t++){
        auto t_start = std::chrono::high_resolution_clock::now();
        for(auto &t : triangles)
            max_edges[label_max_edge(t)] = true;   
        auto t_end = std::chrono::high_resolution_clock::now();
        double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"Labered max edges in "<<elapsed_time_ms<<" ms"<<std::endl;

        t_start = std::chrono::high_resolution_clock::now();
        //Label frontier edges
        for (std::size_t e = 0; e < mesh_input->halfEdges(); e++){
            if(is_frontier_edge(e)){
                frontier_edges[e] = true;
                n_frontier_edges++;
            }
        }
        t_end = std::chrono::high_resolution_clock::now();
        elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"Labeled frontier edges in "<<elapsed_time_ms<<" ms"<<std::endl;
        
        t_start = std::chrono::high_resolution_clock::now();
        //label seeds edges,
        for (std::size_t e = 0; e < mesh_input->halfEdges(); e++)
            if(mesh_input->is_interior_face(e) && is_seed_edge(e))
                seed_edges.push_back(e);
        t_end = std::chrono::high_resolution_clock::now();
        elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"Labeled seed edges in "<<elapsed_time_ms<<" ms"<<std::endl;


        //Travel phase: Generate polygon mesh
        int polygon_seed;
        //Foreach seed edge generate polygon
        t_start = std::chrono::high_resolution_clock::now();
        for(auto &e : seed_edges){
            polygon_seed = travel_triangles(e);

            if(!has_BarrierEdgeTip(polygon_seed)){ //If the polygon is a simple polygon then is part of the mesh
                output_seeds.push_back(polygon_seed);
            }else{ //Else, the polygon is send to reparation phase
                barrieredge_tip_reparation(polygon_seed);
            }         
        }    
        t_end = std::chrono::high_resolution_clock::now();
        elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
        std::cout<<"Polygons generated/repaired in "<<elapsed_time_ms<<" ms"<<std::endl;
        
        this->m_polygons = output_seeds.size();

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


private:

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

        double dist0 = mesh_input->distance(e);
        double dist1 = mesh_input->distance(mesh_input->next(e));
        double dist2 = mesh_input->distance(mesh_input->next(mesh_input->next(e)));

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
        int x, y, i;
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