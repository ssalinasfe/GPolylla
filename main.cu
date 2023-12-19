#include <algorithm>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <getopt.h>
#include <cstdlib>


#include <polylla.cu>
#include <triangulation.cu>

int main(int argc, char **argv) {
    int opt;
    std::string node_file, ele_file, neigh_file, off_file, output;
    int size = 0;

    while ((opt = getopt(argc, argv, "noi")) != -1) {
        switch (opt) {
            case 'n':
                if (argc != 6) {
                    std::cerr << "Usage: " << argv[0] << " -n <node_file> <ele_file> <neigh_file> <output name>\n";
                    return 1;
                }
                node_file = argv[2];
                ele_file = argv[3];
                neigh_file = argv[4];
                output = argv[5];
                break;
            case 'o':
                if (argc != 4) {
                    std::cerr << "Usage: " << argv[0] << " -o <off file> <output name>\n";
                    return 1;
                }
                off_file = argv[2];
                output = argv[3];
                break;
            case 'i':
                if (argc != 4) {
                    std::cerr << "Usage: " << argv[0] << " -i <size> <output name>\n";
                    return 1;
                }
                size = std::atoi(argv[2]);
                output = argv[3];
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " [-n | -o | -i] args\n";
                return 1;
        }
    }

    if (!node_file.empty() && !ele_file.empty() && !neigh_file.empty()) {

        // Process node, ele, neigh files
        Polylla mesh(node_file, ele_file, neigh_file);
        mesh.print_stats(output + ".json");
        std::cout << "output json in " << output << ".json" << std::endl;

    } else if (!off_file.empty()) {

        // Process off file
        Polylla mesh(off_file);
        mesh.print_stats(output + ".json");
        std::cout << "output json in " << output << ".json" << std::endl;

    } else if (size > 0) {

        // Process size directly
        Polylla mesh(size);

        mesh.print_stats(output + ".json");
        std::cout << "output json in " << output << ".json" << std::endl;
        
    } else {
        std::cerr << "Invalid arguments.\n";
        return 1;
    }

    return 0;
}



/*#include <algorithm>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <polylla.cu>

#include <triangulation.cu>

//#include <compresshalfedge.hpp>
//#include <io_void.hpp>
//#include <delfin.hpp>
//

#include <unistd.h> // for getopt

int main(int argc, char **argv) {
    int opt;
    std::string node_file, ele_file, neigh_file, off_file, output;

    while ((opt = getopt(argc, argv, "n:e:i:o:s:")) != -1) {
        switch (opt) {
            case 'n':
                node_file = optarg;
                break;
            case 'e':
                ele_file = optarg;
                break;
            case 'i':
                neigh_file = optarg;
                break;
            case 'o':
                output = optarg;
                break;
            case 's':
                off_file = optarg;
                break;
            default:
                std::cout << "Usage: " << argv[0] << " -n <node_file .node> -e <ele_file .ele> -i <neigh_file .neigh> -o <output name>" << std::endl;
                std::cout << "Or: " << argv[0] << " -s <size> -o <output name>" << std::endl;
                return 1;
        }
    }

    if (!node_file.empty() && !ele_file.empty() && !neigh_file.empty()) {
        // Process node, ele, neigh files
        Polylla mesh(node_file, ele_file, neigh_file);
        mesh.print_stats(output + ".json");
        std::cout << "output json in " << output << ".json" << std::endl;
    } else if (!off_file.empty()) {
        // Process off file
        int size = atoi(off_file.c_str());
        std::string output_name = "" + output;
        Polylla mesh(off_file);
        mesh.print_stats(output_name + ".json");
        std::cout << "output json in " << output_name << ".json" << std::endl;
    } else {
        // Process size directly
        int size = atoi(output.c_str());
        std::string output_name = "" + output;
        Polylla mesh(size);
        mesh.print_stats(output_name + ".json");
        std::cout << "output json in " << output_name << ".json" << std::endl;
    }

    return 0;
}

*/