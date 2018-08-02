#include <fstream>
#include <iostream>
#include <cstdlib>
#include <map>
#include <vector>
#include <utility>
#include <algorithm>
#include <string>
#include <assert.h>
#include <random>
#include <ctime>
#include <cstdlib>
#include "utf8.h"
using namespace std;

using wwstring = basic_string<char32_t>;

map<string, wwstring> special_symbols2{
    {"<eos>", {U"<eos>"}}, 
    {"<self>", {U"<self>"}},
    {"sil", {U"sil"}}, 
    {" ", {U" "}},
    {"\n", {U"\n"}},
    {"\"", {U"\""}},
    {"\";\"", {U"\";\""}},
};
map<string, string> special_symbols_utf8;

void init(){
    for(auto s : special_symbols2){
        string c_utf8;
        utf8::utf32to8(s.second.begin(), s.second.end(), back_inserter(c_utf8));
        special_symbols_utf8[s.first] = c_utf8;
    }
}

int main(int argc, char **argv){
    init();
    cout << "Leyendo fichero " << string(argv[1]) << "\n";
    std::srand ( unsigned ( std::time(0) ) );

    ifstream in(argv[1]);
    int size = atoi(argv[2]);
    int nsamples = atoi(argv[3]);

    vector<streampos> info;
    info.reserve(size);

    string line;
    line.reserve(2000);

    for(int i = 0; i < size; i++){
        streampos pos = in.tellg();
        getline(in, line);
        info.push_back(pos);
    }

    random_shuffle(info.begin(), info.end());    

    string salida(argv[1]);
    salida.append(".sampled");
    ofstream out(salida, ofstream::binary);
    for(int i = 0; i < nsamples; i++){
        in.seekg(info[i]);
        getline(in, line);
        out << line << special_symbols_utf8["\n"];
    }

    in.close();
    out.close();

    cout << "Finalizado fichero " << string(argv[1]) << "\n";

    return 0;
}


/*
int main(int argc, char **argv){
	init();
    std::srand ( unsigned ( std::time(0) ) );

    uint nsamples = atoi(argv[1]);

    for(auto i = 2u; i < argc; i++){
        cout << "Leyendo fichero " << string(argv[i]) << "\n";

        ifstream in(argv[i]);
        string salida(argv[i]);
        salida.append(".sampled");
        ofstream out(salida, ofstream::binary);
        string line;
        line.reserve(2000);

        uint nrows = 0u;
        vector<streampos> rows_pos;
        rows_pos.push_back(in.tellg());
        while(getline(in, line)){
            rows_pos.push_back(in.tellg());
            nrows++;   
        }

        for(uint i = 0; i < 10; i++)
            cout << rows_pos[i] << " ";
        cout << endl;

        //random_shuffle(rows_pos.begin(), rows_pos.end());

        for(uint i = 0; i < 10; i++)
            cout << rows_pos[i] << " ";
        cout << endl;

        for(uint i = 0; i < nsamples; i++){
            in.seekg(rows_pos[i]);
            getline(in, line); cout << line.size() << endl;
            out << line << special_symbols_utf8["\n"];
        }

        in.close();
        out.close();

        cout << "Finalizado fichero " << string(argv[i]) << "\n";
    }

	return 0;
}
*/