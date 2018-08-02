#include <fstream>
#include <iostream>
#include <cstdlib>
#include <map>
#include <vector>
#include <utility>
#include <algorithm>
#include <string>
#include <assert.h>
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


template<typename iterator>
iterator advance_token(const string &line, iterator &pos, bool found){
    char32_t c;
    auto nquotes = found ? 1u : 0u;
    auto previous = pos;

    while(pos != line.end() && !(utf8::peek_next(pos, line.end()) == U',' && nquotes % 2 == 0)){
        previous = pos;
        c = utf8::next(pos, line.end()); // codepoint
        if(c == U'\"')
            nquotes++;
    }

    assert(nquotes % 2 == 0);

    return previous;
}

uint get_size(const string &line){
    auto end = line.begin();
    advance_token(line, end, false);
    end++;
    auto start = end;
    advance_token(line, end, false);
    return utf8::distance(start, end);
}

int main(int argc, char **argv){
	init();

    /*
    percentiles
    [1] "0 :  19"
    [1] "0.25 :  83"
    [1] "0.5 :  131"
    [1] "0.75 :  196"
    [1] "0.9 :  241"
    [1] "0.99 :  311"
    [1] "0.999 :  536"
    [1] "1 :  35457"
    */

    // REAL [1] "0.9 :  126"

    uint max_size = 126; // percentile 0.9

    for(auto i = 1u; i < argc; i++){
        cout << "Leyendo fichero " << string(argv[i]) << "\n";

        ifstream in(argv[i]);
        string salida(argv[i]);
        salida.append(".pruned");
        ofstream out(salida, ofstream::binary);
        string line;
        line.reserve(2000);

        while(getline(in, line)){
            auto size = get_size(line);
            if(size <= max_size)
                out << line << special_symbols_utf8["\n"];
        }

        in.close();
        out.close();

        cout << "Finalizado fichero " << string(argv[i]) << "\n";
    }

	return 0;
}
