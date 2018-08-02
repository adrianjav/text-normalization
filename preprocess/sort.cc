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
	cout << "Leyendo fichero " << string(argv[1]) << "\n";

	ifstream in(argv[1]);
	int size = atoi(argv[2]);

	vector<pair<int,streampos>> info;
	info.reserve(size);

	string line;
	line.reserve(2000);

	for(int i = 0; i < size; i++){
		streampos pos = in.tellg();
		getline(in, line);
		info.push_back(make_pair(get_size(line), pos));
	}

	sort(info.begin(), info.end(), [](auto f, auto s){ return f.first > s.first;} );

	string salida(argv[1]);
	salida.append(".sorted");
	ofstream out(salida, ofstream::binary);
	for(int i = 0; i < size; i++){
		in.seekg(info[i].second);
		getline(in, line);
		out << line << special_symbols_utf8["\n"];
	}

	in.close();
	out.close();

	cout << "Finalizado fichero " << string(argv[1]) << "\n";

	return 0;
}
