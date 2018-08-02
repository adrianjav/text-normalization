#include <fstream>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <utility>
#include <queue>
#include <set>
#include <map>
#include <assert.h>
#include <iterator>
#include <limits>
#include "utf8.h"
using namespace std;

using index_t = pair<uint,uint>;
using pq_t = priority_queue<index_t>;
using wwstring = basic_string<char32_t>;


set<char32_t> allowed_chars{
    U'a', U'b', U'c', U'd', U'e', U'f', U'g', U'h', U'i', U'j', U'k', U'l', U'm', U'n', U'ñ', U'l', U'o', U'p', U'q', U'r', U's', U't', U'u', U'v', U'w', U'x', U'y', U'z',
    U'A', U'B', U'C', U'D', U'E', U'F', U'G', U'H', U'I', U'J', U'K', U'L', U'M', U'N', U'Ñ', U'L', U'O', U'P', U'Q', U'R', U'S', U'T', U'U', U'V', U'W', U'X', U'Y', U'Z',
    U'0', U'1', U'2', U'3', U'4', U'5', U'6', U'7', U'8', U'9', U'+', U'-', U'*', U' ', U',', U'.', U';', U':', U'-', U'_', U'\'', U'"', U'?', U'¿', U'!', U'¡', U'(', U')',
    U'|', U'@', U'#', U'%', U'$', U'€', U'£', U'¢', U'¥', U'[', U']', U'{', U'}', U'~', U'^', U'=', U'<', U'>',  U'/', U'`', U'\\',
    U'á', U'é', U'í', U'ó', U'ú', U'Á', U'É', U'Í', U'Ó', U'Ú', U'ä', U'ë', U'ï', U'ö', U'ü', U'Ä', U'Ë', U'Ï', U'Ö', U'Ü', 
    U'¤', U'®', U'©', U'ª', U'º', U'°', U'«', U'»', U'\r', U'\n', U'&'
};

map<string, wwstring> special_symbols2{
    {"<eos>", {U"<eos>"}}, 
    {"<self>", {U"<self>"}},
    {"sil", {U"sil"}}, 
    {" ", {U" "}},
    {"\n", {U"\n"}},
    {"\"", {U"\""}},
    {"\",\"", {U"\",\""}},
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
        
        if(allowed_chars.find(c) == allowed_chars.end())
            return line.end();
    }

    assert(nquotes % 2 == 0);

    return previous;
}

[[nodiscard]] 
bool take_token(const string &line, string &before, string &after){
    bool found = false;
    auto pos = line.begin();

    if(found = (utf8::peek_next(pos, line.end()) == U'\"')) utf8::next(pos, line.end());
    auto to = advance_token(line, pos, found);
    if(to == line.end() || pos == line.end())
        return false;

    assert(utf8::next(pos, line.end()) == U',');
    if(found = (utf8::peek_next(pos, line.end()) == U'\"')) utf8::next(pos, line.end());
    
    auto from = pos;
    to = advance_token(line, pos, found);
    if(to == line.end() || pos == line.end())
        return false;
    if(before.size() > 0)
        before += special_symbols_utf8[" "];
    string word = line.substr(distance(line.begin(), from), distance(from, to));
    before += word;

    assert(utf8::next(pos, line.end()) == U',');
    if(found = (utf8::peek_next(pos, line.end()) == U'\"')) utf8::next(pos, line.end());
    from = pos;
    to = advance_token(line, pos, found);
    if(to == line.end())
        return false;
    string other_word = line.substr(distance(line.begin(), from), distance(from, to));

    if(after.size() > 0)
        after += special_symbols_utf8[" "];

    if(other_word == special_symbols_utf8["<self>"] || other_word == special_symbols_utf8["sil"])
        after += word;
    else
        after += other_word;
    

    return true;
}

[[nodiscard]] 
bool take_line(ifstream &in, string &before, string &after){
    before.clear(); after.clear();
    
    string line; line.reserve(2000);
    if(!getline(in, line))
        return false;

    bool valid = true;
    while(line.find(special_symbols_utf8["<eos>"]) == string::npos){
        if(utf8::distance(line.begin(), line.end()) > 0)
            valid &= take_token(line, before, after);
        
        if(!getline(in, line))
            return false;
    }

    return valid;
}

void write(ofstream &out, const string &before, const string &after){
    out << special_symbols_utf8["\""] << before << special_symbols_utf8["\",\""] <<
           after << special_symbols_utf8["\""] << special_symbols_utf8["\n"];
}

int main(int argc, char **argv){
    init();

    //pq_t queues[argc-1];
    string outnames[argc-1];
    for(auto i = 0u; i < argc-1; i++){
        outnames[i] = string{argv[i+1]};
        outnames[i].replace(outnames[i].size() - 4, string::npos, ".csv.cleaned");
    }

    cout << "Processing..." << endl;
    for(auto i = 1u; i < argc; i++){
        string filename{argv[i]};
        ifstream in(argv[i]);
        ofstream out(outnames[i-1], ofstream::binary);

        string line;
        line.reserve(2000);
        cout << "File " << filename << ":" << endl;

        getline(in, line); // header

        auto nline = 0u, discarded = 0u;
        string before, after;
        before.reserve(2000); after.reserve(2000);
        while(!in.eof()){
            auto pos = out.tellp();
            if(take_line(in, before, after)){
                //queues[i-1].push({utf8::distance(before.begin(), before.end()), pos});
                nline++;

                write(out, before, after);
            }
            else
                discarded++;
        }

        discarded--;
        cout << discarded << " lines have been discarded out of " << (nline + discarded) << endl;

        in.close();
        out.close();
    }
    
    /*
    cout << "Sorting..." << endl;
    ofstream out("dataset.csv", ofstream::binary);
    ifstream ins[argc-1];
    for(auto i = 0u; i < argc-1; i++)
        ins[i].open(outnames[i]);

    string line;
    line.reserve(2000);
    while(!all_of(queues, queues + argc - 1, [](auto q){ return q.empty(); })){
        uint arg_max;
        uint maximum = numeric_limits<uint>::min();
        uint where;

        for(auto i = 0u; i < argc - 1; i++){
            if(!queues[i].empty() && queues[i].top().first > maximum){
                maximum = queues[i].top().first;
                where = queues[i].top().second;
                arg_max = i;
            }
        }
        queues[arg_max].pop();

        ins[arg_max].seekg(where);
        getline(ins[arg_max], line);
        out << line << special_symbols_utf8["\n"];
    }

    for(auto i = 0u; i < argc-1; i++)
        ins[i].close();
    out.close();
    */

    cout << "Done" << endl;

    return 0;
}