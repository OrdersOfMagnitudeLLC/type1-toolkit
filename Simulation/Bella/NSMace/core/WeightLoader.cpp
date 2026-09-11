// NSMace — OOM Proprietary, All Rights Reserved, Orders of Magnitude LLC
// Loads MACE-MP weights from JSON export
#include "WeightLoader.h"
#include <iostream>
#include <functional>
#include <cstdlib>

// Minimal JSON parser — only handles what our export produces
// Format: {"name": [[...]], "name2": [...]}
namespace NSMace {

static void parse_flat_array(const std::string& src, size_t& pos,
                              std::vector<Real>& out, std::vector<int>& shape) {
    // Skip whitespace
    auto skip = [&]{ while(pos<src.size() && (src[pos]==' '||src[pos]=='\n'||src[pos]=='\r'||src[pos]=='\t')) pos++; };

    std::function<void(int)> parse_level = [&](int depth) {
        skip();
        if(src[pos]=='[') {
            pos++;
            int count=0;
            if((int)shape.size()<=depth) shape.push_back(0);
            while(true) {
                skip();
                if(src[pos]==']') { pos++; break; }
                if(src[pos]==',') { pos++; continue; }
                parse_level(depth+1);
                count++;
            }
            shape[depth] = count;
        } else {
            // scalar
            char* end;
            Real v = std::strtod(src.c_str()+pos, &end);
            out.push_back(v);
            pos = end - src.c_str();
        }
    };
    parse_level(0);
}

void WeightLoader::load(const std::string& path) {
    std::ifstream f(path);
    if(!f) throw std::runtime_error("Cannot open: " + path);
    std::string src((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    
    size_t pos = 0;
    auto skip = [&]{ while(pos<src.size() && (src[pos]==' '||src[pos]=='\n'||src[pos]=='\r'||src[pos]=='\t')) pos++; };
    
    skip(); 
    if(src[pos]!='{') throw std::runtime_error("Expected {");
    pos++;
    
    while(true) {
        skip();
        if(src[pos]=='}') break;
        if(src[pos]==',') { pos++; continue; }
        
        // Parse key string
        if(src[pos]!='"') throw std::runtime_error("Expected key string");
        pos++;
        size_t key_start = pos;
        while(pos<src.size() && src[pos]!='"') pos++;
        std::string key = src.substr(key_start, pos-key_start);
        pos++; // closing "
        
        skip();
        if(src[pos]!=':') throw std::runtime_error("Expected :");
        pos++;
        
        Tensor t;
        parse_flat_array(src, pos, t.data, t.shape);
        weights[key] = std::move(t);
    }
    
    std::cout << "Loaded " << weights.size() << " weight tensors\n";
}

const Tensor& WeightLoader::get(const std::string& name) const {
    auto it = weights.find(name);
    if(it==weights.end()) throw std::runtime_error("Weight not found: "+name);
    return it->second;
}

bool WeightLoader::has(const std::string& name) const {
    return weights.count(name)>0;
}

void WeightLoader::print_summary() const {
    for(auto& [name,t] : weights) {
        std::cout << name << ": [";
        for(int i=0;i<(int)t.shape.size();i++) 
            std::cout << t.shape[i] << (i+1<(int)t.shape.size()?",":"");
        std::cout << "] numel=" << t.numel() << "\n";
    }
}

} // namespace NSMace
