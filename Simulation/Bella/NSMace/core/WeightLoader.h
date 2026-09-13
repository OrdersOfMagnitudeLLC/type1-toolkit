// NSMace: OOM Proprietary, All Rights Reserved, Orders of Magnitude LLC
#pragma once
#include "NSMace.h"
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <stdexcept>

namespace NSMace {

// Flat weight tensor
struct Tensor {
    std::vector<Real> data;
    std::vector<int>  shape;
    int ndim() const { return (int)shape.size(); }
    int numel() const {
        int n=1; for(int d:shape) n*=d; return n;
    }
    Real& at(int i) { return data[i]; }
    const Real& at(int i) const { return data[i]; }
};

class WeightLoader {
public:
    std::unordered_map<std::string, Tensor> weights;

    void load(const std::string& json_path);
    const Tensor& get(const std::string& name) const;
    bool has(const std::string& name) const;
    void print_summary() const;
};

} // namespace NSMace
