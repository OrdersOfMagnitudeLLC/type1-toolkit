// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>

struct GnomeMaterial {
    std::string material_id;
    std::string formula;
    double decomp_energy;
    int n_elements;
    double bandgap;
    double abundance_score;
    double density;
};

inline int count_elements(const std::string& elements_str) {
    // Elements format: "['S', 'Zr', 'Cs']" or similar
    // Count the number of single-quoted items
    int count = 0;
    bool in_quote = false;
    for (char c : elements_str) {
        if (c == '\'') {
            in_quote = !in_quote;
            if (!in_quote) count++;  // End of a quoted element
        }
    }
    return count;
}

inline std::vector<std::string> parse_elements(const std::string& elements_str) {
    // Elements format: "['S', 'Zr', 'Cs']" or similar
    // Extract element symbols from quoted strings
    std::vector<std::string> elements;
    std::string current;
    bool in_quote = false;
    for (char c : elements_str) {
        if (c == '\'') {
            in_quote = !in_quote;
            if (!in_quote && !current.empty()) {
                elements.push_back(current);
                current.clear();
            }
        } else if (in_quote) {
            current += c;
        }
    }
    return elements;
}

inline double compute_abundance_score(const std::vector<std::string>& elements) {
    // Crustal abundance lookup table (ppm by mass)
    static const std::unordered_map<std::string, double> abundance = {
        {"O", 461000},
        {"Si", 277200},
        {"Al", 81300},
        {"Fe", 50000},
        {"Ca", 36300},
        {"Na", 28300},
        {"K", 25900},
        {"Mg", 20900},
        {"Ti", 6200},
        {"H", 1400},
        {"P", 1050},
        {"Mn", 950},
        {"F", 950},
        {"Ba", 425},
        {"Sr", 370},
        {"S", 260},
        {"C", 200},
        {"Zr", 165},
        {"Cl", 130},
        {"V", 120},
        {"Cr", 100},
        {"Rb", 90},
        {"Ni", 80},
        {"Zn", 70},
        {"Cu", 60},
        {"Co", 25},
        {"Li", 20},
        {"N", 19},
        {"Nb", 17},
        {"Sn", 2.3},
        {"Ge", 1.5},
        {"W", 1.25},
        {"Mo", 1.2},
        {"Br", 2.4},
        {"I", 0.45},
        {"La", 39},
        {"Ce", 66.5},
        {"Y", 33},
        {"Er", 3.5},
        {"Dy", 5.2}
    };
    
    double min_abundance = 999999.0;  // Default high value
    for (const auto& elem : elements) {
        auto it = abundance.find(elem);
        double ppm = (it != abundance.end()) ? it->second : 0.001;  // Default for rare elements
        min_abundance = std::min(min_abundance, ppm);
    }
    return min_abundance;
}

inline std::vector<GnomeMaterial> load_gnome_csv(const std::string& filepath) {
    std::vector<GnomeMaterial> materials;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    
    std::string line;
    // Skip header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string field;
        bool in_quotes = false;
        
        // Parse CSV handling quoted fields
        std::string current;
        for (char c : line) {
            if (c == '"') {
                in_quotes = !in_quotes;
            } else if (c == ',' && !in_quotes) {
                fields.push_back(current);
                current.clear();
            } else {
                current += c;
            }
        }
        fields.push_back(current);
        
        if (fields.size() < 18) continue;  // Skip malformed rows
        
        // Column indices (0-based from header):
        // 2: MaterialId
        // 3: Reduced Formula
        // 4: Elements
        // 7: Density
        // 15: Decomposition Energy Per Atom
        // 17: Bandgap
        
        std::string material_id = fields[2];
        std::string formula = fields[3];
        std::string elements_str = fields[4];
        std::string density_str = fields[7];
        std::string decomp_str = fields[15];
        std::string bandgap_str = fields[17];
        
        // Parse density
        double density = 0.0;
        if (!density_str.empty()) {
            try {
                density = std::stod(density_str);
            } catch (...) {
                density = 0.0;  // Default to 0 if invalid
            }
        }
        
        // Parse decomp_energy
        double decomp_energy = 0.0;
        if (!decomp_str.empty()) {
            try {
                decomp_energy = std::stod(decomp_str);
            } catch (...) {
                continue;  // Skip rows with invalid decomp energy
            }
        }
        
        // Skip unstable rows (decomp_energy > 0.05 eV/atom)
        if (decomp_energy > 0.05) continue;
        
        // Count elements
        int n_elements = count_elements(elements_str);
        
        // Parse elements for abundance score
        std::vector<std::string> elements = parse_elements(elements_str);
        double abundance_score = compute_abundance_score(elements);
        
        // Parse bandgap
        double bandgap = 0.0;
        if (!bandgap_str.empty()) {
            try {
                bandgap = std::stod(bandgap_str);
            } catch (...) {
                bandgap = 0.0;  // Default to 0 if invalid
            }
        }
        
        GnomeMaterial mat;
        mat.material_id = material_id;
        mat.formula = formula;
        mat.decomp_energy = decomp_energy;
        mat.n_elements = n_elements;
        mat.bandgap = bandgap;
        mat.abundance_score = abundance_score;
        mat.density = density;
        
        materials.push_back(mat);
    }
    
    return materials;
}
