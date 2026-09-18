// Copyright (c) 2026 Orders of Magnitude LLC
// Licensed under the OOM Commercial License v1.0
// See LICENSE.md in the repository root or ofmagnitude.com

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <string>
#include "../loaders/gnome_loader.hpp"
#include "../bob.hpp"

int main() {
    std::cout << "Loading GNoME dataset...\n";
    
    // Load GNoME data
    std::vector<GnomeMaterial> materials = load_gnome_csv("data/gnome.csv");
    std::cout << "Loaded " << materials.size() << " stable materials (decomp_energy <= 0.05 eV/atom)\n\n";
    
    // Build BobIndex with K=4 properties (decomp_energy, n_elements, bandgap, abundance_score)
    size_t k = 4;
    size_t n = materials.size();
    
    std::cout << "Building BobIndex with K=" << k << " properties, N=" << n << " materials...\n";
    BobIndex index(k, n);
    
    // Add materials to index
    for (size_t i = 0; i < materials.size(); ++i) {
        std::vector<double> props = {
            materials[i].decomp_energy,
            static_cast<double>(materials[i].n_elements),
            materials[i].bandgap,
            materials[i].abundance_score
        };
        index.add_material(i, props);
    }
    
    // Build indices
    index.build_indices();
    std::cout << "Index built successfully.\n\n";
    
    // Define battery electrolyte filter criteria:
    // - decomp_energy between -999.0 and 0.05 eV/atom (stability)
    // - n_elements between 2 and 5
    // - bandgap between 3.0 and 999.0 eV (insulator)
    // - abundance_score between 20.0 and 999999.0 ppm (common elements only)
    // - formula contains "Li" or "Na" or "Zn" (mobile ion)
    
    std::vector<PropBounds> bounds = {
        {-999.0, 0.05},   // decomp_energy: -999 to 0.05 eV/atom
        {2.0, 5.0},        // n_elements: 2 to 5
        {3.0, 999.0},      // bandgap: 3.0 to 999.0 eV
        {20.0, 999999.0}   // abundance_score: 20 to 999999 ppm
    };
    
    std::cout << "Querying for battery electrolytes:\n";
    std::cout << "  decomp_energy: [-999.0, 0.05] eV/atom\n";
    std::cout << "  n_elements: [2, 5]\n";
    std::cout << "  bandgap: [3.0, 999.0] eV\n";
    std::cout << "  abundance_score: [20.0, 999999.0] ppm\n";
    std::cout << "  formula contains: Li or Na or Zn\n\n";
    
    auto candidates = index.query(bounds);
    std::cout << "Found " << candidates.size() << " candidates matching numeric criteria\n\n";
    
    // String-filter for Li/Na/Zn on results
    std::vector<std::tuple<double, double, double, std::string>> electrolytes;
    for (size_t idx : candidates) {
        const auto& mat = materials[idx];
        bool has_mobile_ion = (mat.formula.find("Li") != std::string::npos) ||
                              (mat.formula.find("Na") != std::string::npos) ||
                              (mat.formula.find("Zn") != std::string::npos);
        
        if (has_mobile_ion) {
            electrolytes.push_back({mat.decomp_energy, mat.bandgap, mat.abundance_score, mat.formula});
        }
    }
    
    std::cout << "After Li/Na filter: " << electrolytes.size() << " battery electrolyte candidates\n\n";
    
    // Sort by (bandgap * abundance_score) descending
    std::sort(electrolytes.begin(), electrolytes.end(), 
        [](const auto& a, const auto& b) {
            double score_a = std::get<1>(a) * std::get<2>(a);
            double score_b = std::get<1>(b) * std::get<2>(b);
            return score_a > score_b;
        });
    
    // Print top 20 candidates
    int limit = std::min(20, static_cast<int>(electrolytes.size()));
    std::cout << "Top " << limit << " candidates sorted by (bandgap * abundance_score):\n";
    std::cout << std::setw(4) << "Rank" << " | " << std::setw(12) << "Decomp (eV)" << " | " 
              << std::setw(10) << "Bandgap (eV)" << " | " << std::setw(12) << "Abund (ppm)" << " | Formula\n";
    std::cout << std::string(80, '-') << "\n";
    
    for (int i = 0; i < limit; ++i) {
        std::cout << std::setw(4) << (i + 1) << " | " 
                  << std::fixed << std::setprecision(4) << std::setw(12) << std::get<0>(electrolytes[i]) << " | "
                  << std::fixed << std::setprecision(3) << std::setw(10) << std::get<1>(electrolytes[i]) << " | "
                  << std::fixed << std::setprecision(2) << std::setw(12) << std::get<2>(electrolytes[i]) << " | "
                  << std::get<3>(electrolytes[i]) << "\n";
    }
    
    // Targeted query: Earth-Abundant Na-Zn Sulfide Candidates
    std::cout << "\n\n=== Targeted Earth-Abundant Na-Zn Sulfide Candidates ===\n";
    std::cout << "Filtering for materials with only: Na, Zn, Al, Si, S, Ca, Mg, P, O, F\n";
    std::cout << "At least one of: Na, Zn present\n";
    std::cout << "decomp_energy: [-999.0, 0.05], bandgap: [3.0, 999.0], abundance_score: [20.0, 999999.0], n_elements: [2, 6]\n\n";
    
    std::vector<std::tuple<double, double, double, std::string>> targeted_candidates;
    std::string allowed_elements = "NaZnAlSiSCaMgPOF";
    
    for (size_t i = 0; i < materials.size(); ++i) {
        const auto& mat = materials[i];
        
        // Check numeric criteria
        if (mat.decomp_energy < -999.0 || mat.decomp_energy > 0.05) continue;
        if (mat.bandgap < 3.0 || mat.bandgap > 999.0) continue;
        if (mat.abundance_score < 20.0) continue;
        if (mat.n_elements < 2 || mat.n_elements > 6) continue;
        
        // Check if formula contains Na or Zn
        bool has_mobile_ion = (mat.formula.find("Na") != std::string::npos) ||
                              (mat.formula.find("Zn") != std::string::npos);
        if (!has_mobile_ion) continue;
        
        // Check if formula contains only allowed elements
        bool has_only_allowed = true;
        for (char c : mat.formula) {
            if (isupper(c)) {
                std::string elem(1, c);
                if (allowed_elements.find(elem) == std::string::npos) {
                    has_only_allowed = false;
                    break;
                }
            }
        }
        if (!has_only_allowed) continue;
        
        targeted_candidates.push_back({mat.decomp_energy, mat.bandgap, mat.abundance_score, mat.formula});
    }
    
    std::cout << "Found " << targeted_candidates.size() << " targeted candidates\n\n";
    
    // Sort by bandgap descending (widest electrochemical window first)
    std::sort(targeted_candidates.begin(), targeted_candidates.end(), 
        [](const auto& a, const auto& b) {
            return std::get<1>(a) > std::get<1>(b);
        });
    
    // Print top 30 candidates
    int limit2 = std::min(30, static_cast<int>(targeted_candidates.size()));
    std::cout << "Top " << limit2 << " candidates sorted by bandgap descending:\n";
    std::cout << std::setw(4) << "Rank" << " | " << std::setw(12) << "Decomp (eV)" << " | " 
              << std::setw(10) << "Bandgap (eV)" << " | " << std::setw(12) << "Abund (ppm)" << " | Formula\n";
    std::cout << std::string(80, '-') << "\n";
    
    for (int i = 0; i < limit2; ++i) {
        std::cout << std::setw(4) << (i + 1) << " | " 
                  << std::fixed << std::setprecision(4) << std::setw(12) << std::get<0>(targeted_candidates[i]) << " | "
                  << std::fixed << std::setprecision(3) << std::setw(10) << std::get<1>(targeted_candidates[i]) << " | "
                  << std::fixed << std::setprecision(2) << std::setw(12) << std::get<2>(targeted_candidates[i]) << " | "
                  << std::get<3>(targeted_candidates[i]) << "\n";
    }
    
    // Mixed Anion Na-S-F Candidates
    std::cout << "\n\n=== Mixed Anion Na-S-F Candidates ===\n";
    std::cout << "Filtering for materials with only: Na, Zn, Al, Si, Ca, Mg, S, F, O, P\n";
    std::cout << "Must contain BOTH S and F (mixed anion)\n";
    std::cout << "Must contain Na\n";
    std::cout << "decomp_energy: [-999.0, 0.05], bandgap: [3.5, 999.0], abundance_score: [20.0, 999999.0], n_elements: [3, 6]\n\n";
    
    std::vector<std::tuple<double, double, double, std::string>> mixed_anion_candidates;
    std::string allowed_elements_mixed = "NaZnAlSiCaMgSFO P";
    
    for (size_t i = 0; i < materials.size(); ++i) {
        const auto& mat = materials[i];
        
        // Check numeric criteria
        if (mat.decomp_energy < -999.0 || mat.decomp_energy > 0.05) continue;
        if (mat.bandgap < 3.5 || mat.bandgap > 999.0) continue;
        if (mat.abundance_score < 20.0) continue;
        if (mat.n_elements < 3 || mat.n_elements > 6) continue;
        
        // Check if formula contains Na
        if (mat.formula.find("Na") == std::string::npos) continue;
        
        // Check if formula contains BOTH S and F
        bool has_s = (mat.formula.find("S") != std::string::npos);
        bool has_f = (mat.formula.find("F") != std::string::npos);
        if (!has_s || !has_f) continue;
        
        // Check if formula contains only allowed elements
        bool has_only_allowed = true;
        for (char c : mat.formula) {
            if (isupper(c)) {
                std::string elem(1, c);
                if (allowed_elements_mixed.find(elem) == std::string::npos) {
                    has_only_allowed = false;
                    break;
                }
            }
        }
        if (!has_only_allowed) continue;
        
        mixed_anion_candidates.push_back({mat.decomp_energy, mat.bandgap, mat.abundance_score, mat.formula});
    }
    
    std::cout << "Found " << mixed_anion_candidates.size() << " mixed anion candidates\n\n";
    
    // Sort by bandgap descending (widest electrochemical window first)
    std::sort(mixed_anion_candidates.begin(), mixed_anion_candidates.end(), 
        [](const auto& a, const auto& b) {
            return std::get<1>(a) > std::get<1>(b);
        });
    
    // Print top 20 candidates
    int limit3 = std::min(20, static_cast<int>(mixed_anion_candidates.size()));
    std::cout << "Top " << limit3 << " candidates sorted by bandgap descending:\n";
    std::cout << std::setw(4) << "Rank" << " | " << std::setw(12) << "Decomp (eV)" << " | " 
              << std::setw(10) << "Bandgap (eV)" << " | " << std::setw(12) << "Abund (ppm)" << " | Formula\n";
    std::cout << std::string(80, '-') << "\n";
    
    for (int i = 0; i < limit3; ++i) {
        std::cout << std::setw(4) << (i + 1) << " | " 
                  << std::fixed << std::setprecision(4) << std::setw(12) << std::get<0>(mixed_anion_candidates[i]) << " | "
                  << std::fixed << std::setprecision(3) << std::setw(10) << std::get<1>(mixed_anion_candidates[i]) << " | "
                  << std::fixed << std::setprecision(2) << std::setw(12) << std::get<2>(mixed_anion_candidates[i]) << " | "
                  << std::get<3>(mixed_anion_candidates[i]) << "\n";
    }
    
    // Doped Fluoride Variants
    std::cout << "\n\n=== Doped Fluoride Variants ===\n";
    std::cout << "Filtering for materials with:\n";
    std::cout << "  (Fe AND F AND Na) OR (Zn AND F AND Na)\n";
    std::cout << "decomp_energy: [-999.0, 0.05], bandgap: [3.0, 999.0], abundance_score: [20.0, 999999.0], n_elements: [3, 7]\n\n";
    
    std::vector<std::tuple<double, double, double, std::string>> doped_fluoride_candidates;
    
    for (size_t i = 0; i < materials.size(); ++i) {
        const auto& mat = materials[i];
        
        // Check numeric criteria
        if (mat.decomp_energy < -999.0 || mat.decomp_energy > 0.05) continue;
        if (mat.bandgap < 3.0 || mat.bandgap > 999.0) continue;
        if (mat.abundance_score < 20.0) continue;
        if (mat.n_elements < 3 || mat.n_elements > 7) continue;
        
        // Check if formula contains Na
        if (mat.formula.find("Na") == std::string::npos) continue;
        
        // Check if formula contains F
        if (mat.formula.find("F") == std::string::npos) continue;
        
        // Check if formula contains (Fe AND F AND Na) OR (Zn AND F AND Na)
        bool has_fe = (mat.formula.find("Fe") != std::string::npos);
        bool has_zn = (mat.formula.find("Zn") != std::string::npos);
        
        if (!has_fe && !has_zn) continue;
        
        doped_fluoride_candidates.push_back({mat.decomp_energy, mat.bandgap, mat.abundance_score, mat.formula});
    }
    
    std::cout << "Found " << doped_fluoride_candidates.size() << " doped fluoride variants\n\n";
    
    // Sort by decomp_energy ascending (most stable first)
    std::sort(doped_fluoride_candidates.begin(), doped_fluoride_candidates.end(), 
        [](const auto& a, const auto& b) {
            return std::get<0>(a) < std::get<0>(b);
        });
    
    // Print top 30 candidates
    int limit4 = std::min(30, static_cast<int>(doped_fluoride_candidates.size()));
    std::cout << "Top " << limit4 << " candidates sorted by decomp_energy ascending:\n";
    std::cout << std::setw(4) << "Rank" << " | " << std::setw(12) << "Decomp (eV)" << " | " 
              << std::setw(10) << "Bandgap (eV)" << " | " << std::setw(12) << "Abund (ppm)" << " | Formula\n";
    std::cout << std::string(80, '-') << "\n";
    
    for (int i = 0; i < limit4; ++i) {
        std::cout << std::setw(4) << (i + 1) << " | " 
                  << std::fixed << std::setprecision(4) << std::setw(12) << std::get<0>(doped_fluoride_candidates[i]) << " | "
                  << std::fixed << std::setprecision(3) << std::setw(10) << std::get<1>(doped_fluoride_candidates[i]) << " | "
                  << std::fixed << std::setprecision(2) << std::setw(12) << std::get<2>(doped_fluoride_candidates[i]) << " | "
                  << std::get<3>(doped_fluoride_candidates[i]) << "\n";
    }
    
    // Earth-Abundant Room-Temp Semiconductors
    std::cout << "\n\n=== Earth-Abundant Room-Temp Semiconductors ===\n";
    std::cout << "Filtering for materials with:\n";
    std::cout << "  bandgap: [0.5, 1.8] eV (semiconductor window)\n";
    std::cout << "  decomp_energy: [-999.0, 0.02] (stricter stability)\n";
    std::cout << "  abundance_score: [100.0, 999999.0] ppm (truly cheap)\n";
    std::cout << "  n_elements: [2, 5]\n";
    std::cout << "  NO Li, Na, K, Rb, Cs in formula (pure semiconductors)\n\n";
    
    std::vector<std::tuple<double, double, double, std::string>> semiconductor_candidates;
    std::string excluded_ions = "LiNaKRbCs";
    
    for (size_t i = 0; i < materials.size(); ++i) {
        const auto& mat = materials[i];
        
        // Check numeric criteria
        if (mat.decomp_energy < -999.0 || mat.decomp_energy > 0.02) continue;
        if (mat.bandgap < 0.5 || mat.bandgap > 1.8) continue;
        if (mat.abundance_score < 100.0) continue;
        if (mat.n_elements < 2 || mat.n_elements > 5) continue;
        
        // Check if formula contains NO Li, Na, K, Rb, Cs
        bool has_excluded_ion = false;
        for (char c : excluded_ions) {
            std::string ion(1, c);
            if (mat.formula.find(ion) != std::string::npos) {
                has_excluded_ion = true;
                break;
            }
        }
        if (has_excluded_ion) continue;
        
        semiconductor_candidates.push_back({mat.decomp_energy, mat.bandgap, mat.abundance_score, mat.formula});
    }
    
    std::cout << "Found " << semiconductor_candidates.size() << " semiconductor candidates\n\n";
    
    // Sort by abundance_score descending (cheapest first)
    std::sort(semiconductor_candidates.begin(), semiconductor_candidates.end(), 
        [](const auto& a, const auto& b) {
            return std::get<2>(a) > std::get<2>(b);
        });
    
    // Print top 30 candidates
    int limit5 = std::min(30, static_cast<int>(semiconductor_candidates.size()));
    std::cout << "Top " << limit5 << " candidates sorted by abundance_score descending:\n";
    std::cout << std::setw(4) << "Rank" << " | " << std::setw(12) << "Decomp (eV)" << " | " 
              << std::setw(10) << "Bandgap (eV)" << " | " << std::setw(12) << "Abund (ppm)" << " | Formula\n";
    std::cout << std::string(80, '-') << "\n";
    
    for (int i = 0; i < limit5; ++i) {
        std::cout << std::setw(4) << (i + 1) << " | " 
                  << std::fixed << std::setprecision(4) << std::setw(12) << std::get<0>(semiconductor_candidates[i]) << " | "
                  << std::fixed << std::setprecision(3) << std::setw(10) << std::get<1>(semiconductor_candidates[i]) << " | "
                  << std::fixed << std::setprecision(2) << std::setw(12) << std::get<2>(semiconductor_candidates[i]) << " | "
                  << std::get<3>(semiconductor_candidates[i]) << "\n";
    }
    
    // Potential Superconductor Candidates
    std::cout << "\n\n=== Potential Superconductor Candidates ===\n";
    std::cout << "Filtering for materials with:\n";
    std::cout << "  bandgap: [0.0, 0.05] eV (metallic)\n";
    std::cout << "  decomp_energy: [-999.0, 0.02] (strict stability)\n";
    std::cout << "  abundance_score: [20.0, 999999.0] ppm\n";
    std::cout << "  n_elements: [2, 5]\n";
    std::cout << "  Contains at least one of: Cu, Fe, Mg, Nb, V, Ti, Mn\n";
    std::cout << "  NSites > 4 (not trivial binary metals)\n\n";
    
    // Load full CSV to access NSites and Crystal System columns
    std::ifstream full_csv("data/gnome.csv");
    std::string line;
    std::getline(full_csv, line);  // Skip header
    
    std::vector<std::tuple<double, double, double, std::string, std::string>> superconductor_candidates;
    std::string superconductor_elements = "CuFeMgNbVTiMn";
    
    while (std::getline(full_csv, line)) {
        if (line.empty()) continue;
        
        // Parse CSV line
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string field;
        bool in_quotes = false;
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
        
        if (fields.size() < 18) continue;
        
        // Extract relevant columns
        std::string formula = fields[3];
        std::string decomp_str = fields[15];
        std::string bandgap_str = fields[17];
        std::string nsites_str = fields[5];
        std::string crystal_system = fields[11];
        std::string elements_str = fields[4];
        
        // Parse numeric values
        double decomp_energy = 0.0;
        double bandgap = 0.0;
        int nsites = 0;
        
        try {
            decomp_energy = std::stod(decomp_str);
            bandgap = std::stod(bandgap_str);
            nsites = std::stoi(nsites_str);
        } catch (...) {
            continue;
        }
        
        // Check numeric criteria
        if (decomp_energy < -999.0 || decomp_energy > 0.02) continue;
        if (bandgap < 0.0 || bandgap > 0.05) continue;
        if (nsites <= 4) continue;
        
        // Count elements
        int n_elements = 0;
        bool in_quote = false;
        for (char c : elements_str) {
            if (c == '\'') {
                in_quote = !in_quote;
                if (!in_quote) n_elements++;
            }
        }
        if (n_elements < 2 || n_elements > 5) continue;
        
        // Check if formula contains at least one superconductor element
        bool has_superconductor_elem = false;
        for (char c : superconductor_elements) {
            std::string elem(1, c);
            if (formula.find(elem) != std::string::npos) {
                has_superconductor_elem = true;
                break;
            }
        }
        if (!has_superconductor_elem) continue;
        
        // Compute abundance score from elements
        std::vector<std::string> elements;
        std::string elem_current;
        bool elem_in_quote = false;
        for (char c : elements_str) {
            if (c == '\'') {
                elem_in_quote = !elem_in_quote;
                if (!elem_in_quote && !elem_current.empty()) {
                    elements.push_back(elem_current);
                    elem_current.clear();
                }
            } else if (elem_in_quote) {
                elem_current += c;
            }
        }
        
        // Simple abundance lookup (same as loader)
        double min_abundance = 999999.0;
        for (const auto& elem : elements) {
            double ppm = 0.001;  // Default for rare elements
            if (elem == "O") ppm = 461000;
            else if (elem == "Si") ppm = 277200;
            else if (elem == "Al") ppm = 81300;
            else if (elem == "Fe") ppm = 50000;
            else if (elem == "Ca") ppm = 36300;
            else if (elem == "Na") ppm = 28300;
            else if (elem == "K") ppm = 25900;
            else if (elem == "Mg") ppm = 20900;
            else if (elem == "Ti") ppm = 6200;
            else if (elem == "H") ppm = 1400;
            else if (elem == "P") ppm = 1050;
            else if (elem == "Mn") ppm = 950;
            else if (elem == "F") ppm = 950;
            else if (elem == "Ba") ppm = 425;
            else if (elem == "Sr") ppm = 370;
            else if (elem == "S") ppm = 260;
            else if (elem == "C") ppm = 200;
            else if (elem == "Zr") ppm = 165;
            else if (elem == "Cl") ppm = 130;
            else if (elem == "V") ppm = 120;
            else if (elem == "Cr") ppm = 100;
            else if (elem == "Rb") ppm = 90;
            else if (elem == "Ni") ppm = 80;
            else if (elem == "Zn") ppm = 70;
            else if (elem == "Cu") ppm = 60;
            else if (elem == "Co") ppm = 25;
            else if (elem == "Li") ppm = 20;
            else if (elem == "N") ppm = 19;
            else if (elem == "Nb") ppm = 17;
            min_abundance = std::min(min_abundance, ppm);
        }
        
        if (min_abundance < 20.0) continue;
        
        superconductor_candidates.push_back({decomp_energy, bandgap, min_abundance, formula, crystal_system});
    }
    
    std::cout << "Found " << superconductor_candidates.size() << " superconductor candidates\n\n";
    
    // Sort by decomp_energy ascending (most stable first)
    std::sort(superconductor_candidates.begin(), superconductor_candidates.end(), 
        [](const auto& a, const auto& b) {
            return std::get<0>(a) < std::get<0>(b);
        });
    
    // Print top 30 candidates
    int limit6 = std::min(30, static_cast<int>(superconductor_candidates.size()));
    std::cout << "Top " << limit6 << " candidates sorted by decomp_energy ascending:\n";
    std::cout << std::setw(4) << "Rank" << " | " << std::setw(12) << "Decomp (eV)" << " | " 
              << std::setw(10) << "Bandgap (eV)" << " | " << std::setw(12) << "Abund (ppm)" << " | " 
              << std::setw(15) << "Crystal System" << " | Formula\n";
    std::cout << std::string(95, '-') << "\n";
    
    for (int i = 0; i < limit6; ++i) {
        std::cout << std::setw(4) << (i + 1) << " | " 
                  << std::fixed << std::setprecision(4) << std::setw(12) << std::get<0>(superconductor_candidates[i]) << " | "
                  << std::fixed << std::setprecision(3) << std::setw(10) << std::get<1>(superconductor_candidates[i]) << " | "
                  << std::fixed << std::setprecision(2) << std::setw(12) << std::get<2>(superconductor_candidates[i]) << " | "
                  << std::setw(15) << std::get<4>(superconductor_candidates[i]) << " | "
                  << std::get<3>(superconductor_candidates[i]) << "\n";
    }
    
    // High-Symmetry Superconductor Candidates
    std::cout << "\n\n=== High-Symmetry Superconductor Candidates ===\n";
    std::cout << "Filtering for materials with:\n";
    std::cout << "  bandgap: [0.0, 0.01] eV (truly metallic)\n";
    std::cout << "  decomp_energy: [-999.0, 0.02] (strict stability)\n";
    std::cout << "  abundance_score: [50.0, 999999.0] ppm\n";
    std::cout << "  n_elements: [3, 5]\n";
    std::cout << "  Crystal system: tetragonal OR orthorhombic OR hexagonal OR cubic\n";
    std::cout << "  Contains Cu OR (contains Mn AND F) OR (contains Mg AND F)\n";
    std::cout << "  NSites > 4 (not trivial binary metals)\n\n";
    
    // Reuse the CSV file handle (need to reopen)
    std::ifstream refined_csv("data/gnome.csv");
    std::getline(refined_csv, line);  // Skip header
    
    std::vector<std::tuple<double, double, double, std::string, std::string>> refined_superconductor_candidates;
    std::string allowed_crystal_systems = "tetragonalorthorhombichexagonalcubic";
    
    while (std::getline(refined_csv, line)) {
        if (line.empty()) continue;
        
        // Parse CSV line
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string field;
        bool in_quotes = false;
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
        
        if (fields.size() < 18) continue;
        
        // Extract relevant columns
        std::string formula = fields[3];
        std::string decomp_str = fields[15];
        std::string bandgap_str = fields[17];
        std::string nsites_str = fields[5];
        std::string crystal_system = fields[11];
        std::string elements_str = fields[4];
        
        // Parse numeric values
        double decomp_energy = 0.0;
        double bandgap = 0.0;
        int nsites = 0;
        
        try {
            decomp_energy = std::stod(decomp_str);
            bandgap = std::stod(bandgap_str);
            nsites = std::stoi(nsites_str);
        } catch (...) {
            continue;
        }
        
        // Check numeric criteria
        if (decomp_energy < -999.0 || decomp_energy > 0.02) continue;
        if (bandgap < 0.0 || bandgap > 0.01) continue;
        if (nsites <= 4) continue;
        
        // Count elements
        int n_elements = 0;
        bool in_quote = false;
        for (char c : elements_str) {
            if (c == '\'') {
                in_quote = !in_quote;
                if (!in_quote) n_elements++;
            }
        }
        if (n_elements < 3 || n_elements > 5) continue;
        
        // Check crystal system
        if (allowed_crystal_systems.find(crystal_system) == std::string::npos) continue;
        
        // Check if formula contains Cu OR (Mn AND F) OR (Mg AND F)
        bool has_cu = (formula.find("Cu") != std::string::npos);
        bool has_mn = (formula.find("Mn") != std::string::npos);
        bool has_f = (formula.find("F") != std::string::npos);
        bool has_mg = (formula.find("Mg") != std::string::npos);
        
        bool valid_composition = has_cu || (has_mn && has_f) || (has_mg && has_f);
        if (!valid_composition) continue;
        
        // Compute abundance score from elements
        std::vector<std::string> elements;
        std::string elem_current;
        bool elem_in_quote = false;
        for (char c : elements_str) {
            if (c == '\'') {
                elem_in_quote = !elem_in_quote;
                if (!elem_in_quote && !elem_current.empty()) {
                    elements.push_back(elem_current);
                    elem_current.clear();
                }
            } else if (elem_in_quote) {
                elem_current += c;
            }
        }
        
        // Simple abundance lookup (same as loader)
        double min_abundance = 999999.0;
        for (const auto& elem : elements) {
            double ppm = 0.001;  // Default for rare elements
            if (elem == "O") ppm = 461000;
            else if (elem == "Si") ppm = 277200;
            else if (elem == "Al") ppm = 81300;
            else if (elem == "Fe") ppm = 50000;
            else if (elem == "Ca") ppm = 36300;
            else if (elem == "Na") ppm = 28300;
            else if (elem == "K") ppm = 25900;
            else if (elem == "Mg") ppm = 20900;
            else if (elem == "Ti") ppm = 6200;
            else if (elem == "H") ppm = 1400;
            else if (elem == "P") ppm = 1050;
            else if (elem == "Mn") ppm = 950;
            else if (elem == "F") ppm = 950;
            else if (elem == "Ba") ppm = 425;
            else if (elem == "Sr") ppm = 370;
            else if (elem == "S") ppm = 260;
            else if (elem == "C") ppm = 200;
            else if (elem == "Zr") ppm = 165;
            else if (elem == "Cl") ppm = 130;
            else if (elem == "V") ppm = 120;
            else if (elem == "Cr") ppm = 100;
            else if (elem == "Rb") ppm = 90;
            else if (elem == "Ni") ppm = 80;
            else if (elem == "Zn") ppm = 70;
            else if (elem == "Cu") ppm = 60;
            else if (elem == "Co") ppm = 25;
            else if (elem == "Li") ppm = 20;
            else if (elem == "N") ppm = 19;
            else if (elem == "Nb") ppm = 17;
            min_abundance = std::min(min_abundance, ppm);
        }
        
        if (min_abundance < 50.0) continue;
        
        refined_superconductor_candidates.push_back({decomp_energy, bandgap, min_abundance, formula, crystal_system});
    }
    
    std::cout << "Found " << refined_superconductor_candidates.size() << " high-symmetry superconductor candidates\n\n";
    
    // Sort by decomp_energy ascending (most stable first)
    std::sort(refined_superconductor_candidates.begin(), refined_superconductor_candidates.end(), 
        [](const auto& a, const auto& b) {
            return std::get<0>(a) < std::get<0>(b);
        });
    
    // Print top 20 candidates
    int limit7 = std::min(20, static_cast<int>(refined_superconductor_candidates.size()));
    std::cout << "Top " << limit7 << " candidates sorted by decomp_energy ascending:\n";
    std::cout << std::setw(4) << "Rank" << " | " << std::setw(12) << "Decomp (eV)" << " | " 
              << std::setw(10) << "Bandgap (eV)" << " | " << std::setw(12) << "Abund (ppm)" << " | " 
              << std::setw(15) << "Crystal System" << " | Formula\n";
    std::cout << std::string(95, '-') << "\n";
    
    for (int i = 0; i < limit7; ++i) {
        std::cout << std::setw(4) << (i + 1) << " | " 
                  << std::fixed << std::setprecision(4) << std::setw(12) << std::get<0>(refined_superconductor_candidates[i]) << " | "
                  << std::fixed << std::setprecision(3) << std::setw(10) << std::get<1>(refined_superconductor_candidates[i]) << " | "
                  << std::fixed << std::setprecision(2) << std::setw(12) << std::get<2>(refined_superconductor_candidates[i]) << " | "
                  << std::setw(15) << std::get<4>(refined_superconductor_candidates[i]) << " | "
                  << std::get<3>(refined_superconductor_candidates[i]) << "\n";
    }
    
    // Query 8: Property-First Superconductor
    std::cout << "\n\n=== Query 8: Property-First Superconductor ===\n";
    std::cout << "Filtering for materials with:\n";
    std::cout << "  bandgap: [0.0, 0.001] eV (as close to perfect metal as GNoME computes)\n";
    std::cout << "  decomp_energy: [-999.0, 0.01]\n";
    std::cout << "  abundance_score: [200.0, 999999.0] ppm (truly cheap)\n";
    std::cout << "  crystal_system: tetragonal OR hexagonal OR cubic (layer-friendly)\n";
    std::cout << "  NSites: [4, 12] (not too complex to synthesize)\n";
    std::cout << "  n_elements: [2, 5]\n\n";
    
    // Reopen CSV for new query
    std::ifstream prop_first_csv("data/gnome.csv");
    std::getline(prop_first_csv, line);  // Skip header
    
    std::vector<std::tuple<double, double, double, std::string, std::string, int>> prop_first_candidates;
    std::string allowed_crystal_prop = "tetragonalhexagonalcubic";
    
    while (std::getline(prop_first_csv, line)) {
        if (line.empty()) continue;
        
        // Parse CSV line
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string field;
        bool in_quotes = false;
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
        
        if (fields.size() < 18) continue;
        
        // Extract relevant columns
        std::string formula = fields[3];
        std::string decomp_str = fields[15];
        std::string bandgap_str = fields[17];
        std::string nsites_str = fields[5];
        std::string crystal_system = fields[11];
        std::string elements_str = fields[4];
        
        // Parse numeric values
        double decomp_energy = 0.0;
        double bandgap = 0.0;
        int nsites = 0;
        
        try {
            decomp_energy = std::stod(decomp_str);
            bandgap = std::stod(bandgap_str);
            nsites = std::stoi(nsites_str);
        } catch (...) {
            continue;
        }
        
        // Check numeric criteria
        if (decomp_energy < -999.0 || decomp_energy > 0.01) continue;
        if (bandgap < 0.0 || bandgap > 0.001) continue;
        if (nsites < 4 || nsites > 12) continue;
        
        // Check crystal system
        if (allowed_crystal_prop.find(crystal_system) == std::string::npos) continue;
        
        // Count elements
        int n_elements = 0;
        bool in_quote = false;
        for (char c : elements_str) {
            if (c == '\'') {
                in_quote = !in_quote;
                if (!in_quote) n_elements++;
            }
        }
        if (n_elements < 2 || n_elements > 5) continue;
        
        // Compute abundance score from elements
        std::vector<std::string> elements;
        std::string elem_current;
        bool elem_in_quote = false;
        for (char c : elements_str) {
            if (c == '\'') {
                elem_in_quote = !elem_in_quote;
                if (!elem_in_quote && !elem_current.empty()) {
                    elements.push_back(elem_current);
                    elem_current.clear();
                }
            } else if (elem_in_quote) {
                elem_current += c;
            }
        }
        
        // Simple abundance lookup (same as loader)
        double min_abundance = 999999.0;
        for (const auto& elem : elements) {
            double ppm = 0.001;  // Default for rare elements
            if (elem == "O") ppm = 461000;
            else if (elem == "Si") ppm = 277200;
            else if (elem == "Al") ppm = 81300;
            else if (elem == "Fe") ppm = 50000;
            else if (elem == "Ca") ppm = 36300;
            else if (elem == "Na") ppm = 28300;
            else if (elem == "K") ppm = 25900;
            else if (elem == "Mg") ppm = 20900;
            else if (elem == "Ti") ppm = 6200;
            else if (elem == "H") ppm = 1400;
            else if (elem == "P") ppm = 1050;
            else if (elem == "Mn") ppm = 950;
            else if (elem == "F") ppm = 950;
            else if (elem == "Ba") ppm = 425;
            else if (elem == "Sr") ppm = 370;
            else if (elem == "S") ppm = 260;
            else if (elem == "C") ppm = 200;
            else if (elem == "Zr") ppm = 165;
            else if (elem == "Cl") ppm = 130;
            else if (elem == "V") ppm = 120;
            else if (elem == "Cr") ppm = 100;
            else if (elem == "Rb") ppm = 90;
            else if (elem == "Ni") ppm = 80;
            else if (elem == "Zn") ppm = 70;
            else if (elem == "Cu") ppm = 60;
            else if (elem == "Co") ppm = 25;
            else if (elem == "Li") ppm = 20;
            else if (elem == "N") ppm = 19;
            else if (elem == "Nb") ppm = 17;
            min_abundance = std::min(min_abundance, ppm);
        }
        
        if (min_abundance < 200.0) continue;
        
        prop_first_candidates.push_back({decomp_energy, bandgap, min_abundance, formula, crystal_system, n_elements});
    }
    
    std::cout << "Found " << prop_first_candidates.size() << " property-first superconductor candidates\n\n";
    
    // Sort by abundance_score descending, then decomp_energy ascending
    std::sort(prop_first_candidates.begin(), prop_first_candidates.end(), 
        [](const auto& a, const auto& b) {
            if (std::get<2>(a) != std::get<2>(b)) {
                return std::get<2>(a) > std::get<2>(b);
            }
            return std::get<0>(a) < std::get<0>(b);
        });
    
    // Print top 20 candidates
    int limit8 = std::min(20, static_cast<int>(prop_first_candidates.size()));
    std::cout << "Top " << limit8 << " candidates sorted by abundance_score descending, then decomp_energy ascending:\n";
    std::cout << std::setw(4) << "Rank" << " | " << std::setw(12) << "Decomp (eV)" << " | " 
              << std::setw(10) << "Bandgap (eV)" << " | " << std::setw(12) << "Abund (ppm)" << " | " 
              << std::setw(15) << "Crystal System" << " | " << std::setw(8) << "n_elem" << " | Formula\n";
    std::cout << std::string(105, '-') << "\n";
    
    for (int i = 0; i < limit8; ++i) {
        std::cout << std::setw(4) << (i + 1) << " | " 
                  << std::fixed << std::setprecision(4) << std::setw(12) << std::get<0>(prop_first_candidates[i]) << " | "
                  << std::fixed << std::setprecision(3) << std::setw(10) << std::get<1>(prop_first_candidates[i]) << " | "
                  << std::fixed << std::setprecision(2) << std::setw(12) << std::get<2>(prop_first_candidates[i]) << " | "
                  << std::setw(15) << std::get<4>(prop_first_candidates[i]) << " | "
                  << std::setw(8) << std::get<5>(prop_first_candidates[i]) << " | "
                  << std::get<3>(prop_first_candidates[i]) << "\n";
    }
    
    // Query 9: Extreme Bandgap
    std::cout << "\n\n=== Query 9: Extreme Bandgap ===\n";
    std::cout << "Filtering for materials with:\n";
    std::cout << "  bandgap: [6.0, 999.0] eV (extreme bandgap)\n";
    std::cout << "  decomp_energy: [-999.0, 0.02]\n";
    std::cout << "  abundance_score: [100.0, 999999.0] ppm\n";
    std::cout << "  n_elements: [2, 5]\n\n";
    
    // Reopen CSV for new query
    std::ifstream extreme_bandgap_csv("data/gnome.csv");
    std::getline(extreme_bandgap_csv, line);  // Skip header
    
    std::vector<std::tuple<double, double, double, std::string, std::string>> extreme_bandgap_candidates;
    
    while (std::getline(extreme_bandgap_csv, line)) {
        if (line.empty()) continue;
        
        // Parse CSV line
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string field;
        bool in_quotes = false;
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
        
        if (fields.size() < 18) continue;
        
        // Extract relevant columns
        std::string formula = fields[3];
        std::string decomp_str = fields[15];
        std::string bandgap_str = fields[17];
        std::string crystal_system = fields[11];
        std::string elements_str = fields[4];
        
        // Parse numeric values
        double decomp_energy = 0.0;
        double bandgap = 0.0;
        
        try {
            decomp_energy = std::stod(decomp_str);
            bandgap = std::stod(bandgap_str);
        } catch (...) {
            continue;
        }
        
        // Check numeric criteria
        if (decomp_energy < -999.0 || decomp_energy > 0.02) continue;
        if (bandgap < 6.0 || bandgap > 999.0) continue;
        
        // Count elements
        int n_elements = 0;
        bool in_quote = false;
        for (char c : elements_str) {
            if (c == '\'') {
                in_quote = !in_quote;
                if (!in_quote) n_elements++;
            }
        }
        if (n_elements < 2 || n_elements > 5) continue;
        
        // Compute abundance score from elements
        std::vector<std::string> elements;
        std::string elem_current;
        bool elem_in_quote = false;
        for (char c : elements_str) {
            if (c == '\'') {
                elem_in_quote = !elem_in_quote;
                if (!elem_in_quote && !elem_current.empty()) {
                    elements.push_back(elem_current);
                    elem_current.clear();
                }
            } else if (elem_in_quote) {
                elem_current += c;
            }
        }
        
        // Simple abundance lookup (same as loader)
        double min_abundance = 999999.0;
        for (const auto& elem : elements) {
            double ppm = 0.001;  // Default for rare elements
            if (elem == "O") ppm = 461000;
            else if (elem == "Si") ppm = 277200;
            else if (elem == "Al") ppm = 81300;
            else if (elem == "Fe") ppm = 50000;
            else if (elem == "Ca") ppm = 36300;
            else if (elem == "Na") ppm = 28300;
            else if (elem == "K") ppm = 25900;
            else if (elem == "Mg") ppm = 20900;
            else if (elem == "Ti") ppm = 6200;
            else if (elem == "H") ppm = 1400;
            else if (elem == "P") ppm = 1050;
            else if (elem == "Mn") ppm = 950;
            else if (elem == "F") ppm = 950;
            else if (elem == "Ba") ppm = 425;
            else if (elem == "Sr") ppm = 370;
            else if (elem == "S") ppm = 260;
            else if (elem == "C") ppm = 200;
            else if (elem == "Zr") ppm = 165;
            else if (elem == "Cl") ppm = 130;
            else if (elem == "V") ppm = 120;
            else if (elem == "Cr") ppm = 100;
            else if (elem == "Rb") ppm = 90;
            else if (elem == "Ni") ppm = 80;
            else if (elem == "Zn") ppm = 70;
            else if (elem == "Cu") ppm = 60;
            else if (elem == "Co") ppm = 25;
            else if (elem == "Li") ppm = 20;
            else if (elem == "N") ppm = 19;
            else if (elem == "Nb") ppm = 17;
            min_abundance = std::min(min_abundance, ppm);
        }
        
        if (min_abundance < 100.0) continue;
        
        extreme_bandgap_candidates.push_back({decomp_energy, bandgap, min_abundance, formula, crystal_system});
    }
    
    std::cout << "Found " << extreme_bandgap_candidates.size() << " extreme bandgap candidates\n\n";
    
    // Sort by bandgap descending
    std::sort(extreme_bandgap_candidates.begin(), extreme_bandgap_candidates.end(), 
        [](const auto& a, const auto& b) {
            return std::get<1>(a) > std::get<1>(b);
        });
    
    // Print top 20 candidates
    int limit9 = std::min(20, static_cast<int>(extreme_bandgap_candidates.size()));
    std::cout << "Top " << limit9 << " candidates sorted by bandgap descending:\n";
    std::cout << std::setw(4) << "Rank" << " | " << std::setw(12) << "Decomp (eV)" << " | " 
              << std::setw(10) << "Bandgap (eV)" << " | " << std::setw(12) << "Abund (ppm)" << " | " 
              << std::setw(15) << "Crystal System" << " | Formula\n";
    std::cout << std::string(95, '-') << "\n";
    
    for (int i = 0; i < limit9; ++i) {
        std::cout << std::setw(4) << (i + 1) << " | " 
                  << std::fixed << std::setprecision(4) << std::setw(12) << std::get<0>(extreme_bandgap_candidates[i]) << " | "
                  << std::fixed << std::setprecision(3) << std::setw(10) << std::get<1>(extreme_bandgap_candidates[i]) << " | "
                  << std::fixed << std::setprecision(2) << std::setw(12) << std::get<2>(extreme_bandgap_candidates[i]) << " | "
                  << std::setw(15) << std::get<4>(extreme_bandgap_candidates[i]) << " | "
                  << std::get<3>(extreme_bandgap_candidates[i]) << "\n";
    }
    
    // Query 10: Best Remaining Semiconductors
    std::cout << "\n\n=== Query 10: Best Remaining Semiconductors ===\n";
    std::cout << "Filtering from semiconductor candidates with:\n";
    std::cout << "  bandgap: [1.0, 1.5] eV (strict solar optimal)\n";
    std::cout << "  decomp_energy: [-999.0, 0.01]\n";
    std::cout << "  abundance_score: [200.0, 999999.0] ppm\n";
    std::cout << "  NO Sr (diversity filter)\n\n";
    
    // Filter from existing semiconductor_candidates
    std::vector<std::tuple<double, double, double, std::string>> best_semiconductor_candidates;
    for (const auto& candidate : semiconductor_candidates) {
        double decomp_energy = std::get<0>(candidate);
        double bandgap = std::get<1>(candidate);
        double abundance_score = std::get<2>(candidate);
        std::string formula = std::get<3>(candidate);
        
        // Check criteria
        if (bandgap < 1.0 || bandgap > 1.5) continue;
        if (decomp_energy < -999.0 || decomp_energy > 0.01) continue;
        if (abundance_score < 200.0) continue;
        if (formula.find("Sr") != std::string::npos) continue;
        
        best_semiconductor_candidates.push_back({decomp_energy, bandgap, abundance_score, formula});
    }
    
    std::cout << "Found " << best_semiconductor_candidates.size() << " best semiconductor candidates\n\n";
    
    // Sort by abundance_score descending
    std::sort(best_semiconductor_candidates.begin(), best_semiconductor_candidates.end(), 
        [](const auto& a, const auto& b) {
            return std::get<2>(a) > std::get<2>(b);
        });
    
    // Print top 10 candidates
    int limit10 = std::min(10, static_cast<int>(best_semiconductor_candidates.size()));
    std::cout << "Top " << limit10 << " candidates sorted by abundance_score descending:\n";
    std::cout << std::setw(4) << "Rank" << " | " << std::setw(12) << "Decomp (eV)" << " | " 
              << std::setw(10) << "Bandgap (eV)" << " | " << std::setw(12) << "Abund (ppm)" << " | Formula\n";
    std::cout << std::string(80, '-') << "\n";
    
    for (int i = 0; i < limit10; ++i) {
        std::cout << std::setw(4) << (i + 1) << " | " 
                  << std::fixed << std::setprecision(4) << std::setw(12) << std::get<0>(best_semiconductor_candidates[i]) << " | "
                  << std::fixed << std::setprecision(3) << std::setw(10) << std::get<1>(best_semiconductor_candidates[i]) << " | "
                  << std::fixed << std::setprecision(2) << std::setw(12) << std::get<2>(best_semiconductor_candidates[i]) << " | "
                  << std::get<3>(best_semiconductor_candidates[i]) << "\n";
    }
    
    // Query 11: Property-First Battery Electrolyte
    std::cout << "\n\n=== Query 11: Property-First Battery Electrolyte ===\n";
    std::cout << "Filtering for materials with:\n";
    std::cout << "  bandgap: [4.0, 999.0] eV (wide window)\n";
    std::cout << "  decomp_energy: [-999.0, 0.02]\n";
    std::cout << "  abundance_score: [500.0, 999999.0] ppm (very cheap only)\n";
    std::cout << "  n_elements: [3, 5]\n";
    std::cout << "  NSites: [4, 15]\n\n";
    
    // Reopen CSV for new query
    std::ifstream electrolyte_csv("data/gnome.csv");
    std::getline(electrolyte_csv, line);  // Skip header
    
    std::vector<std::tuple<double, double, double, std::string>> electrolyte_candidates;
    
    while (std::getline(electrolyte_csv, line)) {
        if (line.empty()) continue;
        
        // Parse CSV line
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string field;
        bool in_quotes = false;
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
        
        if (fields.size() < 18) continue;
        
        // Extract relevant columns
        std::string formula = fields[3];
        std::string decomp_str = fields[15];
        std::string bandgap_str = fields[17];
        std::string nsites_str = fields[5];
        std::string elements_str = fields[4];
        
        // Parse numeric values
        double decomp_energy = 0.0;
        double bandgap = 0.0;
        int nsites = 0;
        
        try {
            decomp_energy = std::stod(decomp_str);
            bandgap = std::stod(bandgap_str);
            nsites = std::stoi(nsites_str);
        } catch (...) {
            continue;
        }
        
        // Check numeric criteria
        if (decomp_energy < -999.0 || decomp_energy > 0.02) continue;
        if (bandgap < 4.0 || bandgap > 999.0) continue;
        if (nsites < 4 || nsites > 15) continue;
        
        // Count elements
        int n_elements = 0;
        bool in_quote = false;
        for (char c : elements_str) {
            if (c == '\'') {
                in_quote = !in_quote;
                if (!in_quote) n_elements++;
            }
        }
        if (n_elements < 3 || n_elements > 5) continue;
        
        // Compute abundance score from elements
        std::vector<std::string> elements;
        std::string elem_current;
        bool elem_in_quote = false;
        for (char c : elements_str) {
            if (c == '\'') {
                elem_in_quote = !elem_in_quote;
                if (!elem_in_quote && !elem_current.empty()) {
                    elements.push_back(elem_current);
                    elem_current.clear();
                }
            } else if (elem_in_quote) {
                elem_current += c;
            }
        }
        
        // Simple abundance lookup (same as loader)
        double min_abundance = 999999.0;
        for (const auto& elem : elements) {
            double ppm = 0.001;  // Default for rare elements
            if (elem == "O") ppm = 461000;
            else if (elem == "Si") ppm = 277200;
            else if (elem == "Al") ppm = 81300;
            else if (elem == "Fe") ppm = 50000;
            else if (elem == "Ca") ppm = 36300;
            else if (elem == "Na") ppm = 28300;
            else if (elem == "K") ppm = 25900;
            else if (elem == "Mg") ppm = 20900;
            else if (elem == "Ti") ppm = 6200;
            else if (elem == "H") ppm = 1400;
            else if (elem == "P") ppm = 1050;
            else if (elem == "Mn") ppm = 950;
            else if (elem == "F") ppm = 950;
            else if (elem == "Ba") ppm = 425;
            else if (elem == "Sr") ppm = 370;
            else if (elem == "S") ppm = 260;
            else if (elem == "C") ppm = 200;
            else if (elem == "Zr") ppm = 165;
            else if (elem == "Cl") ppm = 130;
            else if (elem == "V") ppm = 120;
            else if (elem == "Cr") ppm = 100;
            else if (elem == "Rb") ppm = 90;
            else if (elem == "Ni") ppm = 80;
            else if (elem == "Zn") ppm = 70;
            else if (elem == "Cu") ppm = 60;
            else if (elem == "Co") ppm = 25;
            else if (elem == "Li") ppm = 20;
            else if (elem == "N") ppm = 19;
            else if (elem == "Nb") ppm = 17;
            min_abundance = std::min(min_abundance, ppm);
        }
        
        if (min_abundance < 500.0) continue;
        
        electrolyte_candidates.push_back({decomp_energy, bandgap, min_abundance, formula});
    }
    
    std::cout << "Found " << electrolyte_candidates.size() << " property-first electrolyte candidates\n\n";
    
    // Sort by bandgap descending
    std::sort(electrolyte_candidates.begin(), electrolyte_candidates.end(), 
        [](const auto& a, const auto& b) {
            return std::get<1>(a) > std::get<1>(b);
        });
    
    // Print top 20 candidates
    int limit11 = std::min(20, static_cast<int>(electrolyte_candidates.size()));
    std::cout << "Top " << limit11 << " candidates sorted by bandgap descending:\n";
    std::cout << std::setw(4) << "Rank" << " | " << std::setw(12) << "Decomp (eV)" << " | " 
              << std::setw(10) << "Bandgap (eV)" << " | " << std::setw(12) << "Abund (ppm)" << " | Formula\n";
    std::cout << std::string(80, '-') << "\n";
    
    for (int i = 0; i < limit11; ++i) {
        std::cout << std::setw(4) << (i + 1) << " | " 
                  << std::fixed << std::setprecision(4) << std::setw(12) << std::get<0>(electrolyte_candidates[i]) << " | "
                  << std::fixed << std::setprecision(3) << std::setw(10) << std::get<1>(electrolyte_candidates[i]) << " | "
                  << std::fixed << std::setprecision(2) << std::setw(12) << std::get<2>(electrolyte_candidates[i]) << " | "
                  << std::get<3>(electrolyte_candidates[i]) << "\n";
    }
    
    // Query 12: Ultra-Hard Ultra-Light Structural Materials
    std::cout << "\n\n=== Query 12: Ultra-Hard Ultra-Light Structural Candidates ===\n";
    std::cout << "Filtering for materials with:\n";
    std::cout << "  bandgap: [3.0, 999.0] eV (covalent bonding = hard)\n";
    std::cout << "  decomp_energy: [-999.0, 0.01] (very stable)\n";
    std::cout << "  abundance_score: [500.0, 999999.0] ppm\n";
    std::cout << "  n_elements: [2, 4]\n";
    std::cout << "  density: [0.0, 4.0] g/cm³ (light)\n";
    std::cout << "  NSites: [2, 12]\n\n";
    
    // Reopen CSV for new query
    std::ifstream structural_csv("data/gnome.csv");
    std::getline(structural_csv, line);  // Skip header
    
    std::vector<std::tuple<double, double, double, double, std::string, std::string>> structural_candidates;
    
    while (std::getline(structural_csv, line)) {
        if (line.empty()) continue;
        
        // Parse CSV line
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string field;
        bool in_quotes = false;
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
        
        if (fields.size() < 18) continue;
        
        // Extract relevant columns
        std::string formula = fields[3];
        std::string decomp_str = fields[15];
        std::string bandgap_str = fields[17];
        std::string nsites_str = fields[5];
        std::string density_str = fields[7];
        std::string crystal_system = fields[11];
        std::string elements_str = fields[4];
        
        // Parse numeric values
        double decomp_energy = 0.0;
        double bandgap = 0.0;
        int nsites = 0;
        double density = 0.0;
        
        try {
            decomp_energy = std::stod(decomp_str);
            bandgap = std::stod(bandgap_str);
            nsites = std::stoi(nsites_str);
            density = std::stod(density_str);
        } catch (...) {
            continue;
        }
        
        // Check numeric criteria
        if (decomp_energy < -999.0 || decomp_energy > 0.01) continue;
        if (bandgap < 3.0 || bandgap > 999.0) continue;
        if (density < 0.0 || density > 4.0) continue;
        if (nsites < 2 || nsites > 12) continue;
        
        // Count elements
        int n_elements = 0;
        bool in_quote = false;
        for (char c : elements_str) {
            if (c == '\'') {
                in_quote = !in_quote;
                if (!in_quote) n_elements++;
            }
        }
        if (n_elements < 2 || n_elements > 4) continue;
        
        // Compute abundance score from elements
        std::vector<std::string> elements;
        std::string elem_current;
        bool elem_in_quote = false;
        for (char c : elements_str) {
            if (c == '\'') {
                elem_in_quote = !elem_in_quote;
                if (!elem_in_quote && !elem_current.empty()) {
                    elements.push_back(elem_current);
                    elem_current.clear();
                }
            } else if (elem_in_quote) {
                elem_current += c;
            }
        }
        
        // Simple abundance lookup (same as loader)
        double min_abundance = 999999.0;
        for (const auto& elem : elements) {
            double ppm = 0.001;  // Default for rare elements
            if (elem == "O") ppm = 461000;
            else if (elem == "Si") ppm = 277200;
            else if (elem == "Al") ppm = 81300;
            else if (elem == "Fe") ppm = 50000;
            else if (elem == "Ca") ppm = 36300;
            else if (elem == "Na") ppm = 28300;
            else if (elem == "K") ppm = 25900;
            else if (elem == "Mg") ppm = 20900;
            else if (elem == "Ti") ppm = 6200;
            else if (elem == "H") ppm = 1400;
            else if (elem == "P") ppm = 1050;
            else if (elem == "Mn") ppm = 950;
            else if (elem == "F") ppm = 950;
            else if (elem == "Ba") ppm = 425;
            else if (elem == "Sr") ppm = 370;
            else if (elem == "S") ppm = 260;
            else if (elem == "C") ppm = 200;
            else if (elem == "Zr") ppm = 165;
            else if (elem == "Cl") ppm = 130;
            else if (elem == "V") ppm = 120;
            else if (elem == "Cr") ppm = 100;
            else if (elem == "Rb") ppm = 90;
            else if (elem == "Ni") ppm = 80;
            else if (elem == "Zn") ppm = 70;
            else if (elem == "Cu") ppm = 60;
            else if (elem == "Co") ppm = 25;
            else if (elem == "Li") ppm = 20;
            else if (elem == "N") ppm = 19;
            else if (elem == "Nb") ppm = 17;
            min_abundance = std::min(min_abundance, ppm);
        }
        
        if (min_abundance < 500.0) continue;
        
        structural_candidates.push_back({decomp_energy, bandgap, min_abundance, density, formula, crystal_system});
    }
    
    std::cout << "Found " << structural_candidates.size() << " ultra-hard ultra-light structural candidates\n\n";
    
    // Sort by bandgap descending (higher gap = stronger covalent bonds = harder)
    std::sort(structural_candidates.begin(), structural_candidates.end(), 
        [](const auto& a, const auto& b) {
            return std::get<1>(a) > std::get<1>(b);
        });
    
    // Print top 20 candidates
    int limit12 = std::min(20, static_cast<int>(structural_candidates.size()));
    std::cout << "Top " << limit12 << " candidates sorted by bandgap descending:\n";
    std::cout << std::setw(4) << "Rank" << " | " << std::setw(12) << "Decomp (eV)" << " | " 
              << std::setw(10) << "Bandgap (eV)" << " | " << std::setw(12) << "Abund (ppm)" << " | "
              << std::setw(10) << "Density" << " | " << std::setw(15) << "Crystal System" << " | Formula\n";
    std::cout << std::string(100, '-') << "\n";
    
    for (int i = 0; i < limit12; ++i) {
        std::cout << std::setw(4) << (i + 1) << " | " 
                  << std::fixed << std::setprecision(4) << std::setw(12) << std::get<0>(structural_candidates[i]) << " | "
                  << std::fixed << std::setprecision(3) << std::setw(10) << std::get<1>(structural_candidates[i]) << " | "
                  << std::fixed << std::setprecision(2) << std::setw(12) << std::get<2>(structural_candidates[i]) << " | "
                  << std::fixed << std::setprecision(3) << std::setw(10) << std::get<3>(structural_candidates[i]) << " | "
                  << std::setw(15) << std::get<5>(structural_candidates[i]) << " | "
                  << std::get<4>(structural_candidates[i]) << "\n";
    }
    
    // Query 13: Property-First Battery Electrolyte (True Wide Search)
    std::cout << "\n\n=== Query 13: Property-First Wide Battery Search ===\n";
    std::cout << "Filtering for materials with:\n";
    std::cout << "  bandgap: [5.0, 999.0] eV (very wide electrochemical window)\n";
    std::cout << "  decomp_energy: [-999.0, 0.01]\n";
    std::cout << "  abundance_score: [200.0, 999999.0] ppm\n";
    std::cout << "  density: [0.0, 5.0] g/cm³ (light electrolyte)\n";
    std::cout << "  n_elements: [2, 5]\n";
    std::cout << "  NSites: [3, 20]\n\n";
    
    // Reopen CSV for new query
    std::ifstream wide_battery_csv("data/gnome.csv");
    std::getline(wide_battery_csv, line);  // Skip header
    
    std::vector<std::tuple<double, double, double, double, std::string, std::string>> wide_battery_candidates;
    
    while (std::getline(wide_battery_csv, line)) {
        if (line.empty()) continue;
        
        // Parse CSV line
        std::vector<std::string> fields;
        std::stringstream ss(line);
        std::string field;
        bool in_quotes = false;
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
        
        if (fields.size() < 18) continue;
        
        // Extract relevant columns
        std::string formula = fields[3];
        std::string decomp_str = fields[15];
        std::string bandgap_str = fields[17];
        std::string nsites_str = fields[5];
        std::string density_str = fields[7];
        std::string crystal_system = fields[11];
        std::string elements_str = fields[4];
        
        // Parse numeric values
        double decomp_energy = 0.0;
        double bandgap = 0.0;
        int nsites = 0;
        double density = 0.0;
        
        try {
            decomp_energy = std::stod(decomp_str);
            bandgap = std::stod(bandgap_str);
            nsites = std::stoi(nsites_str);
            density = std::stod(density_str);
        } catch (...) {
            continue;
        }
        
        // Check numeric criteria
        if (decomp_energy < -999.0 || decomp_energy > 0.01) continue;
        if (bandgap < 5.0 || bandgap > 999.0) continue;
        if (density < 0.0 || density > 5.0) continue;
        if (nsites < 3 || nsites > 20) continue;
        
        // Count elements
        int n_elements = 0;
        bool in_quote = false;
        for (char c : elements_str) {
            if (c == '\'') {
                in_quote = !in_quote;
                if (!in_quote) n_elements++;
            }
        }
        if (n_elements < 2 || n_elements > 5) continue;
        
        // Compute abundance score from elements
        std::vector<std::string> elements;
        std::string elem_current;
        bool elem_in_quote = false;
        for (char c : elements_str) {
            if (c == '\'') {
                elem_in_quote = !elem_in_quote;
                if (!elem_in_quote && !elem_current.empty()) {
                    elements.push_back(elem_current);
                    elem_current.clear();
                }
            } else if (elem_in_quote) {
                elem_current += c;
            }
        }
        
        // Simple abundance lookup (same as loader)
        double min_abundance = 999999.0;
        for (const auto& elem : elements) {
            double ppm = 0.001;  // Default for rare elements
            if (elem == "O") ppm = 461000;
            else if (elem == "Si") ppm = 277200;
            else if (elem == "Al") ppm = 81300;
            else if (elem == "Fe") ppm = 50000;
            else if (elem == "Ca") ppm = 36300;
            else if (elem == "Na") ppm = 28300;
            else if (elem == "K") ppm = 25900;
            else if (elem == "Mg") ppm = 20900;
            else if (elem == "Ti") ppm = 6200;
            else if (elem == "H") ppm = 1400;
            else if (elem == "P") ppm = 1050;
            else if (elem == "Mn") ppm = 950;
            else if (elem == "F") ppm = 950;
            else if (elem == "Ba") ppm = 425;
            else if (elem == "Sr") ppm = 370;
            else if (elem == "S") ppm = 260;
            else if (elem == "C") ppm = 200;
            else if (elem == "Zr") ppm = 165;
            else if (elem == "Cl") ppm = 130;
            else if (elem == "V") ppm = 120;
            else if (elem == "Cr") ppm = 100;
            else if (elem == "Rb") ppm = 90;
            else if (elem == "Ni") ppm = 80;
            else if (elem == "Zn") ppm = 70;
            else if (elem == "Cu") ppm = 60;
            else if (elem == "Co") ppm = 25;
            else if (elem == "Li") ppm = 20;
            else if (elem == "N") ppm = 19;
            else if (elem == "Nb") ppm = 17;
            min_abundance = std::min(min_abundance, ppm);
        }
        
        if (min_abundance < 200.0) continue;
        
        wide_battery_candidates.push_back({decomp_energy, bandgap, min_abundance, density, formula, crystal_system});
    }
    
    std::cout << "Found " << wide_battery_candidates.size() << " property-first wide battery candidates\n\n";
    
    // Sort by (bandgap × abundance_score) descending
    std::sort(wide_battery_candidates.begin(), wide_battery_candidates.end(), 
        [](const auto& a, const auto& b) {
            double score_a = std::get<1>(a) * std::get<2>(a);
            double score_b = std::get<1>(b) * std::get<2>(b);
            return score_a > score_b;
        });
    
    // Print top 20 candidates
    int limit13 = std::min(20, static_cast<int>(wide_battery_candidates.size()));
    std::cout << "Top " << limit13 << " candidates sorted by (bandgap × abundance_score) descending:\n";
    std::cout << std::setw(4) << "Rank" << " | " << std::setw(12) << "Decomp (eV)" << " | " 
              << std::setw(10) << "Bandgap (eV)" << " | " << std::setw(12) << "Abund (ppm)" << " | "
              << std::setw(10) << "Density" << " | " << std::setw(15) << "Crystal System" << " | Formula\n";
    std::cout << std::string(100, '-') << "\n";
    
    for (int i = 0; i < limit13; ++i) {
        std::cout << std::setw(4) << (i + 1) << " | " 
                  << std::fixed << std::setprecision(4) << std::setw(12) << std::get<0>(wide_battery_candidates[i]) << " | "
                  << std::fixed << std::setprecision(3) << std::setw(10) << std::get<1>(wide_battery_candidates[i]) << " | "
                  << std::fixed << std::setprecision(2) << std::setw(12) << std::get<2>(wide_battery_candidates[i]) << " | "
                  << std::fixed << std::setprecision(3) << std::setw(10) << std::get<3>(wide_battery_candidates[i]) << " | "
                  << std::setw(15) << std::get<5>(wide_battery_candidates[i]) << " | "
                  << std::get<4>(wide_battery_candidates[i]) << "\n";
    }
    
    return 0;
}
